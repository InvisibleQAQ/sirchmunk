"""ImageSigner — visual fingerprint extraction.

Computes per-image features consumed by Visual Grep operators:

  Core
    1. Perceptual hash (pHash, 64-bit) — structural similarity.
    2. HSV colour moments (mean / std / skew × 3 channels = 9-d) — global
       chromatic filtering.
    3. EXIF metadata — contextual ranking.

  Enhanced (multi-operator Grep)
    4. 3×3 block pHash + per-block hue means — spatial awareness (MSBH).
    5. Center / edge HSV moments — contrastive colour analysis.
    6. GIST descriptor (Gabor-energy, 192-d) — scene-level filtering.
    7. Shannon entropy + Canny edge density — complexity pruning.

All heavy arrays (RGB → HSV, grayscale) are computed once per image and
shared across extractors.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import imagehash
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

from sirchmunk.schema.vision import ImageSignature

IMAGE_EXTENSIONS = frozenset({
    ".jpg", ".jpeg", ".png", ".bmp", ".gif",
    ".webp", ".tiff", ".tif", ".heic", ".heif",
})

# ------------------------------------------------------------------ #
# GIST constants & pre-built Gabor bank
# ------------------------------------------------------------------ #
_GIST_IMG_SIZE = 128
_GIST_GRID = 4
_GIST_CELL = _GIST_IMG_SIZE // _GIST_GRID
_GIST_SCALES = (4, 8, 16)
_GIST_N_ORIENT = 4


def _build_gabor_bank() -> List[np.ndarray]:
    """Pre-compute Gabor kernels for the GIST descriptor."""
    kernels: List[np.ndarray] = []
    for wavelength in _GIST_SCALES:
        sigma = wavelength * 0.56
        ksize = max(int(sigma * 6) | 1, 3)
        for i in range(_GIST_N_ORIENT):
            theta = i * np.pi / _GIST_N_ORIENT
            kernels.append(
                cv2.getGaborKernel(
                    (ksize, ksize), sigma, theta, wavelength, 0.5, 0,
                    ktype=cv2.CV_64F,
                )
            )
    return kernels


_GABOR_BANK: List[np.ndarray] = _build_gabor_bank()

# ------------------------------------------------------------------ #
# Standalone feature extractors (stateless, thread-safe)
# ------------------------------------------------------------------ #


def _moments_from_hsv(hsv: np.ndarray) -> List[float]:
    """9-d HSV colour moments: (mean, std, skew) per channel."""
    moments: List[float] = []
    for ch in range(3):
        channel = hsv[:, :, ch].astype(np.float64)
        mean = float(channel.mean())
        std = float(channel.std())
        skew = float(((channel - mean) ** 3).mean() / (std ** 3 + 1e-8))
        moments.extend([mean, std, skew])
    return moments


def _block_hashes(img: Image.Image, hash_size: int, grid: int = 3) -> List[str]:
    """3×3 grid of perceptual hashes for spatial awareness (MSBH)."""
    w, h = img.size
    cell_w, cell_h = w // grid, h // grid
    hashes: List[str] = []
    for row in range(grid):
        for col in range(grid):
            left = col * cell_w
            upper = row * cell_h
            right = w if col == grid - 1 else (col + 1) * cell_w
            lower = h if row == grid - 1 else (row + 1) * cell_h
            cell = img.crop((left, upper, right, lower))
            hashes.append(str(imagehash.phash(cell, hash_size=hash_size)))
    return hashes


def _block_hue_means(hsv: np.ndarray, grid: int = 3) -> List[float]:
    """Mean hue per grid cell (H channel, 0-180)."""
    h, w = hsv.shape[:2]
    cell_h, cell_w = h // grid, w // grid
    means: List[float] = []
    for row in range(grid):
        for col in range(grid):
            upper = row * cell_h
            lower = h if row == grid - 1 else (row + 1) * cell_h
            left = col * cell_w
            right = w if col == grid - 1 else (col + 1) * cell_w
            means.append(float(hsv[upper:lower, left:right, 0].mean()))
    return means


def _region_moments(hsv: np.ndarray) -> Tuple[List[float], List[float]]:
    """HSV moments for centre (inner 50 %) and edge (outer ring) regions."""
    h, w = hsv.shape[:2]
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4

    center = hsv[y0:y1, x0:x1]
    mask = np.ones((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = False

    center_m = _moments_from_hsv(center)

    edge_pixels = hsv[mask].reshape(-1, 3)
    edge_m: List[float] = []
    for ch in range(3):
        col = edge_pixels[:, ch].astype(np.float64)
        mean = float(col.mean())
        std = float(col.std())
        skew = float(((col - mean) ** 3).mean() / (std ** 3 + 1e-8))
        edge_m.extend([mean, std, skew])

    return center_m, edge_m


def _gist_descriptor(gray: np.ndarray) -> List[float]:
    """192-d GIST descriptor via multi-scale Gabor energy pooling."""
    resized = cv2.resize(gray, (_GIST_IMG_SIZE, _GIST_IMG_SIZE))
    descriptor: List[float] = []
    for kernel in _GABOR_BANK:
        response = np.abs(cv2.filter2D(resized, cv2.CV_64F, kernel))
        for gi in range(_GIST_GRID):
            for gj in range(_GIST_GRID):
                cell = response[
                    gi * _GIST_CELL : (gi + 1) * _GIST_CELL,
                    gj * _GIST_CELL : (gj + 1) * _GIST_CELL,
                ]
                descriptor.append(float(cell.mean()))
    return descriptor


def _entropy(gray: np.ndarray) -> float:
    """Shannon entropy of the grayscale histogram (bits)."""
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def _edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels flagged by Canny edge detector."""
    edges = cv2.Canny(gray, 100, 200)
    return float(np.count_nonzero(edges) / edges.size)


# ------------------------------------------------------------------ #
# ImageSigner
# ------------------------------------------------------------------ #

class ImageSigner:
    """Extract structural and chromatic fingerprints from images."""

    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size

    # ------------------------------------------------------------------ #
    # Legacy public helpers (kept for backward compatibility)
    # ------------------------------------------------------------------ #

    def compute_phash(self, image: Image.Image) -> str:
        return str(imagehash.phash(image, hash_size=self.hash_size))

    @staticmethod
    def compute_color_moments(image: Image.Image) -> List[float]:
        arr = np.array(image.convert("RGB"))
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        return _moments_from_hsv(hsv)

    @staticmethod
    def extract_exif(image_path: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        try:
            img = Image.open(image_path)
            exif_data = img.getexif()
            if not exif_data:
                return result
            _WANTED = {
                "DateTime", "DateTimeOriginal", "Model",
                "Make", "GPSInfo", "ImageWidth", "ImageLength",
            }
            for tag_id, value in exif_data.items():
                tag_name = TAGS.get(tag_id, str(tag_id))
                if tag_name in _WANTED:
                    result[tag_name] = str(value)
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def sign(self, image_path: str) -> ImageSignature:
        """Compute the full visual signature for one image.

        Opens the image once and pre-computes shared arrays (RGB, HSV,
        grayscale) to avoid redundant I/O and conversion.
        """
        img = Image.open(image_path)
        w, h = img.size

        rgb = np.array(img.convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        center_m, edge_m = _region_moments(hsv)

        return ImageSignature(
            path=os.path.abspath(image_path),
            phash=self.compute_phash(img),
            color_moments=_moments_from_hsv(hsv),
            file_size=os.path.getsize(image_path),
            width=w,
            height=h,
            exif=self.extract_exif(image_path),
            block_hashes=_block_hashes(img, self.hash_size),
            block_hue_means=_block_hue_means(hsv),
            center_moments=center_m,
            edge_moments=edge_m,
            gist_descriptor=_gist_descriptor(gray),
            entropy=_entropy(gray),
            edge_density=_edge_density(gray),
        )

    def sign_image(self, img: Image.Image, path: str = "<query>") -> ImageSignature:
        """Compute the full visual signature for an in-memory PIL Image.

        Same feature extraction as :meth:`sign` but skips file I/O.
        Useful when the image source is bytes, base64, or a URL.
        """
        w, h = img.size
        rgb = np.array(img.convert("RGB"))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        center_m, edge_m = _region_moments(hsv)
        return ImageSignature(
            path=path,
            phash=self.compute_phash(img),
            color_moments=_moments_from_hsv(hsv),
            file_size=0,
            width=w,
            height=h,
            exif={},
            block_hashes=_block_hashes(img, self.hash_size),
            block_hue_means=_block_hue_means(hsv),
            center_moments=center_m,
            edge_moments=edge_m,
            gist_descriptor=_gist_descriptor(gray),
            entropy=_entropy(gray),
            edge_density=_edge_density(gray),
        )

    def _safe_sign(self, image_path: str) -> Optional[ImageSignature]:
        """Sign a single image, returning ``None`` on failure."""
        try:
            return self.sign(image_path)
        except Exception:
            return None

    def scan_directory(
        self,
        directory: str,
        max_depth: int = 10,
        max_files: int = 50_000,
    ) -> List[ImageSignature]:
        """Recursively discover and sign all images under *directory*.

        Image signing is parallelised across multiple threads for I/O
        and CPU throughput.
        """
        print(f"    [ImageSigner] Scanning directory: {directory} (max_depth={max_depth})")
        root = Path(directory).resolve()
        image_paths: List[str] = []

        for path in sorted(root.rglob("*")):
            if len(image_paths) >= max_files:
                break
            if len(path.relative_to(root).parts) > max_depth:
                continue
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(str(path))

        if not image_paths:
            print("    [ImageSigner] No images found")
            return []

        n_workers = min(8, os.cpu_count() or 4)
        print(
            f"    [ImageSigner] Discovered {len(image_paths)} images, "
            f"signing with {n_workers} threads..."
        )
        t0 = time.time()

        signatures: List[ImageSignature] = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self._safe_sign, p): p
                for p in image_paths
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    signatures.append(result)

        elapsed = time.time() - t0
        print(
            f"    [ImageSigner] Signed {len(signatures)}/{len(image_paths)} "
            f"images in {elapsed:.1f}s"
        )
        return signatures
