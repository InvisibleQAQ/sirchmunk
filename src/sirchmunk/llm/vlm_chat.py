"""OpenAI-compatible VLM client with multimodal (text + image) support.

Default model: ``qwen3.5-plus`` via the standard OpenAI Chat Completions
API.  Supports local file paths, HTTP URLs, and in-memory PIL images as
image inputs.

This client uses ``httpx`` directly (rather than the ``openai`` SDK)
because multimodal payloads require fine-grained control over the
``content`` array structure.

Environment variables (all optional, overridden by constructor args):
    VLM_BASE_URL  – API base URL   (default: DashScope compatible endpoint)
    VLM_API_KEY   – Bearer token
    VLM_MODEL     – Model name     (default: qwen3.5-plus)
"""

from __future__ import annotations

import base64
import io
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx


@dataclass
class VLMResponse:
    """Structured response returned by :meth:`VLMClient.achat`."""

    content: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = ""


class VLMClient:
    """OpenAI-compatible multimodal chat client.

    Builds the ``content`` array with interleaved ``text`` and ``image_url``
    blocks as defined by the OpenAI vision API specification.
    """

    _DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    _DEFAULT_MODEL = "qwen3.5-plus"

    def __init__(
        self,
        base_url: str = "",
        api_key: str = "",
        model: str = "",
        timeout: float = 120.0,
        max_tokens: int = 40960,
    ):
        self.base_url = (
            base_url
            or os.getenv("VLM_BASE_URL", self._DEFAULT_BASE_URL)
        ).rstrip("/")
        self.api_key = api_key or os.getenv("VLM_API_KEY", "")
        self.model = model or os.getenv("VLM_MODEL", self._DEFAULT_MODEL)
        self.timeout = timeout
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------ #
    # Image encoding helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def encode_image_base64(image_path: str) -> str:
        """Read a local image file and return a ``data:<mime>;base64,...`` URI."""
        mime, _ = mimetypes.guess_type(image_path)
        if mime is None:
            mime = "image/jpeg"
        with open(image_path, "rb") as fh:
            b64 = base64.b64encode(fh.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    @staticmethod
    def encode_pil_image(image, fmt: str = "JPEG", quality: int = 85) -> str:
        """Encode an in-memory PIL Image to a base64 data URI."""
        buf = io.BytesIO()
        image.save(buf, format=fmt, quality=quality)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/{fmt.lower()};base64,{b64}"

    @classmethod
    def build_image_content(cls, image: Any) -> Dict[str, Any]:
        """Build an ``image_url`` content block.

        Accepts:
            - ``str`` / ``Path``: local path or HTTP(S) URL
            - ``PIL.Image.Image``: in-memory image
        """
        try:
            from PIL import Image as PILImage
            if isinstance(image, PILImage.Image):
                return {
                    "type": "image_url",
                    "image_url": {"url": cls.encode_pil_image(image)},
                }
        except ImportError:
            pass

        image_str = str(image)
        if image_str.startswith(("http://", "https://", "data:")):
            url = image_str
        else:
            url = cls.encode_image_base64(image_str)
        return {"type": "image_url", "image_url": {"url": url}}

    @classmethod
    def build_user_message(
        cls,
        text: str,
        images: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Build a ``user`` role message with text and optional images."""
        content: List[Dict[str, Any]] = []
        for img in images or []:
            content.append(cls.build_image_content(img))
        content.append({"type": "text", "text": text})
        return {"role": "user", "content": content}

    # ------------------------------------------------------------------ #
    # API call
    # ------------------------------------------------------------------ #

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        messages: List[Dict],
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": kwargs.pop("model", self.model),
            "messages": messages,
            "max_tokens": kwargs.pop("max_tokens", self.max_tokens),
            "temperature": temperature,
            "stream": False,
            "enable_thinking": kwargs.pop("enable_thinking", False),
        }
        body.update(kwargs)
        return body

    @staticmethod
    def _parse(data: Dict[str, Any]) -> VLMResponse:
        choice = data.get("choices", [{}])[0]
        return VLMResponse(
            content=choice.get("message", {}).get("content", ""),
            usage=data.get("usage", {}),
            model=data.get("model", ""),
            finish_reason=choice.get("finish_reason", ""),
        )

    async def achat(
        self,
        messages: List[Dict],
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> VLMResponse:
        """Async (non-streaming) chat completion."""
        payload = self._payload(messages, temperature, **kwargs)
        n_images = sum(
            1 for m in messages for c in (m.get("content") or [])
            if isinstance(c, dict) and c.get("type") == "image_url"
        )
        print(
            f"      [VLMClient] achat → model={self.model}, "
            f"images={n_images}, temp={temperature}"
        )
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            result = self._parse(resp.json())
            print(
                f"      [VLMClient] achat ← {len(result.content)} chars, "
                f"usage={result.usage}, finish={result.finish_reason}"
            )
            return result

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> VLMResponse:
        """Synchronous chat completion."""
        payload = self._payload(messages, temperature, **kwargs)
        n_images = sum(
            1 for m in messages for c in (m.get("content") or [])
            if isinstance(c, dict) and c.get("type") == "image_url"
        )
        print(
            f"      [VLMClient] chat → model={self.model}, "
            f"images={n_images}, temp={temperature}"
        )
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            result = self._parse(resp.json())
            print(
                f"      [VLMClient] chat ← {len(result.content)} chars, "
                f"usage={result.usage}, finish={result.finish_reason}"
            )
            return result
