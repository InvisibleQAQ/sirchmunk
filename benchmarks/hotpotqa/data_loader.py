"""Load HotpotQA from parquet files and validate wiki corpus.

Uses duckdb for efficient parquet reading. Each row contains:
id, question, answer, type, level, supporting_facts, context.

In the fullwiki setting, AgenticSearch searches the global wiki corpus
directly — no per-question document preparation is needed.
"""

import glob
import random
from pathlib import Path
from typing import Any, Dict, List

from config import ExperimentConfig


def _load_parquet(directory: Path, split: str) -> List[Dict[str, Any]]:
    """Load all parquet files matching {split}-*.parquet from directory."""
    import duckdb

    pattern = str(directory / f"{split}-*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matching: {pattern}")

    con = duckdb.connect()
    file_list = ", ".join(f"'{f}'" for f in files)
    result = con.execute(f"SELECT * FROM read_parquet([{file_list}])")
    cols = [d[0] for d in result.description]
    return [dict(zip(cols, row)) for row in result.fetchall()]


def load_samples(cfg: ExperimentConfig) -> List[Dict[str, Any]]:
    """Load HotpotQA samples from parquet.

    Returns list of dicts: _id, question, answer, type, level,
    supporting_facts, context.
    """
    raw = _load_parquet(cfg.parquet_dir, cfg.split)

    samples = []
    for row in raw:
        samples.append({
            "_id": row["id"],
            "question": row["question"],
            "answer": row.get("answer", ""),
            "type": row.get("type", ""),
            "level": row.get("level", ""),
            "supporting_facts": row.get("supporting_facts", {}),
            "context": row.get("context", {}),
        })

    if cfg.seed is not None:
        random.Random(cfg.seed).shuffle(samples)
    if cfg.limit is not None and cfg.limit < len(samples):
        samples = samples[: cfg.limit]

    print(f"[data] Loaded {len(samples)} samples "
          f"(setting={cfg.setting}, split={cfg.split}, limit={cfg.limit})")
    return samples


def validate_wiki_corpus(wiki_dir: Path) -> int:
    """Check that the wiki corpus directory exists and has decompressed files.

    Returns the number of corpus files found.
    """
    if not wiki_dir.exists():
        raise FileNotFoundError(
            f"Wiki corpus directory not found: {wiki_dir}\n"
            "Download from: https://nlp.stanford.edu/projects/hotpotqa/"
            "enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2"
        )
    corpus_files = list(wiki_dir.rglob("wiki_*"))
    bz2_files = [f for f in corpus_files if f.suffix == ".bz2"]
    decompressed = [f for f in corpus_files if f.suffix != ".bz2"]

    if bz2_files and not decompressed:
        raise RuntimeError(
            f"Found {len(bz2_files)} compressed .bz2 files but no decompressed files.\n"
            f"Run: find {wiki_dir} -name '*.bz2' -print0 | xargs -0 -P 8 bunzip2"
        )

    if not decompressed:
        raise FileNotFoundError(
            f"No wiki files found in {wiki_dir}. "
            "Ensure corpus is downloaded and decompressed."
        )

    return len(decompressed)
