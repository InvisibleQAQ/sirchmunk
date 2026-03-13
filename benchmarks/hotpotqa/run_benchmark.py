#!/usr/bin/env python3
"""
HotpotQA Fullwiki Benchmark for AgenticSearch
==============================================
End-to-end RAG evaluation on the HotpotQA Fullwiki setting.

Search target: global Wikipedia corpus (~5.3M article abstracts, 11GB).
Each question searches the SAME corpus — no per-question document prep.

Metrics (aligned with LinearRAG, Zhuang et al. 2025):
  - Answer EM / F1 (standard HotpotQA)
  - Contain-Match Accuracy (Contain-Acc)
  - GPT-Evaluation Accuracy (GPT-Acc)
  - Evidence Recall (title-level SP coverage)
  - Efficiency: latency, tokens, loops
  - LLM Judge for borderline cases

Config: all options from .env.hotpotqa (unified experiment config).
Optional: --env PATH to use a different env file.

Ref:
  HotpotQA paper: https://arxiv.org/pdf/1809.09600
  LinearRAG paper: https://arxiv.org/pdf/2510.10114
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

WORK_DIR = Path(__file__).resolve().parent
LOGS_DIR = WORK_DIR / "logs"


class _Tee:
    """Write to multiple streams (e.g. stdout + log file).
    
    Handles graceful degradation when streams are closed (e.g., during
    program shutdown when Loguru may still attempt to write logs).
    """

    def __init__(self, *streams):
        self._streams = streams
        self._closed = False

    def write(self, data):
        if self._closed:
            return
        for s in self._streams:
            try:
                if not s.closed:
                    s.write(data)
                    s.flush()
            except (ValueError, OSError):
                # Stream may be closed during shutdown; ignore gracefully
                pass

    def flush(self):
        if self._closed:
            return
        for s in self._streams:
            try:
                if not s.closed:
                    s.flush()
            except (ValueError, OSError):
                pass

    def close(self):
        """Mark tee as closed to prevent further writes."""
        self._closed = True


def _setup_logging():
    """Create logs dir and tee stdout/stderr to a timestamped log file.
    Returns (log_path, log_file). Caller must close log_file when done.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"benchmark_{timestamp}.log"
    log_file = open(log_path, "w", encoding="utf-8")
    return log_path, log_file


def _install_log_tee(log_file):
    """Tee stdout/stderr to log_file. Returns (orig_stdout, orig_stderr, tee_out, tee_err)."""
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    tee_out = _Tee(orig_stdout, log_file)
    tee_err = _Tee(orig_stderr, log_file)
    sys.stdout = tee_out
    sys.stderr = tee_err
    return orig_stdout, orig_stderr, tee_out, tee_err


def _restore_stdout_stderr(orig_stdout, orig_stderr):
    """Restore original stdout/stderr."""
    sys.stdout, sys.stderr = orig_stdout, orig_stderr


from config import get_config
from data_loader import load_samples, validate_wiki_corpus
from runner import run_batch
from evaluate import evaluate_predictions
from llm_judge import run_llm_judge, run_gpt_eval


_DEFAULT_ENV = Path(__file__).resolve().parent / ".env.hotpotqa"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HotpotQA Fullwiki benchmark — config from .env.hotpotqa")
    p.add_argument(
        "--env",
        type=Path,
        default=_DEFAULT_ENV,
        help=f"Path to env file (default: {_DEFAULT_ENV})",
    )
    return p.parse_args()


def _pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def print_report(metrics, results, total_time, judge_results, gpt_acc, cfg):
    """Print formatted evaluation report to stdout."""
    W = 72
    print("\n" + "=" * W)
    print("  HOTPOTQA FULLWIKI EVALUATION REPORT")
    print("=" * W)

    ov = metrics["overall"]
    print(f"\n  [Overall]  N={ov['count']}")
    print(f"    Answer EM:          {_pct(ov['ans_em'])}")
    print(f"    Answer F1:          {_pct(ov['ans_f1'])}")
    print(f"    Contain-Match Acc:  {_pct(ov['contain_acc'])}")
    print(f"    Evidence Recall:    {_pct(ov['ev_recall'])}")
    if gpt_acc is not None:
        print(f"    GPT-Eval Acc:       {_pct(gpt_acc)}")

    # By Type
    bt = metrics["by_type"]
    if bt:
        print(f"\n  [By Type]")
        hdr = f"    {'Type':<12} {'N':>5}  {'EM':>8}  {'F1':>8}  {'Cont':>8}  {'EvR':>8}"
        print(hdr)
        print(f"    {'-'*12} {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        for t in sorted(bt):
            v = bt[t]
            print(f"    {t:<12} {v['count']:>5}  "
                  f"{_pct(v['ans_em']):>8}  {_pct(v['ans_f1']):>8}  "
                  f"{_pct(v['contain_acc']):>8}  {_pct(v['ev_recall']):>8}")

    # By Level
    bl = metrics["by_level"]
    if bl:
        print(f"\n  [By Level]")
        hdr = f"    {'Level':<12} {'N':>5}  {'EM':>8}  {'F1':>8}  {'Cont':>8}  {'EvR':>8}"
        print(hdr)
        print(f"    {'-'*12} {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
        for lv in sorted(bl):
            v = bl[lv]
            print(f"    {lv:<12} {v['count']:>5}  "
                  f"{_pct(v['ans_em']):>8}  {_pct(v['ans_f1']):>8}  "
                  f"{_pct(v['contain_acc']):>8}  {_pct(v['ev_recall']):>8}")

    # Efficiency
    ok = [r for r in results if not r["error"]]
    n = len(ok) or 1
    errors = len(results) - len(ok)
    avg_time = sum(r["elapsed"] for r in ok) / n
    total_tokens = sum(r.get("telemetry", {}).get("total_tokens", 0) for r in ok)
    avg_tokens = total_tokens / n
    avg_loops = sum(
        r.get("telemetry", {}).get("loop_count", 0) for r in ok) / n
    avg_files = sum(
        len(r.get("telemetry", {}).get("files_read", [])) for r in ok) / n
    avg_titles = sum(len(r.get("retrieved_titles", [])) for r in ok) / n

    print(f"\n  [Efficiency]")
    print(f"    Avg latency:       {avg_time:.2f}s / query")
    print(f"    Avg tokens:        {avg_tokens:.0f} / query")
    print(f"    Avg loops (hops):  {avg_loops:.1f}")
    print(f"    Avg files read:    {avg_files:.1f}")
    print(f"    Avg titles retr:   {avg_titles:.0f}")
    print(f"    Total time:        {total_time:.1f}s")
    print(f"    Errors:            {errors}/{len(results)}")

    # LLM Judge
    if judge_results:
        correct = sum(1 for j in judge_results if j["llm_correct"])
        print(f"\n  [LLM Judge]  (EM=0, F1>={cfg.judge_f1_threshold})")
        print(f"    Candidates:     {len(judge_results)}")
        print(f"    Judged correct: {correct}/{len(judge_results)}")
        if ov["count"]:
            adjusted_em = (ov["ans_em"] * ov["count"] + correct) / ov["count"]
            print(f"    Adjusted EM:    {_pct(adjusted_em)}  (after LLM re-judge)")

    print("\n" + "=" * W)


async def main():
    args = parse_args()
    log_path, log_file = _setup_logging()
    orig_stdout, orig_stderr, tee_out, tee_err = _install_log_tee(log_file)
    try:
        return await _main_impl(args, log_path, log_file, orig_stdout, orig_stderr)
    finally:
        # Close tee objects first to prevent writes to closed log_file
        tee_out.close()
        tee_err.close()
        # Restore original streams before closing log file
        _restore_stdout_stderr(orig_stdout, orig_stderr)
        log_file.flush()
        log_file.close()


async def _main_impl(args, log_path, log_file, orig_stdout, orig_stderr):
    cfg = get_config(env_file=args.env)

    if not cfg.llm_api_key:
        print(f"[ERROR] LLM_API_KEY not set in {args.env}")
        return
    if not cfg.dataset_dir or not Path(cfg.dataset_dir).exists():
        print(f"[ERROR] HOTPOT_DATASET_DIR not set or missing: {cfg.dataset_dir}")
        return

    try:
        n_corpus = validate_wiki_corpus(cfg.wiki_corpus_dir)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}")
        return

    W = 72
    print("=" * W)
    print("  HotpotQA Fullwiki Benchmark — AgenticSearch")
    print("=" * W)
    print(f"  Config:      {args.env.resolve()}")
    print(f"  Wiki Corpus: {cfg.wiki_corpus_dir}")
    print(f"  Corpus Files:{n_corpus}")
    print(f"  Setting:     {cfg.setting}")
    print(f"  Split:       {cfg.split}")
    print(f"  Limit:       {cfg.limit or 'ALL'}")
    print(f"  Mode:        {cfg.mode}")
    print(f"  Top-K:       {cfg.top_k_files}")
    print(f"  Model:       {cfg.llm_model}")
    print(f"  Concurrent:  {cfg.max_concurrent}")
    print(f"  Dir Scan:    {'ON' if cfg.enable_dir_scan else 'OFF'}")
    print(f"  Extract:     {'ON' if cfg.extract_answer else 'OFF'}")
    print(f"  GPT-Eval:    {'ON' if cfg.enable_gpt_eval else 'OFF'}")
    print(f"  LLM Judge:   {'ON' if cfg.enable_llm_judge else 'OFF'}")
    print("=" * W)

    # --- Load data ---
    samples = load_samples(cfg)
    if not samples:
        print("[ERROR] No samples loaded. "
              "Check hotpotqa_dataset/{setting}/{split}*.parquet")
        return

    # --- Type/level distribution ---
    type_dist: dict = {}
    level_dist: dict = {}
    for s in samples:
        type_dist[s["type"]] = type_dist.get(s["type"], 0) + 1
        level_dist[s["level"]] = level_dist.get(s["level"], 0) + 1
    print(f"\n[data] Type distribution:  {type_dist}")
    print(f"[data] Level distribution: {level_dist}")

    # --- Run predictions ---
    print(f"\n[run] Starting evaluation on {len(samples)} questions "
          f"against wiki corpus ({n_corpus} files) ...")
    t0 = time.time()
    results = await run_batch(samples, cfg)
    total_time = time.time() - t0

    # --- Evaluate ---
    metrics = evaluate_predictions(results, samples)

    # --- GPT-Eval ---
    gpt_acc = None
    gpt_details = []
    if cfg.enable_gpt_eval and metrics["per_sample"]:
        print(f"\n[gpt-eval] Running GPT-Evaluation on "
              f"{len(metrics['per_sample'])} samples ...")
        gpt_acc, gpt_details = await run_gpt_eval(metrics["per_sample"], cfg)

    # --- LLM Judge ---
    judge_results = []
    if cfg.enable_llm_judge:
        candidates = [
            s for s in metrics["per_sample"]
            if s["em"] == 0 and s["f1"] >= cfg.judge_f1_threshold
        ]
        if candidates:
            print(f"\n[judge] Running LLM judge on "
                  f"{len(candidates)} borderline samples ...")
            judge_results = await run_llm_judge(candidates, cfg)

    # --- Report ---
    print_report(metrics, results, total_time, judge_results, gpt_acc, cfg)

    # --- Save artifacts ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = (f"{cfg.setting}_{cfg.split}_{cfg.mode}_"
           f"{cfg.llm_model}_{len(results)}q_{timestamp}")

    def _round_metric(v):
        return round(v * 100, 2) if isinstance(v, float) else v

    report = {
        "config": {
            "setting": cfg.setting, "split": cfg.split, "limit": cfg.limit,
            "mode": cfg.mode, "model": cfg.llm_model, "seed": cfg.seed,
            "top_k_files": cfg.top_k_files,
            "enable_dir_scan": cfg.enable_dir_scan,
            "extract_answer": cfg.extract_answer,
            "wiki_corpus_dir": str(cfg.wiki_corpus_dir),
        },
        "metrics": {
            "overall": {k: _round_metric(v)
                        for k, v in metrics["overall"].items()},
            "by_type": {
                t: {k: _round_metric(v) for k, v in d.items()}
                for t, d in metrics["by_type"].items()
            },
            "by_level": {
                lv: {k: _round_metric(v) for k, v in d.items()}
                for lv, d in metrics["by_level"].items()
            },
        },
        "gpt_eval": {
            "accuracy": round(gpt_acc * 100, 2) if gpt_acc is not None else None,
            "correct": sum(1 for d in gpt_details if d["llm_correct"]),
            "total": len(gpt_details),
        } if cfg.enable_gpt_eval else None,
        "efficiency": {
            "avg_latency_sec": round(
                sum(r["elapsed"] for r in results) / max(len(results), 1), 2),
            "avg_tokens": round(
                sum(r.get("telemetry", {}).get("total_tokens", 0)
                    for r in results) / max(len(results), 1)),
            "avg_loops": round(
                sum(r.get("telemetry", {}).get("loop_count", 0)
                    for r in results) / max(len(results), 1), 1),
            "total_time_sec": round(total_time, 2),
            "errors": sum(1 for r in results if r["error"]),
        },
        "llm_judge": {
            "candidates": len(judge_results),
            "correct": sum(1 for j in judge_results if j["llm_correct"]),
            "details": judge_results,
        } if judge_results else None,
        "timestamp": timestamp,
    }

    results_path = cfg.output_dir / f"results_{tag}.json"
    report_path = cfg.output_dir / f"report_{tag}.json"

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Results: {results_path}")
    print(f"  Report:  {report_path}")
    print(f"  Log:     {log_path}")


if __name__ == "__main__":
    asyncio.run(main())
