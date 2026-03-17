"""LLM-based evaluation: semantic equivalence judge + GPT-Eval accuracy.

Two evaluation modes:
1. LLM Judge (borderline): For samples where EM=0 but F1 >= threshold,
   checks semantic equivalence. Catches cross-language, abbreviation,
   and superset-answer cases.
2. GPT-Eval (all samples): LinearRAG-style LLM-based accuracy check
   applied to every sample. Reports GPT-Evaluation Accuracy.

Both use OpenAIChat (same provider as .env.hotpotqa).
Ref: LinearRAG (Zhuang et al., 2025) §4.1 — GPT-Evaluation Accuracy
"""

import asyncio
from typing import Any, Dict, List, Tuple

from config import ExperimentConfig

_JUDGE_SYSTEM = (
    "You are a strict answer equivalence judge for question-answering evaluation."
)

_JUDGE_TEMPLATE = """\
Determine if the PREDICTED answer is semantically equivalent to the GOLD answer \
for the given question.

Question: {question}
Gold Answer: {gold}
Predicted Answer: {prediction}

Consider:
- Different languages for the same meaning \
(e.g. "2008年北京奥运会" ↔ "2008 Summer Olympics in Beijing")
- Abbreviations or alternative names for the same entity
- Answers containing the gold answer with no contradictory information
- Minor formatting or phrasing differences

Reply with EXACTLY one word: CORRECT or INCORRECT"""

_GPT_EVAL_TEMPLATE = """\
Given the question, determine if the predicted answer is correct by comparing \
it to the gold answer.

Question: {question}
Gold Answer: {gold}
Predicted Answer: {prediction}

Consider: same meaning in different wording, abbreviations, minor formatting \
differences, or answers that fully contain the gold answer.

Reply with EXACTLY one word: CORRECT or INCORRECT"""


async def _eval_single(
    sample: Dict[str, Any],
    llm: Any,
    semaphore: asyncio.Semaphore,
    delay: float,
    template: str,
    system_msg: str = "",
) -> Dict[str, Any]:
    """Evaluate one sample via LLM. Returns verdict dict."""
    prompt = template.format(
        question=sample.get("question", ""),
        gold=sample["gold"],
        prediction=sample["pred"],
    )
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})

    async with semaphore:
        try:
            resp = await llm.achat(messages=messages, stream=False)
            verdict = resp.content.strip().split()[0].upper()
            is_correct = verdict == "CORRECT"
        except Exception as e:
            verdict = f"ERROR: {e}"
            is_correct = False
        if delay > 0:
            await asyncio.sleep(delay)

    return {
        "_id": sample["_id"],
        "gold": sample["gold"],
        "pred": sample["pred"],
        "llm_verdict": verdict,
        "llm_correct": is_correct,
    }


async def run_llm_judge(
    candidates: List[Dict[str, Any]],
    cfg: ExperimentConfig,
) -> List[Dict[str, Any]]:
    """Run LLM judge on candidate samples (EM=0, F1 >= threshold)."""
    if not candidates:
        return []

    from sirchmunk.llm.openai_chat import OpenAIChat

    llm = OpenAIChat(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
    )
    semaphore = asyncio.Semaphore(cfg.max_concurrent)
    tasks = [
        _eval_single(s, llm, semaphore, cfg.request_delay,
                      _JUDGE_TEMPLATE, _JUDGE_SYSTEM)
        for s in candidates
    ]
    return list(await asyncio.gather(*tasks))


async def run_gpt_eval(
    per_sample: List[Dict[str, Any]],
    cfg: ExperimentConfig,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Run GPT-based evaluation on all samples (LinearRAG-style GPT-Acc).

    Uses ``cfg.max_concurrent`` to parallelise LLM calls and
    ``cfg.request_delay`` between requests to stay within rate limits.

    Returns (accuracy, details_list).
    """
    if not per_sample:
        return 0.0, []

    from sirchmunk.llm.openai_chat import OpenAIChat

    llm = OpenAIChat(
        api_key=cfg.llm_api_key,
        base_url=cfg.llm_base_url,
        model=cfg.llm_model,
    )
    semaphore = asyncio.Semaphore(cfg.max_concurrent)
    tasks = [
        _eval_single(s, llm, semaphore, cfg.request_delay, _GPT_EVAL_TEMPLATE)
        for s in per_sample
    ]
    details = list(await asyncio.gather(*tasks))
    correct = sum(1 for d in details if d["llm_correct"])
    acc = correct / len(details) if details else 0.0
    return acc, details
