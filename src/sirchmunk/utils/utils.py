# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import math
import re
from typing import Any, Dict, List, LiteralString, Optional

from pydantic import RootModel, model_validator


class KeywordValidation(RootModel):
    root: Dict[str, float]

    @model_validator(mode="after")
    def validate_values(self) -> "KeywordValidation":
        """Ensure all keyword scores are within the range [1.0, 10.0]."""
        for k, v in self.root.items():
            self.root[k] = max(1.0, min(10.0, v))
        return self


def log_tf_norm(count: int):
    """Log normalization for term frequency."""
    return 1 + math.log(count) if count > 0 else 0


def log_tf_norm_penalty(count, ideal_range=(1, 5), penalty_alpha=0.2):
    """Refined Log Normalization with Double-Ended Penalty for Term Frequency."""
    if count <= 0:
        return 0.0

    min_t, max_t = ideal_range

    # Base Log Scale
    score = math.log(count + 1)

    # 1. Low Frequency Penalty
    if count < min_t:
        score *= count / min_t

    # 2. High Frequency Penalty
    if count > max_t:
        overage = count - max_t
        penalty = math.exp(-penalty_alpha * (overage**0.5))
        score *= penalty

    return score


def extract_fields(
    content: str, tags: Optional[List[str]] = None
) -> Dict[str, LiteralString | None]:
    """
    Extracts specified fields from the LLM output content.
        e.g. <DESCRIPTION>xxx</DESCRIPTION>, <NAME>xxx</NAME>, <CONTENT>xxx</CONTENT>.

    Args:
        content (str): The raw output content from the LLM.
        tags (Optional[List[str]]): List of tags to extract. Defaults to ["DESCRIPTION", "NAME", "CONTENT"].

    Returns:
        Dict[str, LiteralString | None]: A dictionary with extracted fields.
            Keys are the lowercase tag names, and values are the extracted content or None if not found.
    """
    # Define the list of tags to extract
    tags = tags or ["DESCRIPTION", "NAME", "CONTENT"]
    extracted_data = {}

    for tag in tags:
        # Regex Breakdown:
        # <{tag}>: Matches the opening tag
        # (.*?): Non-greedy match to capture everything inside the tags
        # </{tag}>: Matches the closing tag
        # re.DOTALL: Allows the dot (.) to match newlines, handling multi-line content
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, content, re.DOTALL)

        if match:
            # .strip() removes leading/trailing whitespace or newlines
            extracted_data[tag.lower()] = match.group(1).strip()
        else:
            # Handle cases where the LLM might miss a tag
            extracted_data[tag.lower()] = None

    return extracted_data


def parse_llm_json(text: str) -> Dict[str, Any]:
    """Extract a JSON object from an LLM response.

    Handles common LLM quirks: markdown fences, preamble text, and
    trailing commentary.  Falls back gracefully to an empty dict.

    Args:
        text: Raw LLM response that should contain a JSON object.

    Returns:
        Parsed dict, or ``{}`` if extraction fails.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass
    # Last resort: extract first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


if __name__ == "__main__":

    # --- Test Case ---
    llm_raw_output = """
    Some irrelevant preamble from the LLM...
    <DESCRIPTION>
    This document set provides a detailed overview of the installation steps for Open-Agentic-Search.
    It covers environment configuration and core dependencies.
    </DESCRIPTION>
    <NAME>Environment Setup Guide</NAME>
    <CONTENT>
    1. Install Python 3.10+
    2. Run pip install -r requirements.txt
    3. Configure the .env environment variables.
    </CONTENT>
    """

    result = extract_fields(llm_raw_output)

    # Print results
    print(f"Name: {result['name']}")
    print(f"Description: {result['description']}")
    print(f"Content: \n{result['content']}")
