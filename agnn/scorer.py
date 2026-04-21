"""
Auto-scorer for AGNN final deliverables.

Uses the LLM itself as a judge to score the output on 5 dimensions (0-10 each).
Falls back to a heuristic scorer if the LLM call fails.

Dimensions:
  completeness   — Did it address everything the user asked for?
  specificity    — Are there concrete details, numbers, examples? (not vague)
  coherence      — Is the document logically structured and contradiction-free?
  actionability  — Can someone act on this immediately?
  depth          — Does it go beyond surface-level observations?
"""

from __future__ import annotations

import json
import re
from typing import Dict

from .llm_client import chat_completion, LLMResponse


_JUDGE_SYSTEM = """\
You are an expert document evaluator. Score the provided document on each \
of these five dimensions from 0 to 10:

1. completeness   — Does it address all parts of the request?
2. specificity    — Concrete numbers, names, examples vs vague generalities
3. coherence      — Clear structure, no contradictions, logical flow
4. actionability  — A reader can act on it immediately
5. depth          — Goes beyond surface-level; shows real analysis

Output ONLY valid JSON, no explanation:
{"completeness":7,"specificity":6,"coherence":8,"actionability":5,"depth":6}"""


def _heuristic_score(deliverable: str, user_prompt: str) -> Dict[str, float]:
    """Fast rule-based fallback scorer when LLM judge fails."""
    words     = deliverable.split()
    wc        = len(words)
    sections  = len(re.findall(r"^#{1,3}\s+", deliverable, re.MULTILINE))
    bullets   = len(re.findall(r"^[-*•]\s+", deliverable, re.MULTILINE))
    numbers   = len(re.findall(r"\b\d+[\.,]?\d*\s*(?:%|\$|USD|B|M|K|x)?\b",
                               deliverable))
    prompt_kw = set(re.findall(r"\b[a-z]{4,}\b", user_prompt.lower()))
    doc_kw    = set(re.findall(r"\b[a-z]{4,}\b", deliverable.lower()))
    coverage  = len(prompt_kw & doc_kw) / max(len(prompt_kw), 1)

    completeness  = min(10.0, coverage * 12)
    specificity   = min(10.0, numbers * 0.4 + bullets * 0.15)
    coherence     = min(10.0, sections * 1.2 + (3.0 if wc > 400 else 0))
    actionability = min(10.0, bullets * 0.3 + (2.0 if "step" in deliverable.lower() or
                                                 "action" in deliverable.lower() else 0))
    depth         = min(10.0, wc / 80)

    scores = {
        "completeness":  round(completeness, 1),
        "specificity":   round(specificity,  1),
        "coherence":     round(coherence,    1),
        "actionability": round(actionability,1),
        "depth":         round(depth,        1),
    }
    scores["overall"] = round(sum(scores.values()) / len(scores), 1)
    return scores


def score_deliverable(
    deliverable:  str,
    user_prompt:  str,
    model:        str,
    base_url:     str,
    timeout:      float = 40.0,
) -> Dict[str, float]:
    """
    Score *deliverable* against *user_prompt* on 5 dimensions.

    Returns a dict:
        {completeness, specificity, coherence, actionability, depth, overall}
    All values are floats 0–10.

    Uses LLM-as-judge; falls back to heuristic if the LLM call fails or
    returns unparseable output.
    """
    if not deliverable or not deliverable.strip():
        return {"completeness": 0, "specificity": 0, "coherence": 0,
                "actionability": 0, "depth": 0, "overall": 0}

    user_msg = (
        f"ORIGINAL REQUEST:\n{user_prompt[:500]}\n\n"
        f"DOCUMENT TO SCORE:\n{deliverable[:2500]}"
    )

    try:
        resp: LLMResponse = chat_completion(
            system_prompt=_JUDGE_SYSTEM,
            user_prompt=user_msg,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_tokens=80,
            temperature=0.0,
        )
        raw = resp.text.strip()

        # extract first {...} block
        m = re.search(r"\{[^}]+\}", raw)
        if not m:
            raise ValueError("No JSON found in judge response")

        data = json.loads(m.group())
        dims = ["completeness", "specificity", "coherence", "actionability", "depth"]
        scores = {k: max(0.0, min(10.0, float(data.get(k, 5)))) for k in dims}
        scores["overall"] = round(sum(scores[k] for k in dims) / len(dims), 1)
        scores = {k: round(v, 1) for k, v in scores.items()}
        return scores

    except Exception:
        return _heuristic_score(deliverable, user_prompt)
