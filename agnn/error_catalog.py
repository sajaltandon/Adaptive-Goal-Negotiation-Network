"""
AGNN API Error Catalog  ·  v3.0
─────────────────────────────────
Centralised knowledge base for every known error from:
  - Groq Cloud API
  - Gemini Cloud API
  - LM Studio (local)

Each error class provides:
  - Detection (string/code matchers)
  - Severity (FATAL / RETRY / DEGRADE / WARN)
  - Root cause explanation
  - Automatic mitigation strategy
"""

from __future__ import annotations

import os
import time
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

# ── Severity levels ──────────────────────────────────────────────────────────

class Severity(Enum):
    FATAL   = "FATAL"    # cannot continue this run
    RETRY   = "RETRY"    # transient — wait and retry
    DEGRADE = "DEGRADE"  # reduce payload and retry
    WARN    = "WARN"     # log and continue


@dataclass
class ErrorEntry:
    code: str
    provider: str
    severity: Severity
    cause: str
    mitigation: str
    wait_secs: float = 0.0
    matchers: List[str] = field(default_factory=list)


# ── The Error Catalog ─────────────────────────────────────────────────────────

CATALOG: List[ErrorEntry] = [

    # ── GROQ ──────────────────────────────────────────────────────────────────

    ErrorEntry(
        code="GROQ_429_RATE_LIMIT",
        provider="groq",
        severity=Severity.RETRY,
        cause="Groq free tier: ~14,400 tokens/min. Hit when AGNN sends too many turns too fast.",
        mitigation="Wait 20-60s and retry. Use llama-3.1-8b-instant for cheaper tasks to save quota.",
        wait_secs=25.0,
        matchers=["429", "rate_limit_exceeded", "Rate limit", "Too Many Requests"],
    ),
    ErrorEntry(
        code="GROQ_400_CONTEXT_TOO_LONG",
        provider="groq",
        severity=Severity.DEGRADE,
        cause="The combined system_prompt + messages exceeds the model's context window (32k for Llama-70B).",
        mitigation="Shrink system_prompt to 1500 chars. Reduce max_tokens to 600. Clear message history.",
        wait_secs=0.0,
        matchers=["400", "context_length_exceeded", "maximum context", "too many tokens"],
    ),
    ErrorEntry(
        code="GROQ_400_BAD_SCHEMA",
        provider="groq",
        severity=Severity.DEGRADE,
        cause="Tool schema sent to Groq is malformed or has unsupported fields.",
        mitigation="Reduce tool list to just [bash]. Remove 'default' fields from schema properties.",
        wait_secs=0.0,
        matchers=["400", "Bad Request", "invalid_request_error", "tool"],
    ),
    ErrorEntry(
        code="GROQ_401_AUTH",
        provider="groq",
        severity=Severity.FATAL,
        cause="GROQ_API_KEY is missing, expired, or invalid.",
        mitigation="Set GROQ_API_KEY in run_with_groq.bat. Verify key at console.groq.com.",
        wait_secs=0.0,
        matchers=["401", "Unauthorized", "invalid_api_key", "Authentication"],
    ),
    ErrorEntry(
        code="GROQ_403_CLOUDFLARE",
        provider="groq",
        severity=Severity.RETRY,
        cause="Cloudflare 1010 geo-block. Groq API blocked your IP/region without User-Agent header.",
        mitigation="Browser User-Agent header is already injected. If this persists, retry in 30s.",
        wait_secs=30.0,
        matchers=["403", "1010", "Cloudflare", "Access denied"],
    ),
    ErrorEntry(
        code="GROQ_503_OVERLOADED",
        provider="groq",
        severity=Severity.RETRY,
        cause="Groq servers overloaded — common during peak hours (UTC 14:00-20:00).",
        mitigation="Wait 15s and retry. Switch to llama-3.1-8b-instant which has higher availability.",
        wait_secs=15.0,
        matchers=["503", "Service Unavailable", "overloaded", "server_error"],
    ),
    ErrorEntry(
        code="GROQ_TIMEOUT",
        provider="groq",
        severity=Severity.RETRY,
        cause="Request timed out. Groq response took longer than the configured timeout.",
        mitigation="Increase timeout to 120s for heavy prompts. Reduce max_tokens.",
        wait_secs=5.0,
        matchers=["timed out", "timeout", "TimeoutError"],
    ),

    # ── GEMINI ────────────────────────────────────────────────────────────────

    ErrorEntry(
        code="GEMINI_429_QUOTA",
        provider="gemini",
        severity=Severity.RETRY,
        cause="Gemini free quota exceeded. Free tier: 15 RPM / 1M TPD for Flash, 2 RPM for Pro.",
        mitigation="Wait 60s. Switch to gemini-2.0-flash-lite which has highest free quota.",
        wait_secs=60.0,
        matchers=["429", "RESOURCE_EXHAUSTED", "quota", "Quota exceeded"],
    ),
    ErrorEntry(
        code="GEMINI_400_SAFETY",
        provider="gemini",
        severity=Severity.DEGRADE,
        cause="Gemini safety filters blocked the request (content policy violation).",
        mitigation="Rephrase the task prompt to be less ambiguous. Avoid words like 'delete', 'hack', 'exploit'.",
        wait_secs=0.0,
        matchers=["400", "SAFETY", "safety_settings", "blocked", "HARM"],
    ),
    ErrorEntry(
        code="GEMINI_400_CONTEXT",
        provider="gemini",
        severity=Severity.DEGRADE,
        cause="Input exceeds Gemini's context window (1M tokens for Flash, 2M for Pro).",
        mitigation="Truncate the accumulated message content before sending.",
        wait_secs=0.0,
        matchers=["400", "context", "token", "INVALID_ARGUMENT"],
    ),
    ErrorEntry(
        code="GEMINI_401_AUTH",
        provider="gemini",
        severity=Severity.FATAL,
        cause="GEMINI_API_KEY missing or invalid.",
        mitigation="Set GEMINI_API_KEY in run_with_gemini.bat. Get key at aistudio.google.com.",
        wait_secs=0.0,
        matchers=["401", "API_KEY_INVALID", "API key not valid", "UNAUTHENTICATED"],
    ),
    ErrorEntry(
        code="GEMINI_503_CAPACITY",
        provider="gemini",
        severity=Severity.RETRY,
        cause="Gemini model temporarily unavailable — image generation model especially prone.",
        mitigation="Wait 10s and retry with a different model variant.",
        wait_secs=10.0,
        matchers=["503", "MODEL_CAPACITY_EXHAUSTED", "UNAVAILABLE"],
    ),

    # ── LM STUDIO ─────────────────────────────────────────────────────────────

    ErrorEntry(
        code="LMS_CONNECTION_REFUSED",
        provider="lmstudio",
        severity=Severity.FATAL,
        cause="LM Studio server is not running or not listening on the configured port.",
        mitigation="Start LM Studio → Server tab → click Start. Confirm port matches (default 1234).",
        wait_secs=0.0,
        matchers=["Connection refused", "ConnectionRefusedError", "Connect call failed"],
    ),
    ErrorEntry(
        code="LMS_400_SCHEMA_REJECTION",
        provider="lmstudio",
        severity=Severity.DEGRADE,
        cause="Small model rejected the tool JSON schema (too large or unsupported fields).",
        mitigation="Reduce to 1-2 tools only. Remove 'default' fields. Adaptive Tool Surface now handles this.",
        wait_secs=0.0,
        matchers=["400", "Bad Request", "schema", "tool"],
    ),
    ErrorEntry(
        code="LMS_MODEL_NOT_LOADED",
        provider="lmstudio",
        severity=Severity.FATAL,
        cause="LM Studio has no model loaded in memory.",
        mitigation="Open LM Studio → Models tab → select and load a model before starting AGNN.",
        wait_secs=0.0,
        matchers=["No model loaded", "model not found", "404", "no model"],
    ),
    ErrorEntry(
        code="LMS_TIMEOUT_LAG",
        provider="lmstudio",
        severity=Severity.RETRY,
        cause="Model inference is too slow (large model on CPU, or GPU VRAM overflow causing swap).",
        mitigation="Use a smaller quantized model (Q4_K_M). Enable GPU offloading in LM Studio settings. Or use Groq instead.",
        wait_secs=5.0,
        matchers=["timed out", "timeout", "TimeoutError", "Read timed out"],
    ),
    ErrorEntry(
        code="LMS_QUEUE_LAG",
        provider="lmstudio",
        severity=Severity.WARN,
        cause="LM Studio is processing requests sequentially. 'Simultaneous Requests' not enabled.",
        mitigation="LM Studio → Server settings → Enable Simultaneous Requests. Or use Groq for parallel tasks.",
        wait_secs=0.0,
        matchers=["Single-slot", "queue", "queued"],
    ),
    ErrorEntry(
        code="LMS_OOM",
        provider="lmstudio",
        severity=Severity.FATAL,
        cause="GPU/CPU ran out of memory during inference.",
        mitigation="Use a smaller model or lower context window in LM Studio settings.",
        wait_secs=0.0,
        matchers=["out of memory", "OOM", "CUDA error", "malloc failed"],
    ),
]


# ── Lookup engine ──────────────────────────────────────────────────────────────

def classify_error(error_str: str, provider_hint: str = "") -> Optional[ErrorEntry]:
    """
    Match an error string against the catalog.
    Returns the most specific ErrorEntry or None if unknown.
    """
    err_lower = error_str.lower()
    best: Optional[ErrorEntry] = None
    best_score = 0

    for entry in CATALOG:
        # Filter by provider hint if given
        if provider_hint and entry.provider not in ("any", provider_hint):
            # Still check but score lower
            multiplier = 0.5
        else:
            multiplier = 1.0

        score = 0
        for matcher in entry.matchers:
            if matcher.lower() in err_lower:
                score += multiplier

        if score > best_score:
            best_score = score
            best = entry

    return best if best_score > 0 else None


def handle_error(error_str: str, provider_hint: str = "", verbose: bool = True) -> dict:
    """
    Classify an error and return a mitigation plan dict:
    {
        'code': str,
        'severity': Severity,
        'cause': str,
        'mitigation': str,
        'wait_secs': float,
        'should_retry': bool,
        'should_degrade': bool,
        'is_fatal': bool,
    }
    """
    entry = classify_error(error_str, provider_hint)

    if entry is None:
        if verbose and not os.environ.get("_AGNN_PROBE_MODE"):
            print(f"  [ErrorCatalog] UNKNOWN error: {error_str[:120]}")
        return {
            "code": "UNKNOWN",
            "severity": Severity.RETRY,
            "cause": "Unclassified error",
            "mitigation": "Check logs. Retry once. If persists, escalate to cloud model.",
            "wait_secs": 5.0,
            "should_retry": True,
            "should_degrade": False,
            "is_fatal": False,
        }

    is_probe = os.environ.get("_AGNN_PROBE_MODE")

    if verbose and not is_probe:
        if entry.severity == Severity.FATAL:
            # Full box — user must act
            print(f"\n  ┌─ [ErrorCatalog] ──────────────────────────────────────")
            print(f"  │  Code     : {entry.code}")
            print(f"  │  Severity : {entry.severity.value}")
            print(f"  │  Cause    : {entry.cause}")
            print(f"  │  Fix      : {entry.mitigation}")
            print(f"  └──────────────────────────────────────────────────────\n")
        elif entry.severity == Severity.RETRY:
            # Compact box — transient, shows wait time
            print(f"\n  ┌─ [ErrorCatalog] ──────────────────────────────────────")
            print(f"  │  Code     : {entry.code}")
            print(f"  │  Severity : {entry.severity.value}")
            print(f"  │  Cause    : {entry.cause}")
            if entry.wait_secs > 0:
                print(f"  │  Waiting  : {entry.wait_secs}s before retry")
            print(f"  └──────────────────────────────────────────────────────\n")
        elif entry.severity == Severity.DEGRADE:
            # Silent self-heal — just a quiet inline note, no box
            print(f"  [auto-degrade] {entry.code} — reducing tool surface and retrying.")
        # WARN: fully silent

    # Wait if needed
    if entry.wait_secs > 0:
        time.sleep(entry.wait_secs)

    return {
        "code": entry.code,
        "severity": entry.severity,
        "cause": entry.cause,
        "mitigation": entry.mitigation,
        "wait_secs": entry.wait_secs,
        "should_retry": entry.severity in (Severity.RETRY, Severity.WARN),
        "should_degrade": entry.severity == Severity.DEGRADE,
        "is_fatal": entry.severity == Severity.FATAL,
    }
