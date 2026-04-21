"""
LM Studio LLM Client

Endpoint strategy:
- Prefer native LM Studio v1 REST endpoints:
  - /api/v1/models
  - /api/v1/chat
- Fall back to OpenAI-compatible endpoints:
  - /v1/models
  - /v1/chat/completions

This keeps AGNN compatible with both endpoint families while using native
LM Studio routes first when possible.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import threading
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# Global inference semaphore — limits concurrent LLM calls to LM Studio.
# Default: 1 (safe/conservative). Call configure_parallel_slots() at startup
# to auto-detect LM Studio's actual parallel capacity and unlock true concurrency.
_INFERENCE_LOCK = threading.Semaphore(1)
_PARALLEL_SLOTS: int = 1  # current configured slot count
_PROBE_LOCK = threading.Lock()  # guards one-time probe logic


def _probe_response_has_content(raw: Optional[Dict]) -> bool:
    """
    Strict validation: return True only if the response contains real LLM-generated
    text, not an error message masquerading as a response body.
    """
    if not isinstance(raw, dict):
        return False
    # OpenAI-compatible shape: {"choices": [{"message": {"content": "..."}}]}
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            msg = first.get("message")
            if isinstance(msg, dict):
                content = (msg.get("content") or "").strip()
                reasoning = (msg.get("reasoning_content") or "").strip()
                tool_calls = msg.get("tool_calls")
                return len(content) > 0 or len(reasoning) > 0 or bool(tool_calls)
    # Native LM Studio shape: {"message": {"content": "..."}}
    msg = raw.get("message")
    if isinstance(msg, dict):
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        tool_calls = msg.get("tool_calls")
        return len(content) > 0 or len(reasoning) > 0 or bool(tool_calls)
    # If we got an "error" key or only a string "message", it's a failure
    return False


def configure_parallel_slots(base_url: str, model: str, max_slots: int = 4) -> int:
    """
    Probe LM Studio to discover how many concurrent requests it can handle,
    then reconfigure _INFERENCE_LOCK accordingly.

    LM Studio 0.3.x+ supports parallel inference via llama.cpp n_parallel.
    We fire N simultaneous realistic requests (not just tiny probes) and
    validate that each returns actual content — not just an HTTP 200 with
    an error body.

    Args:
        base_url:  LM Studio base URL
        model:     Any loaded model (used for probing)
        max_slots: Maximum concurrent slots to probe for (default 4)

    Returns:
        Detected parallel capacity (1 = sequential, N = truly parallel)
    """
    global _INFERENCE_LOCK, _PARALLEL_SLOTS

    with _PROBE_LOCK:
        print(f"[Parallel Probe] Testing LM Studio parallel capacity (max={max_slots})...")

        # Use a realistic payload: similar prompt length to what AGNN actually sends
        def _make_payload(idx: int) -> Dict[str, Any]:
            return {
                "model": model,
                "messages": [
                    {"role": "system",
                     "content": "You are a helpful assistant. Always respond concisely."},
                    {"role": "user",
                     "content": f"Respond with exactly one word: 'ready{idx}'."},
                ],
                "max_tokens": 10,
                "temperature": 0.0,
                "stream": False,
            }

        # Determine best working URL (prefer OpenAI-compat /v1 for broader support)
        probe_url: Optional[str] = None
        for api in _candidate_chat_api_prefixes(base_url):
            url = f"{api}/chat" if api.endswith("/api/v1") else f"{api}/chat/completions"
            try:
                raw = _http_post_json(url, _make_payload(0), timeout=20.0)
                if _probe_response_has_content(raw):
                    probe_url = url
                    break
            except Exception:
                continue

        if not probe_url:
            print("[Parallel Probe] Could not reach LM Studio — keeping Semaphore(1)")
            return 1

        print(f"[Parallel Probe] Using endpoint: {probe_url}")

        # Fire n requests simultaneously — suppress error boxes during probe
        import os as _os
        _os.environ["_AGNN_PROBE_MODE"] = "1"
        detected = 1  # default: sequential
        for n in range(2, max_slots + 1):
            raw_results:  List[Optional[Dict]] = [None] * n
            exc_results:  List[Optional[str]]  = [None] * n

            def _fire(idx: int, _n: int = n) -> None:
                try:
                    raw_results[idx] = _http_post_json(
                        probe_url, _make_payload(idx), timeout=25.0
                    )
                except Exception as exc:
                    exc_results[idx] = str(exc)

            threads = [threading.Thread(target=_fire, args=(i,)) for i in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Count only responses with actual generated text
            ok    = sum(1 for r in raw_results if _probe_response_has_content(r))
            fails = sum(1 for e in exc_results if e)
            # Also count responses that came back but had no real content (error bodies)
            bad_bodies = n - ok - fails

            print(f"[Parallel Probe] {n} concurrent → "
                  f"{ok} with content / {bad_bodies} empty body / {fails} HTTP error")

            if ok == n:          # all succeeded with real content
                detected = n
            else:
                break            # first failure level → stop

        # Reconfigure the global semaphore
        _PARALLEL_SLOTS = detected
        _INFERENCE_LOCK = threading.Semaphore(detected)

        _os.environ.pop("_AGNN_PROBE_MODE", None)  # re-enable error catalog output

        if detected > 1:
            print(f"[Parallel Probe] {detected} parallel slot(s) confirmed — "
                  f"true concurrent inference ENABLED.")
        else:
            print("[Parallel Probe] Single-slot mode (1) — "
                  "LM Studio queues requests. Enable 'Simultaneous Requests' in "
                  "LM Studio Server settings to unlock parallelism.")

        return detected


# ----------------------------
# Configuration
# ----------------------------

REQUEST_TIMEOUT_SECONDS = 180  # Increased for slow local models during draft phase
DEFAULT_MAX_TOKENS = 3000      # Increased to prevent truncation
DEFAULT_TEMPERATURE = 0.7


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class LLMResponse:
    """Response from LLM API"""
    text: str
    model: str
    tokens_used: int
    latency_seconds: float
    tool_calls: Optional[List[Dict[str, Any]]] = None


# ----------------------------
# Helpers
# ----------------------------


def _sanitize_base_url(base_url: str) -> str:
    clean = (base_url or "").strip().rstrip("/")
    if not clean:
        raise ValueError("base_url is empty")
    return clean


def _root_base_url(base_url: str) -> str:
    """Return host root by stripping '/api/v1' or '/v1' suffix if present."""
    clean = _sanitize_base_url(base_url)
    for suffix in ("/api/v1", "/v1"):
        if clean.endswith(suffix):
            return clean[: -len(suffix)].rstrip("/")
    return clean


def _candidate_api_prefixes(base_url: str) -> List[str]:
    """
    Build ordered API prefixes from a user-provided URL.

    Order policy:
    - If user explicitly passed /api/v1, keep it first.
    - If user explicitly passed /v1, keep it first.
    - If user passed bare host, prefer native /api/v1 first.
    """
    clean = _sanitize_base_url(base_url)

    if clean.endswith("/api/v1"):
        root = _root_base_url(clean)
        return [clean, f"{root}/v1"]

    if clean.endswith("/v1"):
        root = _root_base_url(clean)
        return [clean, f"{root}/api/v1"]

    return [f"{clean}/api/v1", f"{clean}/v1"]


def _candidate_chat_api_prefixes(base_url: str) -> List[str]:
    """
    Prefer the OpenAI-compatible /v1 endpoint for chat completions.

    Some LM Studio builds expose /api/v1/models successfully while still
    rejecting normal /api/v1/chat payloads that work on /v1/chat/completions.
    """
    prefixes = _candidate_api_prefixes(base_url)
    openai_compat = [api for api in prefixes if api.endswith("/v1")]
    native = [api for api in prefixes if api.endswith("/api/v1")]
    return openai_compat + native


def _filter_chat_models(model_ids: List[str]) -> List[str]:
    models: List[str] = []
    for model_id in model_ids:
        model_id = (model_id or "").strip()
        if not model_id:
            continue

        # Filter likely embedding models.
        embedding_keywords = ("embed", "embedding", "nomic", "bge", "gte", "jina", "e5-")
        if any(k in model_id.lower() for k in embedding_keywords):
            continue
        models.append(model_id)
    return models


def _extract_model_ids(response: Dict[str, Any]) -> List[str]:
    """
    Extract model IDs from responses shaped like:
    - {"data": [{"id": "..."}]}
    - {"models": [{"id": "..."}]} or {"models": ["..."]}
    """
    out: List[str] = []

    data = response.get("data")
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str):
                    out.append(model_id)

    models = response.get("models")
    if isinstance(models, list):
        for item in models:
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str):
                    out.append(model_id)
            elif isinstance(item, str):
                out.append(item)

    return out


def _get_auth_headers(url: str) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    }
    
    # Attach provider-specific auth key only.
    # This avoids sending the wrong key when multiple keys exist.
    api_key = None
    lowered = (url or "").lower()
    if "generativelanguage.googleapis.com" in lowered or "googleapis.com" in lowered:
        api_key = os.environ.get("GEMINI_API_KEY")
    elif "api.groq.com" in lowered or "groq.com" in lowered:
        api_key = os.environ.get("GROQ_API_KEY")
    elif "openai.com" in lowered:
        api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
            
    return headers


def _ssl_context_for_url(url: str) -> ssl.SSLContext:
    """
    Create SSL context. Secure by default; allow explicit opt-out for local dev.
    Set AGNN_INSECURE_TLS=1 to disable cert verification.
    """
    context = ssl.create_default_context()
    insecure_tls = os.environ.get("AGNN_INSECURE_TLS", "").strip().lower() in ("1", "true", "yes", "on")
    if insecure_tls and url.lower().startswith("https://"):
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return context

def _http_post_json(url: str, data: Dict[str, Any], timeout: float = REQUEST_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Make HTTP POST request with JSON payload and return JSON response."""
    
    # HYBRID ROUTING: Route Gemini models to the cloud endpoint regardless of base_url
    model_id = data.get("model", "")
    if isinstance(model_id, str):
        if model_id.startswith("models/gemini"):
            if "/chat/completions" in url:
                url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        elif model_id.startswith("groq/"):
            if "/chat/completions" in url:
                url = "https://api.groq.com/openai/v1/chat/completions"
            data["model"] = model_id.replace("groq/", "")

    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers=_get_auth_headers(url),
        method="POST",
    )

    context = _ssl_context_for_url(url)

    with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)


def _http_get_json(url: str, timeout: float = REQUEST_TIMEOUT_SECONDS) -> Dict[str, Any]:
    """Make HTTP GET request and return JSON response."""
    req = urllib.request.Request(url, headers=_get_auth_headers(url), method="GET")

    context = _ssl_context_for_url(url)

    with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw)


def _request_with_retries(fn, retries: int = 3, base_sleep: float = 0.6,
                          provider_hint: str = "", verbose_errors: bool = False):
    """Execute function with catalog-aware retries — knows exactly what each error means."""
    from .error_catalog import handle_error, Severity
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            err_str = str(e)
            plan = handle_error(err_str, provider_hint=provider_hint,
                                verbose=(verbose_errors and attempt == 0))

            if plan["is_fatal"]:
                raise RuntimeError(
                    f"[{plan['code']}] FATAL — {plan['cause']}\n  Fix: {plan['mitigation']}"
                ) from e

            if attempt < retries - 1:
                # Use catalog wait_secs (e.g. 25s for Groq 429) instead of tiny backoff
                catalog_wait = plan.get("wait_secs", 0.0)
                sleep_time = catalog_wait if catalog_wait > 0 else base_sleep * (2 ** attempt)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                raise RuntimeError(
                    f"Request failed after {retries} retries: {last_exc}"
                ) from last_exc
    raise RuntimeError(f"Request failed after retries: {last_exc}") from last_exc



def _extract_content_and_tokens(data: Dict[str, Any]) -> Tuple[str, Optional[int], Optional[List[Dict[str, Any]]]]:
    """Parse text/tokens and tool_calls from response shapes."""
    # OpenAI-compatible shape.
    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first, dict) else {}
        if isinstance(message, dict):
            content = (message.get("content") or "").strip()
            tool_calls = message.get("tool_calls")
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
            return content, int(total_tokens) if isinstance(total_tokens, int) else None, tool_calls

    # Native alternatives.
    message = data.get("message")
    if isinstance(message, dict):
        content = (message.get("content") or "").strip()
        tool_calls = message.get("tool_calls")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
        return content, int(total_tokens) if isinstance(total_tokens, int) else None, tool_calls

    for key in ("output_text", "text", "content"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
            return value.strip(), int(total_tokens) if isinstance(total_tokens, int) else None, None

    raise ValueError(f"No assistant text in response. Keys: {list(data.keys())}")


def _tool_schema_name(schema: Dict[str, Any]) -> str:
    if not isinstance(schema, dict):
        return ""
    fn = schema.get("function")
    if isinstance(fn, dict):
        return str(fn.get("name") or "").strip()
    return ""


def _build_tool_variants(tools: Optional[List[Dict[str, Any]]]) -> List[Optional[List[Dict[str, Any]]]]:
    if not tools:
        return [None]

    variants: List[List[Dict[str, Any]]] = [tools]
    for limit in (4, 3, 2, 1):
        if len(tools) > limit:
            variants.append(tools[:limit])

    unique: List[List[Dict[str, Any]]] = []
    seen = set()
    for variant in variants:
        names = tuple(_tool_schema_name(schema) for schema in variant)
        if names in seen:
            continue
        seen.add(names)
        unique.append(variant)
    return unique


def _looks_like_schema_rejection(error_text: str) -> bool:
    lowered = (error_text or "").lower()
    if "400" not in lowered and "schema" not in lowered and "tool" not in lowered:
        return False
    try:
        from .error_catalog import classify_error
        entry = classify_error(error_text, provider_hint="lmstudio")
        if entry and entry.code == "LMS_400_SCHEMA_REJECTION":
            return True
    except Exception:
        pass
    return "400" in lowered and any(token in lowered for token in ("tool", "schema", "bad request"))


# ----------------------------
# Public API
# ----------------------------


def list_models(base_url: str, timeout: float = 30.0) -> List[str]:
    """
    List available models from LM Studio, filtering out embedding models.
    If GEMINI_API_KEY is present, also inject Gemini models for Hybrid execution!
    """
    errors: List[str] = []
    discovered_models: List[str] = []

    import os
    if os.environ.get("GEMINI_API_KEY"):
        discovered_models.extend([
            "models/gemini-2.5-flash",
            "models/gemini-2.5-pro",
            "models/gemini-2.0-flash-lite",
            "models/gemini-2.5-flash-8b"
        ])
    if os.environ.get("GROQ_API_KEY"):
        discovered_models.extend([
            "groq/llama-3.3-70b-versatile",
            "groq/llama-3.1-8b-instant",
            "groq/gemma2-9b-it"
        ])

    for api in _candidate_api_prefixes(base_url):
        models_url = f"{api}/models"
        try:
            response = _request_with_retries(lambda: _http_get_json(models_url, timeout=timeout))
            model_ids = _extract_model_ids(response if isinstance(response, dict) else {})
            models = _filter_chat_models(model_ids)
            if models:
                discovered_models.extend(models)
                break
            errors.append(f"{models_url}: no chat models returned")
        except Exception as e:
            errors.append(f"{models_url}: {e}")

    if errors and not any(not err.endswith("no chat models returned") for err in errors):
        # Keep logs quieter for empty-model cases.
        pass
    elif errors:
        print(f"Error listing models: {' | '.join(errors)}")

    # Deduplicate while preserving order so cloud + local appear in one combined list.
    return list(dict.fromkeys(discovered_models))


def _http_stream_sse(url: str, data: Dict[str, Any],
                     timeout: float = REQUEST_TIMEOUT_SECONDS):
    """
    POST *data* to *url* with stream=True and yield decoded SSE text tokens.
    Works with LM Studio's /v1/chat/completions endpoint.
    """
    import urllib.request, ssl, json as _json
    
    # HYBRID ROUTING: Route Gemini models to the cloud endpoint regardless of base_url
    model_id = data.get("model", "")
    if isinstance(model_id, str):
        if model_id.startswith("models/gemini"):
            if "/chat/completions" in url:
                url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        elif model_id.startswith("groq/"):
            if "/chat/completions" in url:
                url = "https://api.groq.com/openai/v1/chat/completions"
            data["model"] = model_id.replace("groq/", "")
            
    body = _json.dumps(data).encode("utf-8")
    req  = urllib.request.Request(
        url, data=body,
        headers=_get_auth_headers(url),
        method="POST",
    )
    ctx = _ssl_context_for_url(url)

    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                return
            try:
                chunk = _json.loads(payload)
                choices = chunk.get("choices") or []
                if choices:
                    delta   = choices[0].get("delta") or {}
                    token   = delta.get("content") or ""
                    if token:
                        yield token
            except Exception:
                continue


def stream_chat_completion(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    base_url: str,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    on_token=None,          # Callable[[str], None] — called for each streamed token
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    retry_attempts: int = 3,
) -> "LLMResponse":
    """
    Streaming version of chat_completion.


    Calls *on_token(token_str)* for every token as it arrives, then returns
    the complete LLMResponse when the stream finishes.

    Falls back to non-streaming chat_completion if the streaming endpoint
    fails or on_token is None.
    """
    if on_token is None:
        return chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_attempts=retry_attempts,
        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    # --- Context Pre-flight Check ---
    # Fast intercept for known small-context models to prevent 40s LM Studio hangs.
    # We estimate ~4 chars per token.
    estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4
    if "gemma" in model.lower() or "lfm" in model.lower() or "1.5b" in model.lower() or "1b" in model.lower():
        if estimated_tokens > 2500:
            raise ValueError(
                f"HTTP Error 400: Bad Request (Context Pre-flight Catch: "
                f"estimated {estimated_tokens} tokens > ~2500 limit for small model '{model}')"
            )

    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        "temperature": temperature,
        "stream":      True,
    }

    # If tools are requested, we bypass streaming (reconstructing streaming tool calls is complex)
    # and delegate straight to non-streaming chat_completion.
    if tools:
        return chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            retry_attempts=retry_attempts,
        )

    start = time.time()
    _INFERENCE_LOCK.acquire()
    try:
        tokens: list = []
        # Only the OpenAI-compat endpoint supports SSE reliably
        for api in _candidate_api_prefixes(base_url):
            if api.endswith("/api/v1"):
                continue          # native endpoint doesn't do SSE well
            url = f"{api}/chat/completions"
            try:
                for tok in _http_stream_sse(url, payload, timeout=timeout):
                    tokens.append(tok)
                    on_token(tok)
                if tokens:
                    text = "".join(tokens).strip()
                    latency = time.time() - start
                    return LLMResponse(
                        text=text,
                        model=model,
                        tokens_used=len(tokens),
                        latency_seconds=latency,
                    )
            except Exception:
                break  # fall through to non-streaming

        # Fallback: non-streaming
        _INFERENCE_LOCK.release()
        return chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            base_url=base_url,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
            retry_attempts=retry_attempts,
        )

    except Exception as e:
        raise RuntimeError(f"Stream chat completion failed: {e}") from e
    finally:
        try:
            _INFERENCE_LOCK.release()
        except RuntimeError:
            pass   # already released in fallback path


def chat_completion(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    base_url: str,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
    retry_attempts: int = 3,
) -> LLMResponse:
    """
    Send chat request to LM Studio.
    Primary: /api/v1/chat
    Fallback: /v1/chat/completions
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # --- Context Pre-flight Check ---
    estimated_tokens = (len(system_prompt) + len(user_prompt)) // 4
    if "gemma" in model.lower() or "lfm" in model.lower() or "1.5b" in model.lower() or "1b" in model.lower():
        if estimated_tokens > 2500:
            raise ValueError(
                f"HTTP Error 400: Bad Request (Context Pre-flight Catch: "
                f"estimated {estimated_tokens} tokens > ~2500 limit for small model '{model}')"
            )

    start = time.time()
    _INFERENCE_LOCK.acquire()
    try:
        # --- CLOUD ROUTING INTERCEPT ---
        # If the model is a cloud model, bypass LM Studio entirely.
        cloud_url = None
        tool_variants = _build_tool_variants(tools)
        if model.startswith("groq/"):
            cloud_url = "https://api.groq.com/openai/v1/chat/completions"
        elif model.startswith("models/gemini"):
            cloud_url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

        if cloud_url:
            openai_payload: Dict[str, Any] = {
                "model": model.replace("groq/", "") if model.startswith("groq/") else model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            if tools:
                openai_payload["tools"] = tools
                openai_payload["tool_choice"] = tool_choice
            try:
                data = _request_with_retries(
                    lambda: _http_post_json(cloud_url, openai_payload, timeout=timeout),
                    retries=max(1, retry_attempts),
                    base_sleep=1.0,
                    provider_hint="groq" if model.startswith("groq/") else "gemini",
                    verbose_errors=True,
                )
                content, total_tokens, tool_calls = _extract_content_and_tokens(data if isinstance(data, dict) else {})
                if not content and not tool_calls:
                    raise ValueError("empty response content and no tool calls")
                latency = time.time() - start
                return LLMResponse(
                    text=content,
                    model=model,
                    tokens_used=int(total_tokens) if total_tokens is not None else max(0, len(content) // 4),
                    latency_seconds=latency,
                    tool_calls=tool_calls,
                )
            except Exception as cloud_exc:
                raise RuntimeError(f"Chat completion failed for {model}: {cloud_exc} (latency: {time.time()-start:.2f}s)") from cloud_exc

        final_errors: List[str] = []
        for variant in tool_variants:
            variant_errors: List[str] = []
            should_try_smaller = False

            openai_payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            native_payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False,
            }
            if variant:
                openai_payload["tools"] = variant
                openai_payload["tool_choice"] = tool_choice
                native_payload["tools"] = variant
                native_payload["tool_choice"] = tool_choice

            for api in _candidate_chat_api_prefixes(base_url):
                is_native = api.endswith("/api/v1")
                if variant and is_native:
                    # LM Studio's native /api/v1/chat endpoint rejects valid tool
                    # payloads that succeed on the OpenAI-compatible /v1 endpoint.
                    continue
                url = f"{api}/chat" if is_native else f"{api}/chat/completions"
                payload = native_payload if is_native else openai_payload

                try:
                    data = _request_with_retries(
                        lambda: _http_post_json(url, payload, timeout=timeout),
                        retries=max(1, retry_attempts),
                        base_sleep=0.5,
                        provider_hint="lmstudio",
                        verbose_errors=True,
                    )

                    if isinstance(data, dict) and "error" in data:
                        raise ValueError(f"LM Studio error: {data['error']}")

                    content, total_tokens, tool_calls = _extract_content_and_tokens(data if isinstance(data, dict) else {})
                    if not content and not tool_calls:
                        raise ValueError("empty response content and no tool calls")

                    latency = time.time() - start
                    tokens_used = int(total_tokens) if total_tokens is not None else max(0, len(content) // 4)

                    return LLMResponse(
                        text=content,
                        model=model,
                        tokens_used=tokens_used,
                        latency_seconds=latency,
                        tool_calls=tool_calls,
                    )
                except Exception as endpoint_exc:
                    err_text = f"{url}: {endpoint_exc}"
                    variant_errors.append(err_text)
                    if variant and len(variant) > 1 and _looks_like_schema_rejection(str(endpoint_exc)):
                        should_try_smaller = True

            final_errors = variant_errors
            if not should_try_smaller:
                break

        raise RuntimeError(" | ".join(final_errors))

    except Exception as e:
        latency = time.time() - start
        raise RuntimeError(f"Chat completion failed for {model}: {e} (latency: {latency:.2f}s)") from e
    finally:
        _INFERENCE_LOCK.release()
