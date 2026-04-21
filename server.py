"""
AGNN streaming API (SSE).
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agnn.orchestrator import Orchestrator
from agnn.llm_client import list_models
from agnn.task_analyzer import analyze_task
from agnn.model_selector import select_models


app = FastAPI(title="AGNN API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunRequest(BaseModel):
    prompt: str
    base_url: str
    models: List[str] = []
    enable_tier2: bool = True
    auto_mode: bool = True
    max_agents: int = 4
    selection_probe: bool = True
    execution_mode: str = "balanced"


_runs: Dict[str, queue.Queue] = {}


def _emit(run_id: str, event: Dict) -> None:
    if run_id not in _runs:
        return
    _runs[run_id].put(event)



def _root_base_url(base_url: str) -> str:
    clean = (base_url or '').strip().rstrip('/')
    for suffix in ('/api/v1', '/v1'):
        if clean.endswith(suffix):
            return clean[: -len(suffix)].rstrip('/')
    return clean


def _resolve_embedding_url(base_url: str) -> str:
    # Embeddings currently use OpenAI-compatible /v1/embeddings.
    return f"{_root_base_url(base_url)}/v1"

def _normalize_model_pool(discovered: List[str], requested: List[str]) -> List[str]:
    if not requested:
        return discovered

    discovered_set = set(discovered)
    requested_existing = [m for m in requested if m in discovered_set]

    # If requested models are all unknown, use discovered list as fallback.
    if not requested_existing:
        return discovered
    return requested_existing


def _run_task(run_id: str, req: RunRequest) -> None:
    try:
        requested_models = [m.strip() for m in req.models if m and m.strip()]
        discovered = list_models(req.base_url)
        if not discovered:
            raise RuntimeError("No usable chat models found at the provided LM Studio URL.")

        if req.auto_mode:
            model_pool = discovered
        else:
            model_pool = _normalize_model_pool(discovered, requested_models)

        profiler_model = model_pool[0] if model_pool else None
        analysis = analyze_task(
            req.prompt,
            available_model_count=len(model_pool),
            base_url=req.base_url,
            profiler_model=profiler_model,
            llm_passes=3,
        )
        _emit(run_id, {
            "type": "task_analysis",
            "analysis": analysis.to_dict(),
            "timestamp": time.time(),
        })

        _emit(run_id, {
            "type": "candidate_model_pool",
            "count": len(model_pool),
            "timestamp": time.time(),
        })

        if req.auto_mode or not requested_models:
            selection = select_models(
                base_url=req.base_url,
                models=model_pool,
                analysis=analysis,
                max_agents=req.max_agents,
                enable_probe=req.selection_probe,
            )
            selected_models = selection.get("selected_models", [])
            if not selected_models:
                fallback_count = max(1, min(analysis.team_size, req.max_agents, len(model_pool)))
                selected_models = model_pool[:fallback_count]

            _emit(run_id, {
                "type": "auto_model_ranking",
                "selected_models": selected_models,
                "rationale": selection.get("rationale", []),
                "timestamp": time.time(),
            })
            _emit(run_id, {
                "type": "auto_models_selected",
                "models": selected_models,
                "timestamp": time.time(),
            })
        else:
            selected_models = model_pool

        _emit(run_id, {
            "type": "agents_initialized",
            "agents": [
                {"agent_id": f"Agent{chr(65 + i)}", "model": model}
                for i, model in enumerate(selected_models)
            ],
            "timestamp": time.time(),
        })

        orch = Orchestrator(
            base_url=req.base_url,
            models=selected_models,
            force_agent_mode=False,
            max_turns=50,
            debug=False,
            enable_tier2=req.enable_tier2,
            embedding_base_url=_resolve_embedding_url(req.base_url),
            event_callback=lambda ev: _emit(run_id, ev),
            task_analysis=analysis.__dict__,
            execution_mode=req.execution_mode,
        )
        _emit(run_id, {"type": "status", "status": "running", "timestamp": time.time()})
        orch.run(req.prompt)
        _emit(run_id, {"type": "status", "status": "completed", "timestamp": time.time()})
    except Exception as exc:
        _emit(run_id, {"type": "error", "message": str(exc), "timestamp": time.time()})
    finally:
        _emit(run_id, {"type": "done", "timestamp": time.time()})


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/models")
def models(base_url: str) -> Dict[str, List[str]]:
    if not base_url:
        raise HTTPException(status_code=400, detail="base_url required")
    return {"models": list_models(base_url)}


# ── Real model ranking (same pipeline as the actual run) ─────────────────────────────

class RankRequest(BaseModel):
    base_url: str
    prompt: str = ""
    max_agents: int = 4


@app.post("/rank-models")
def rank_models(req: RankRequest) -> Dict:
    """Run the real analyze_task + select_models pipeline and return ranked results."""
    if not req.base_url.strip():
        raise HTTPException(status_code=400, detail="base_url required")
    discovered = list_models(req.base_url)
    if not discovered:
        raise HTTPException(status_code=502, detail="No models found at LM Studio URL")

    profiler_model = discovered[0]
    try:
        analysis = analyze_task(
            req.prompt or "General multi-agent task requiring research, analysis and synthesis.",
            available_model_count=len(discovered),
            base_url=req.base_url,
            profiler_model=profiler_model,
            llm_passes=1,
        )
    except Exception:
        return {
            "ranked_models": [{"model": m, "score": round(0.9 - i * 0.05, 2), "selected": i < req.max_agents}
                               for i, m in enumerate(discovered)],
            "selected_models": discovered[:req.max_agents],
            "rationale": [],
            "task_type": "general",
            "complexity": "unknown",
        }

    try:
        selection = select_models(
            base_url=req.base_url,
            models=discovered,
            analysis=analysis,
            max_agents=req.max_agents,
            enable_probe=False,
        )
        selected = selection.get("selected_models", []) or discovered[:req.max_agents]
        rationale = selection.get("rationale", [])
    except Exception as e:
        print(f"Error selecting models: {e}")
        selected = discovered[:req.max_agents]
        rationale = []

    selected_set = set(selected)
    ranked = []
    score = 0.95
    for m in selected:
        ranked.append({"model": m, "score": round(score, 2), "selected": True})
        score -= 0.06
    for m in discovered:
        if m not in selected_set:
            ranked.append({"model": m, "score": round(max(0.05, score), 2), "selected": False})
            score -= 0.04

    return {
        "ranked_models": ranked,
        "selected_models": selected,
        "rationale": rationale,
        "task_type": getattr(analysis, "task_type", "general"),
        "complexity": str(getattr(analysis, "complexity", "unknown")),
        "team_size": getattr(analysis, "team_size", len(selected)),
        "budget": getattr(analysis, "budget", "quality_first"),
        "sub_steps": getattr(analysis, "sub_steps", []),
    }


# ── Per-model reliability probe ──────────────────────────────────────────────────────

class ProbeRequest(BaseModel):
    base_url: str
    model: str


@app.post("/probe-model")
def probe_model(req: ProbeRequest) -> Dict:
    """Quick single-shot probe: is this model responsive?"""
    import urllib.request as _ur
    import json as _json
    import time as _time
    payload = _json.dumps({
        "model": req.model,
        "messages": [{"role": "user", "content": "Reply with one word: ready"}],
        "max_tokens": 10, "temperature": 0, "stream": False,
    }).encode()
    base = req.base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    t0 = _time.time()
    try:
        res = _ur.urlopen(
            _ur.Request(f"{base}/chat/completions", data=payload,
                        headers={"Content-Type": "application/json"}, method="POST"),
            timeout=14,
        )
        res.read()
        return {"model": req.model, "ok": True, "latency_ms": int((_time.time() - t0) * 1000)}
    except Exception as exc:
        return {"model": req.model, "ok": False, "latency_ms": None, "error": str(exc)[:120]}


# ── Run management ───────────────────────────────────────────────────────────────

@app.post("/runs")
def start_run(req: RunRequest) -> Dict[str, str]:
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt required")
    if not req.base_url.strip():
        raise HTTPException(status_code=400, detail="base_url required")
    run_id = uuid.uuid4().hex
    _runs[run_id] = queue.Queue()
    thread = threading.Thread(target=_run_task, args=(run_id, req), daemon=True)
    thread.start()
    return {"run_id": run_id}


@app.get("/runs/{run_id}/stream")
def stream(run_id: str):
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail="run_id not found")

    def event_stream():
        while True:
            event = _runs[run_id].get()
            payload = json.dumps(event, ensure_ascii=False)
            yield f"data: {payload}\n\n"
            if event.get("type") == "done":
                break

    return StreamingResponse(event_stream(), media_type="text/event-stream")




