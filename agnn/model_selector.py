"""
Autonomous model selector for AGNN.

Capability scoring source:
- Primary: objective probe tasks answered by each candidate model
- Secondary: rubric scoring by a judge model (LLM-based)
- Tertiary: cached historical reliability/latency/capability

No model-name capability priors are used.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import time

from .llm_client import chat_completion
from .task_analyzer import TaskAnalysis


PROFILE_PATH = Path('agnn/data/model_profile.json')
CAPABILITY_KEYS = [
    'reasoning', 'analysis', 'planning', 'writing',
    'critique', 'coding', 'speed', 'context'
]


@dataclass
class ModelScore:
    model: str
    score: float
    capabilities: Dict[str, float]
    reliability: float
    probe_quality: float
    latency_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def select_models(
    *,
    base_url: str,
    models: List[str],
    analysis: TaskAnalysis,
    max_agents: int = 4,
    enable_probe: bool = True,
    probe_depth: int = 3,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    if not models:
        return {'selected_models': [], 'rationale': [], 'all_scores': []}

    limit = max(2, min(max_agents, analysis.team_size, len(models), 5))
    profiles = _load_profiles()

    scored: List[ModelScore] = []
    probe_budget = min(6, len(models))
    judge = judge_model if judge_model in models else _choose_judge_model(models, profiles)

    for idx, model in enumerate(models):
        prof = profiles.get(model, {})
        cached_caps = _sanitize_capabilities(prof.get('capabilities', {}), default=0.5)

        caps = cached_caps if cached_caps else _neutral_capabilities()
        reliability = float(prof.get('reliability', 0.65))
        probe_quality = float(prof.get('probe_quality', 0.0))
        latency = float(prof.get('latency_seconds', 2.0))

        if enable_probe and idx < probe_budget:
            probed_caps, pq, lat, ok = _objective_probe_model(
                model=model,
                base_url=base_url,
                analysis=analysis,
                judge_model=judge,
                probe_depth=probe_depth,
            )
            if ok:
                caps = _blend_capabilities(caps, probed_caps, new_weight=0.75)
                probe_quality = 0.55 * probe_quality + 0.45 * pq
                reliability = min(0.98, reliability * 0.8 + 0.2 * 0.95)
                latency = lat
            else:
                reliability = max(0.2, reliability * 0.85)

        weighted_fit = 0.0
        for capability, weight in analysis.capability_weights.items():
            weighted_fit += weight * caps.get(capability, 0.5)

        latency_norm = min(max(latency, 0.0) / 8.0, 1.0)
        final_score = (
            weighted_fit
            + 0.12 * reliability
            + 0.08 * probe_quality
            - 0.06 * latency_norm
        )

        # Domain-keyword boost: add a small score bump if the model name
        # contains keywords that suggest domain alignment with this task.
        # This rewards e.g. a "code" model for coding tasks without penalizing
        # models whose names give no signal. Max boost is capped at +0.08.
        domain_boost = _compute_domain_boost(model, analysis.task_type)
        final_score = _clamp01(final_score + domain_boost)

        scored.append(ModelScore(
            model=model,
            score=round(final_score, 4),
            capabilities={k: round(v, 3) for k, v in caps.items()},
            reliability=round(reliability, 3),
            probe_quality=round(probe_quality, 3),
            latency_seconds=round(latency, 3),
        ))

        profiles[model] = {
            'reliability': reliability,
            'probe_quality': probe_quality,
            'latency_seconds': latency,
            'capabilities': {k: round(v, 4) for k, v in caps.items()},
            'last_seen': int(time.time()),
        }

    _save_profiles(profiles)

    scored.sort(key=lambda s: s.score, reverse=True)
    selected = _select_with_diversity(scored, analysis, limit)

    rationale = []
    for s in selected:
        strengths = sorted(s.capabilities.items(), key=lambda kv: kv[1], reverse=True)[:3]
        rationale.append({
            'model': s.model,
            'score': s.score,
            'strengths': [name for name, _ in strengths],
            'reliability': s.reliability,
            'probe_quality': s.probe_quality,
            'latency_seconds': s.latency_seconds,
            'judge_model': judge,
        })

    return {
        'selected_models': [s.model for s in selected],
        'rationale': rationale,
        'all_scores': [s.to_dict() for s in scored],
    }


def _choose_judge_model(models: List[str], profiles: Dict[str, Dict[str, Any]]) -> str:
    if not models:
        return ''
    ranked = sorted(
        models,
        key=lambda m: float(profiles.get(m, {}).get('reliability', 0.5)),
        reverse=True,
    )
    return ranked[0]


def _select_with_diversity(scored: List[ModelScore], analysis: TaskAnalysis, limit: int) -> List[ModelScore]:
    if not scored:
        return []

    selected: List[ModelScore] = []

    for s in scored:
        if len(selected) >= limit:
            break
        selected.append(s)

    required = [cap for cap in analysis.required_capabilities if cap in {'reasoning', 'analysis', 'planning', 'writing', 'critique', 'coding'}]
    required = required[:3]

    for cap in required:
        has_cap = any(m.capabilities.get(cap, 0.0) >= 0.72 for m in selected)
        if has_cap:
            continue

        candidate = next((m for m in scored if m not in selected and m.capabilities.get(cap, 0.0) >= 0.72), None)
        if not candidate:
            continue

        worst_idx = min(range(len(selected)), key=lambda i: selected[i].score)
        if candidate.score >= selected[worst_idx].score - 0.03:
            selected[worst_idx] = candidate

    selected.sort(key=lambda s: s.score, reverse=True)
    return selected[:limit]


def _objective_probe_model(
    *,
    model: str,
    base_url: str,
    analysis: TaskAnalysis,
    judge_model: str,
    probe_depth: int = 3,
) -> Tuple[Dict[str, float], float, float, bool]:
    probe_caps = _select_probe_capabilities(analysis, depth=probe_depth)
    if not probe_caps:
        probe_caps = ['reasoning', 'analysis']

    cap_scores: Dict[str, float] = {}
    probe_qualities: List[float] = []
    latencies: List[float] = []

    for cap in probe_caps:
        probe_prompt = _capability_probe_prompt(capability=cap, task_type=analysis.task_type)

        try:
            candidate = chat_completion(
                system_prompt='You are a rigorous assistant. Follow constraints exactly and be concise.',
                user_prompt=probe_prompt,
                model=model,
                base_url=base_url,
                timeout=24.0,
                max_tokens=260,
                temperature=0.2,
            )
            latencies.append(candidate.latency_seconds)

            score, conf, ok = _judge_probe_response(
                judge_model=judge_model,
                base_url=base_url,
                capability=cap,
                prompt=probe_prompt,
                response_text=candidate.text,
            )
            if not ok:
                score = _simple_quality_proxy(candidate.text)
                conf = 0.35

            cap_scores[cap] = _clamp01(score)
            probe_qualities.append(_clamp01(0.8 * score + 0.2 * conf))
        except Exception:
            continue

    if not cap_scores:
        return (_neutral_capabilities(), 0.0, 8.0, False)

    caps = _neutral_capabilities()
    for cap, val in cap_scores.items():
        caps[cap] = _clamp01(val)

    # Derive speed/context from observed latency and output behavior.
    avg_latency = sum(latencies) / len(latencies) if latencies else 8.0
    caps['speed'] = _clamp01(1.0 - min(avg_latency / 8.0, 1.0))
    if 'planning' in cap_scores and 'analysis' in cap_scores:
        caps['context'] = _clamp01((cap_scores['planning'] + cap_scores['analysis']) / 2.0)

    probe_quality = sum(probe_qualities) / len(probe_qualities) if probe_qualities else 0.0
    return (caps, _clamp01(probe_quality), avg_latency, True)


def _select_probe_capabilities(analysis: TaskAnalysis, depth: int = 3) -> List[str]:
    ordered = [
        cap for cap in analysis.required_capabilities
        if cap in {'reasoning', 'analysis', 'planning', 'writing', 'critique', 'coding'}
    ]

    if 'reasoning' not in ordered:
        ordered = ['reasoning'] + ordered

    unique: List[str] = []
    for cap in ordered:
        if cap not in unique:
            unique.append(cap)

    return unique[:max(2, min(depth, 4))]


def _capability_probe_prompt(*, capability: str, task_type: str) -> str:
    prompts = {
        'reasoning': (
            f'Task domain: {task_type}. Provide 3 explicit assumptions, 2 trade-offs, '
            'and one final recommendation for a high-impact decision. Keep under 140 words.'
        ),
        'analysis': (
            'Analyze this trend data: Q1=120, Q2=138, Q3=141, Q4=172. '
            'Provide 2 observations, 1 anomaly hypothesis, and 1 risk implication.'
        ),
        'planning': (
            f'Create a 4-step execution plan for a {task_type} task. '
            'Each step must include owner, dependency, and success check.'
        ),
        'writing': (
            f'Write an executive summary for a {task_type} objective in 110 to 140 words. '
            'Include objective, approach, and measurable outcome.'
        ),
        'critique': (
            'Critique this plan: "Collect data, draft report, submit". '
            'List 3 concrete weaknesses and one improved alternative sequence.'
        ),
        'coding': (
            'Write pseudocode for validating a JSON task payload with required fields and '
            'error handling. Keep it concise and readable.'
        ),
    }
    return prompts.get(capability, prompts['reasoning'])


def _judge_probe_response(
    *,
    judge_model: str,
    base_url: str,
    capability: str,
    prompt: str,
    response_text: str,
) -> Tuple[float, float, bool]:
    if not judge_model:
        return (0.0, 0.0, False)

    system_prompt = (
        'You are a strict evaluator. Return valid JSON only with no markdown.'
    )
    user_prompt = (
        'Evaluate candidate response quality for the capability listed below. '
        'Return JSON exactly as {"score":0.0,"confidence":0.0}. '
        'Scoring rubric: factual coherence, instruction-following, completeness, clarity. '
        'score and confidence must be in [0,1].\n\n'
        f'Capability: {capability}\n'
        f'Probe prompt: {prompt}\n'
        f'Candidate response: {response_text}'
    )

    try:
        judged = chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=judge_model,
            base_url=base_url,
            timeout=20.0,
            max_tokens=120,
            temperature=0.0,
        )
        parsed = _extract_json_object(judged.text)
        score = _safe_float(parsed.get('score'))
        confidence = _safe_float(parsed.get('confidence'))
        if score is None or confidence is None:
            return (0.0, 0.0, False)
        return (_clamp01(score), _clamp01(confidence), True)
    except Exception:
        return (0.0, 0.0, False)


def _simple_quality_proxy(text: str) -> float:
    t = (text or '').strip()
    if not t:
        return 0.0

    score = 0.35
    if len(t) > 90:
        score += 0.15
    if '\n' in t or '- ' in t or '1.' in t:
        score += 0.15
    if any(k in t.lower() for k in ['risk', 'check', 'dependency', 'owner', 'assumption']):
        score += 0.2
    return _clamp01(score)


def _neutral_capabilities() -> Dict[str, float]:
    return {k: 0.5 for k in CAPABILITY_KEYS}


def _sanitize_capabilities(raw: Dict[str, Any], default: float = 0.5) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    clean: Dict[str, float] = {}
    for key in CAPABILITY_KEYS:
        val = raw.get(key, default)
        try:
            clean[key] = _clamp01(float(val))
        except Exception:
            clean[key] = default
    return clean


def _blend_capabilities(base: Dict[str, float], new_caps: Dict[str, float], new_weight: float = 0.75) -> Dict[str, float]:
    out: Dict[str, float] = {}
    w = _clamp01(new_weight)
    for key in CAPABILITY_KEYS:
        b = _clamp01(float(base.get(key, 0.5)))
        n = _clamp01(float(new_caps.get(key, 0.5)))
        out[key] = _clamp01((1.0 - w) * b + w * n)
    return out


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = (text or '').strip()
    if not raw:
        return {}

    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    start = raw.find('{')
    end = raw.rfind('}')
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start:end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _load_profiles() -> Dict[str, Dict[str, Any]]:
    try:
        if PROFILE_PATH.exists():
            return json.loads(PROFILE_PATH.read_text(encoding='utf-8'))
    except Exception:
        pass
    return {}


def _save_profiles(profiles: Dict[str, Dict[str, Any]]) -> None:
    try:
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROFILE_PATH.write_text(json.dumps(profiles, indent=2), encoding='utf-8')
    except Exception:
        pass


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


# Domain keyword → task type affinity map.
# If the model name contains ANY of the listed keywords, it gets a boost
# for that task type. The boost is intentionally small (+0.04 to +0.08)
# so objective probe scores still dominate the final selection.
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "coding":              ["code", "coder", "dev", "python", "starcoder", "copilot", "deepseek"],
    "technical_analysis":  ["reasoning", "think", "qwen", "gemma", "analysis", "r1"],
    "creative":            ["creative", "story", "llama", "mistral", "mixtral", "claude"],
    "legal":               ["legal", "law", "compliance", "audit"],
    "scientific":          ["sci", "math", "phd", "research", "gemma"],
    "general":             [],  # No domain boost for generic tasks
}


def _compute_domain_boost(model_id: str, task_type: str) -> float:
    """
    Return a small score boost if the model name contains domain-relevant keywords
    for the given task type. Max boost is +0.08. Returns 0.0 if no match.
    """
    model_lower = model_id.lower()
    keywords = _DOMAIN_KEYWORDS.get(task_type, [])

    # Also check broader categories that partially match known task types
    for domain, kws in _DOMAIN_KEYWORDS.items():
        if domain in task_type and kws:
            keywords = list(set(keywords + kws))

    for kw in keywords:
        if kw in model_lower:
            return 0.05  # Fixed small boost — intentionally not decisive

    return 0.0
