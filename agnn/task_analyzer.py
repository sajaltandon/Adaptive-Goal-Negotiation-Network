"""
Hybrid task analyzer for autonomous AGNN runs.

Primary signal:
- LLM profiler with self-consistency aggregation (multi-pass)

Fallback signal:
- heuristic analyzer when LLM profiling is unavailable or invalid
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional
import json
import re

from .llm_client import chat_completion


TASK_TYPES = {
    'coding', 'policy', 'strategy', 'technical_analysis', 'research_writing', 'general'
}

CAPABILITY_KEYS = [
    'reasoning', 'analysis', 'planning', 'writing',
    'critique', 'coding', 'speed', 'context'
]

BASE_PHASES = ['research', 'analysis', 'draft', 'review']


@dataclass
class TaskAnalysis:
    task_type: str
    complexity: str
    complexity_score: float
    team_size: int
    budget_profile: str
    required_capabilities: List[str]
    capability_weights: Dict[str, float]
    phase_plan: List[str]
    confidence: float = 0.0
    source: str = 'heuristic'
    risk_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_task(
    prompt: str,
    available_model_count: int = 0,
    base_url: Optional[str] = None,
    profiler_model: Optional[str] = None,
    llm_passes: int = 3,
) -> TaskAnalysis:
    text = (prompt or '').strip()
    heuristic = _heuristic_analysis(text, available_model_count)

    if not text:
        return heuristic

    if not base_url or not profiler_model or llm_passes <= 0:
        return heuristic

    profiles: List[Dict[str, Any]] = []
    for _ in range(max(1, llm_passes)):
        prof = _profile_task_with_llm(text, base_url=base_url, model=profiler_model)
        if prof:
            profiles.append(prof)

    if not profiles:
        return heuristic

    merged = _aggregate_profiles(profiles, fallback=heuristic)
    merged = _finalize_analysis(merged, available_model_count)
    return merged


def _heuristic_analysis(prompt: str, available_model_count: int = 0) -> TaskAnalysis:
    lower = prompt.lower()

    task_type = _detect_task_type(lower)
    complexity_score = _compute_complexity_score(lower)
    complexity = _bucket_complexity(complexity_score)

    phase_plan = list(BASE_PHASES)
    capability_weights = _weights_for_task_type(task_type)
    required_capabilities = sorted(capability_weights, key=capability_weights.get, reverse=True)

    if any(k in lower for k in ['quick', 'fast', 'brief', 'short', 'concise']):
        budget_profile = 'token_sensitive'
        capability_weights['speed'] = max(capability_weights.get('speed', 0.0), 0.18)
        _renormalize(capability_weights)
    else:
        budget_profile = 'quality_first'

    team_size = 2 if complexity == 'low' else (3 if complexity == 'medium' else 4)
    if available_model_count > 0:
        team_size = max(2, min(team_size, available_model_count, 5))

    return TaskAnalysis(
        task_type=task_type,
        complexity=complexity,
        complexity_score=round(complexity_score, 3),
        team_size=team_size,
        budget_profile=budget_profile,
        required_capabilities=required_capabilities,
        capability_weights={k: round(v, 3) for k, v in capability_weights.items()},
        phase_plan=phase_plan,
        confidence=0.55,
        source='heuristic_fallback',
        risk_flags=[],
    )


def _profile_task_with_llm(prompt: str, *, base_url: str, model: str) -> Optional[Dict[str, Any]]:
    system_prompt = (
        'You are a task profiler for multi-agent orchestration. '
        'Output valid JSON only, no markdown, no extra text.'
    )
    user_prompt = (
        'Analyze the user task and return JSON with this exact schema: '
        '{"task_type":"coding|policy|strategy|technical_analysis|research_writing|general",'
        '"complexity_score":0.0,'
        '"recommended_team_size":2,'
        '"budget_profile":"quality_first|token_sensitive",'
        '"capability_weights":{'
        '"reasoning":0.0,"analysis":0.0,"planning":0.0,"writing":0.0,'
        '"critique":0.0,"coding":0.0,"speed":0.0,"context":0.0},'
        '"phase_plan":["research","analysis","draft","review"],'
        '"risk_flags":["string"],'
        '"confidence":0.0}. '
        'Rules: capability weights must sum to 1.0; complexity_score and confidence in [0,1]; '
        'phase_plan max length 6 and must be practical for this task. '
        f'Task: {prompt}'
    )

    try:
        resp = chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            base_url=base_url,
            timeout=20.0,
            max_tokens=260,
            temperature=0.1,
        )
        parsed = _extract_json_object(resp.text)
        if not isinstance(parsed, dict):
            return None
        return parsed
    except Exception:
        return None


def _aggregate_profiles(profiles: List[Dict[str, Any]], fallback: TaskAnalysis) -> TaskAnalysis:
    task_type = _majority_label([
        str(p.get('task_type', '')).strip().lower() for p in profiles
        if str(p.get('task_type', '')).strip().lower() in TASK_TYPES
    ], default=fallback.task_type)

    complexity_values = [_safe_float(p.get('complexity_score')) for p in profiles]
    complexity_values = [v for v in complexity_values if v is not None]
    complexity_score = _clamp01(sum(complexity_values) / len(complexity_values)) if complexity_values else fallback.complexity_score
    complexity = _bucket_complexity(complexity_score)

    team_sizes = [_safe_int(p.get('recommended_team_size')) for p in profiles]
    team_sizes = [v for v in team_sizes if v is not None]
    if team_sizes:
        team_sizes.sort()
        team_size = team_sizes[len(team_sizes) // 2]
    else:
        team_size = fallback.team_size

    budget_profile = _majority_label([
        str(p.get('budget_profile', '')).strip().lower()
        for p in profiles
        if str(p.get('budget_profile', '')).strip().lower() in {'quality_first', 'token_sensitive'}
    ], default=fallback.budget_profile)

    merged_weights = _merge_capability_weights(
        [p.get('capability_weights') for p in profiles],
        fallback=_weights_for_task_type(task_type)
    )
    required = sorted(merged_weights, key=merged_weights.get, reverse=True)

    phase_plan = _merge_phase_plans([p.get('phase_plan') for p in profiles], fallback=fallback.phase_plan)

    conf_values = [_safe_float(p.get('confidence')) for p in profiles]
    conf_values = [v for v in conf_values if v is not None]
    confidence = _clamp01(sum(conf_values) / len(conf_values)) if conf_values else 0.6

    risk_flags = _merge_risk_flags([p.get('risk_flags') for p in profiles])

    return TaskAnalysis(
        task_type=task_type,
        complexity=complexity,
        complexity_score=round(complexity_score, 3),
        team_size=max(2, min(team_size, 5)),
        budget_profile=budget_profile,
        required_capabilities=required,
        capability_weights={k: round(v, 3) for k, v in merged_weights.items()},
        phase_plan=phase_plan,
        confidence=round(confidence, 3),
        source='llm_consensus',
        risk_flags=risk_flags,
    )


def _finalize_analysis(analysis: TaskAnalysis, available_model_count: int) -> TaskAnalysis:
    team_size = analysis.team_size
    if available_model_count > 0:
        team_size = max(2, min(team_size, available_model_count, 5))

    analysis.team_size = team_size
    analysis.complexity = _bucket_complexity(analysis.complexity_score)
    analysis.required_capabilities = sorted(
        analysis.capability_weights,
        key=analysis.capability_weights.get,
        reverse=True,
    )
    if not analysis.phase_plan:
        analysis.phase_plan = list(BASE_PHASES)
    return analysis


def _detect_task_type(lower: str) -> str:
    rules = {
        'coding': ['code', 'debug', 'implement', 'api', 'backend', 'frontend', 'refactor', 'algorithm'],
        'policy': ['policy', 'compliance', 'regulation', 'governance', 'standard', 'framework'],
        'strategy': ['strategy', 'roadmap', 'gtm', 'market', 'positioning', 'business plan'],
        'technical_analysis': ['root cause', 'architecture', 'performance', 'analysis', 'benchmark', 'evaluation'],
        'research_writing': ['paper', 'report', 'survey', 'literature', 'write', 'draft'],
    }
    scores = {k: 0 for k in rules}
    for kind, keywords in rules.items():
        for kw in keywords:
            if kw in lower:
                scores[kind] += 1
    best_kind = max(scores, key=scores.get)
    if scores[best_kind] == 0:
        return 'general'
    return best_kind


def _compute_complexity_score(lower: str) -> float:
    words = re.findall(r"\b\w+\b", lower)
    wlen = len(words)

    score = 0.0
    score += min(wlen / 400.0, 0.35)

    high_complexity_markers = [
        'multi-stage', 'multi step', 'end-to-end', 'with constraints',
        'tradeoff', 'evaluate', 'compare', 'benchmark', 'ablation',
        'phase', 'workflow', 'autonomous', 'governance'
    ]
    score += min(sum(1 for m in high_complexity_markers if m in lower) * 0.05, 0.35)

    structural_markers = ['1)', '2)', '3)', 'i.', 'ii.', 'iii.', ':']
    score += min(sum(1 for m in structural_markers if m in lower) * 0.02, 0.2)

    return max(0.0, min(score, 1.0))


def _weights_for_task_type(task_type: str) -> Dict[str, float]:
    templates: Dict[str, Dict[str, float]] = {
        'coding': {
            'reasoning': 0.24,
            'analysis': 0.17,
            'planning': 0.2,
            'coding': 0.23,
            'critique': 0.1,
            'writing': 0.06,
            'speed': 0.0,
            'context': 0.0,
        },
        'policy': {
            'reasoning': 0.21,
            'analysis': 0.24,
            'planning': 0.16,
            'critique': 0.18,
            'writing': 0.21,
            'coding': 0.0,
            'speed': 0.0,
            'context': 0.0,
        },
        'strategy': {
            'reasoning': 0.23,
            'analysis': 0.24,
            'planning': 0.22,
            'critique': 0.15,
            'writing': 0.16,
            'coding': 0.0,
            'speed': 0.0,
            'context': 0.0,
        },
        'technical_analysis': {
            'reasoning': 0.23,
            'analysis': 0.27,
            'planning': 0.15,
            'critique': 0.18,
            'writing': 0.14,
            'coding': 0.03,
            'speed': 0.0,
            'context': 0.0,
        },
        'research_writing': {
            'reasoning': 0.2,
            'analysis': 0.2,
            'planning': 0.14,
            'writing': 0.29,
            'critique': 0.17,
            'coding': 0.0,
            'speed': 0.0,
            'context': 0.0,
        },
        'general': {
            'reasoning': 0.23,
            'analysis': 0.22,
            'planning': 0.18,
            'writing': 0.18,
            'critique': 0.17,
            'coding': 0.02,
            'speed': 0.0,
            'context': 0.0,
        },
    }
    weights = dict(templates.get(task_type, templates['general']))
    _renormalize(weights)
    return weights


def _merge_capability_weights(items: List[Any], fallback: Dict[str, float]) -> Dict[str, float]:
    merged: Dict[str, List[float]] = {k: [] for k in CAPABILITY_KEYS}

    for raw in items:
        if not isinstance(raw, dict):
            continue
        for k in CAPABILITY_KEYS:
            val = _safe_float(raw.get(k))
            if val is not None:
                merged[k].append(_clamp01(val))

    out: Dict[str, float] = {}
    for k in CAPABILITY_KEYS:
        if merged[k]:
            out[k] = sum(merged[k]) / len(merged[k])
        else:
            out[k] = float(fallback.get(k, 0.0))

    _renormalize(out)
    return out


def _merge_phase_plans(items: List[Any], fallback: List[str]) -> List[str]:
    votes: Dict[str, int] = {}
    normalized_plans: List[List[str]] = []

    for raw in items:
        if not isinstance(raw, list):
            continue
        normalized = []
        for p in raw:
            phase = _normalize_phase(str(p))
            if phase and phase not in normalized:
                normalized.append(phase)
        if normalized:
            key = '|'.join(normalized)
            votes[key] = votes.get(key, 0) + 1
            normalized_plans.append(normalized)

    if not votes:
        return list(fallback)

    best_key = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[0][0]
    plan = best_key.split('|')

    # ensure standard closure
    for phase in BASE_PHASES:
        if phase not in plan:
            plan.append(phase)
    return plan[:6]


def _merge_risk_flags(items: List[Any]) -> List[str]:
    counts: Dict[str, int] = {}
    for raw in items:
        if not isinstance(raw, list):
            continue
        for item in raw:
            flag = str(item).strip().lower()
            if not flag:
                continue
            counts[flag] = counts.get(flag, 0) + 1

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [k for k, _ in ranked[:5]]


def _normalize_phase(raw: str) -> str:
    p = raw.strip().lower()
    if not p:
        return ''
    if any(k in p for k in ['research', 'discover', 'investigat']):
        return 'research'
    if any(k in p for k in ['analysis', 'diagnos', 'evaluate']):
        return 'analysis'
    if any(k in p for k in ['draft', 'write', 'synthes']):
        return 'draft'
    if any(k in p for k in ['review', 'validate', 'qa', 'check']):
        return 'review'
    return ''


def _bucket_complexity(score: float) -> str:
    if score < 0.34:
        return 'low'
    if score < 0.67:
        return 'medium'
    return 'high'


def _majority_label(values: List[str], default: str) -> str:
    if not values:
        return default
    votes: Dict[str, int] = {}
    for v in values:
        votes[v] = votes.get(v, 0) + 1
    return sorted(votes.items(), key=lambda kv: kv[1], reverse=True)[0][0]


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


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except Exception:
        return None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _renormalize(weights: Dict[str, float]) -> None:
    s = sum(max(0.0, float(v)) for v in weights.values())
    if s <= 0:
        return
    for k in list(weights.keys()):
        weights[k] = max(0.0, float(weights[k])) / s
