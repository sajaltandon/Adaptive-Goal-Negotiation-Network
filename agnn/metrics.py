"""
AGNN Tier-0 Metrics System

Implements comprehensive message scoring including:
- Traditional metrics (Relevance, Novelty, Actionability, Information Gain)
- Enhanced Tier-0 metrics (TIS components: SD, RC, IS, EIC, St)
- Hashed BOW embeddings for semantic analysis
- Information Flow Control Matrix (IFCM)

Enhanced with Week 2 improvements for better artifact detection.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set, Tuple
import hashlib
import math
import re
import threading
from collections import Counter

from .config import TISWeights, tis_weights

import urllib.request
import json
import ssl

# Global embedding cache to prevent re-fetching same text
_EMBEDDING_CACHE: dict = {}
# Per-host embedding model cache: {host_root -> model_id or None}
_EMBEDDING_MODEL_CACHE: dict = {}

# Keywords that identify an embedding model (same set as llm_client filter)
_EMBEDDING_KEYWORDS = ("embed", "embedding", "nomic", "bge", "gte", "jina", "e5-")

# Suppress repeated embedding-fail noise: only print the first failure per session
_embed_fail_lock = threading.Lock()
_embed_fail_printed = False


def _host_root(base_url: str) -> str:
    """Strip /api/v1 or /v1 suffix and return bare host URL."""
    clean = (base_url or "").rstrip("/")
    for suffix in ("/api/v1", "/v1"):
        if clean.endswith(suffix):
            return clean[: -len(suffix)]
    return clean


def _discover_embedding_model(base_url: str) -> Optional[str]:
    """
    Query /v1/models once per host, pick the first embedding model,
    and cache the result. LM Studio JIT loading requires the model
    field in the embeddings payload — without it the server returns
    'input' is required (invalid_union validation error).
    """
    host = _host_root(base_url)
    if host in _EMBEDDING_MODEL_CACHE:
        return _EMBEDDING_MODEL_CACHE[host]

    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(f"{host}/v1/models", method="GET")
        with urllib.request.urlopen(req, timeout=5, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))

        for item in data.get("data", []):
            model_id = item.get("id", "") if isinstance(item, dict) else str(item)
            if any(k in model_id.lower() for k in _EMBEDDING_KEYWORDS):
                print(f"[Metrics] Using embedding model: {model_id}")
                _EMBEDDING_MODEL_CACHE[host] = model_id
                return model_id
    except Exception as e:
        print(f"[Metrics] Embedding model discovery failed: {e}")

    _EMBEDDING_MODEL_CACHE[host] = None
    return None


def _call_embedding_api(text: str, base_url: str) -> Optional[List[float]]:
    """
    Call LM Studio /v1/embeddings.

    LM Studio's JIT loader requires BOTH 'input' and 'model' fields.
    We discover the correct embedding model from /v1/models once per
    session (cached) so we never hardcode a model name.
    """
    if text in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[text]

    try:
        host = _host_root(base_url)
        url = f"{host}/v1/embeddings"

        # Always include the model field — LM Studio JIT loading requires it.
        embedding_model = _discover_embedding_model(base_url)
        payload_dict: dict = {"input": text}
        if embedding_model:
            payload_dict["model"] = embedding_model

        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(req, timeout=5, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))

        if "data" in data and data["data"]:
            embedding = data["data"][0].get("embedding")
            if embedding:
                _EMBEDDING_CACHE[text] = embedding
                return embedding
    except Exception as e:
        global _embed_fail_printed
        with _embed_fail_lock:
            if not _embed_fail_printed:
                print(f"[Metrics] Embedding unavailable ({e}) — falling back to heuristic SD.")
                _embed_fail_printed = True

    return None

def _get_embedding(text: str, base_url: str = None, dim: int = 384) -> List[float]:
    """
    Generate embedding for text.
    Tries LM Studio API first (if base_url provided), falls back to Hashed BOW.
    """
    if not text.strip():
        return [0.0] * dim
        
    # Try API if base_url is available
    if base_url:
        embedding = _call_embedding_api(text, base_url)
        if embedding:
            return embedding
    
    # Fallback to Hashed BOW
    return _hash_bow_embedding_fallback(text, dim)

def _hash_bow_embedding_fallback(text: str, dim: int = 384) -> List[float]:
    """Fallback Hashed-BOW implementation"""
    # Tokenize and clean
    words = re.findall(r'\b[a-z]+\b', text.lower())
    if not words:
        return [0.0] * dim
    
    # Count words and hash to buckets
    word_counts = Counter(words)
    embedding = [0.0] * dim
    
    for word, count in word_counts.items():
        # Hash word to bucket index
        hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
        bucket = hash_val % dim
        embedding[bucket] += count
    
    # L2 normalize
    norm = math.sqrt(sum(x * x for x in embedding))
    if norm > 0:
        embedding = [x / norm for x in embedding]
    
    return embedding


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def _jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts"""
    words1 = set(re.findall(r'\b[a-z]+\b', text1.lower()))
    words2 = set(re.findall(r'\b[a-z]+\b', text2.lower()))
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    return intersection / union if union > 0 else 0.0


def _relevance_embedding(candidate_text: str, context_messages: List[str], base_url: str = None) -> float:
    """Calculate relevance using embeddings"""
    if not context_messages:
        return 0.5  # Neutral relevance for first message
    
    candidate_emb = _get_embedding(candidate_text, base_url)
    
    # Calculate similarity with recent context (last 3 messages)
    recent_context = context_messages[-3:]
    similarities = []
    
    for msg in recent_context:
        msg_emb = _get_embedding(msg, base_url)
        sim = _cosine_similarity(candidate_emb, msg_emb)
        similarities.append(sim)
    
    # Return max similarity (most relevant connection)
    return max(similarities) if similarities else 0.5


def _relevance_lexical(candidate_text: str, context_messages: List[str]) -> float:
    """Calculate lexical relevance using keyword overlap"""
    if not context_messages:
        return 0.5
    
    # Combine recent context
    recent_context = " ".join(context_messages[-3:])
    
    # Calculate Jaccard similarity
    return _jaccard_similarity(candidate_text, recent_context)


def _novelty_lex(candidate_text: str, accepted_history: List[str], K: int = 5) -> float:
    """Calculate novelty using lexical similarity with recent history"""
    if not accepted_history:
        return 1.0  # First message is always novel
    
    # Compare with last K messages
    recent_history = accepted_history[-K:]
    similarities = []
    
    for msg in recent_history:
        sim = _jaccard_similarity(candidate_text, msg)
        similarities.append(sim)
    
    # Novelty = 1 - max_similarity (most similar message determines novelty)
    max_similarity = max(similarities) if similarities else 0.0
    return 1.0 - max_similarity


def _actionability(candidate_text: str) -> float:
    """
    Calculate actionability score based on concrete actions and structured content.
    Enhanced to prioritize artifact structure.
    """
    text_lower = candidate_text.lower()
    
    # Artifact structure indicators (high actionability)
    structure_indicators = [
        r'##\s+',  # Headers
        r'###\s+', # Subheaders
        r'^\s*[-*•]\s+', # Bullet points
        r'^\s*\d+\.\s+', # Numbered lists
        r'\*\*.*?\*\*', # Bold text
        r'```', # Code blocks
        r'\|.*\|', # Tables
    ]
    
    structure_score = 0
    for pattern in structure_indicators:
        matches = len(re.findall(pattern, candidate_text, re.MULTILINE))
        structure_score += min(matches, 3)  # Cap at 3 per type
    
    # Normalize structure score (0-1)
    structure_score = min(1.0, structure_score / 10)
    
    # Action words and phrases
    action_words = [
        "implement", "create", "develop", "design", "build", "establish",
        "define", "specify", "configure", "setup", "install", "deploy",
        "analyze", "evaluate", "assess", "review", "examine", "investigate",
        "plan", "schedule", "organize", "coordinate", "manage", "execute"
    ]
    
    action_score = 0
    for word in action_words:
        if word in text_lower:
            action_score += 1
    
    # Normalize action score (0-1)
    action_score = min(1.0, action_score / 5)
    
    # Constraint-bearing questions (medium actionability)
    constraint_patterns = [
        r'should we.*\b(first|next|before|after)\b',
        r'what.*\b(specific|exactly|precisely)\b',
        r'how.*\b(should|would|can)\b.*\b(we|I)\b',
        r'\b(which|what).*\b(option|approach|method)\b'
    ]
    
    constraint_score = 0
    for pattern in constraint_patterns:
        if re.search(pattern, text_lower):
            constraint_score += 0.3
    
    constraint_score = min(1.0, constraint_score)
    
    # Combine scores (prioritize structure)
    if structure_score > 0.3:  # Has significant structure
        return 1.0  # Maximum actionability for structured content
    else:
        return max(action_score, constraint_score)


def _information_gain(candidate_text: str) -> float:
    """Calculate information gain based on content richness and specificity"""
    if not candidate_text.strip():
        return 0.0
    
    # Word count factor
    word_count = len(candidate_text.split())
    word_factor = min(1.0, word_count / 50)  # Normalize to 50 words
    
    # Specificity indicators
    specific_patterns = [
        r'\d+',  # Numbers
        r'\b[A-Z][a-z]+\b',  # Proper nouns
        r'\$\d+',  # Money amounts
        r'\d+%',  # Percentages
        r'\b\d+\s*(days?|weeks?|months?|years?)\b',  # Time periods
    ]
    
    specificity_score = 0
    for pattern in specific_patterns:
        matches = len(re.findall(pattern, candidate_text))
        specificity_score += min(matches, 5)  # Cap at 5 per type
    
    specificity_factor = min(1.0, specificity_score / 10)
    
    # Technical depth indicators
    technical_words = [
        "algorithm", "framework", "methodology", "implementation", "architecture",
        "specification", "requirement", "protocol", "standard", "guideline",
        "analysis", "evaluation", "assessment", "optimization", "integration"
    ]
    
    technical_score = 0
    text_lower = candidate_text.lower()
    for word in technical_words:
        if word in text_lower:
            technical_score += 1
    
    technical_factor = min(1.0, technical_score / 5)
    
    # Combine factors
    return (word_factor + specificity_factor + technical_factor) / 3


# ----------------------------
# Tier-0 TIS Components
# ----------------------------

def _semantic_distance(candidate_text: str, accepted_history: List[str], base_url: str = None) -> float:
    """
    Calculate Semantic Distance (novelty) using embeddings.
    Higher values indicate more novel content.
    """
    if not accepted_history:
        return 1.0  # First message is maximally novel
    
    candidate_emb = _get_embedding(candidate_text, base_url)
    
    # Compare with recent history (last 5 messages)
    recent_history = accepted_history[-5:]
    similarities = []
    
    for msg in recent_history:
        msg_emb = _get_embedding(msg, base_url)
        sim = _cosine_similarity(candidate_emb, msg_emb)
        similarities.append(sim)
    
    # Semantic Distance = 1 - max_similarity
    max_similarity = max(similarities) if similarities else 0.0
    return 1.0 - max_similarity


def _reciprocal_coherence(candidate_text: str, context_messages: List[str], base_url: str = None) -> float:
    """
    Calculate Reciprocal Coherence (relevance) using embedding similarity.
    Higher values indicate better coherence with context.
    """
    return _relevance_embedding(candidate_text, context_messages, base_url)


def _interaction_smoothness(current_text: str, previous_text: Optional[str], base_url: str = None) -> float:
    """
    Calculate Interaction Smoothness using semantic similarity with previous message.
    Higher values indicate smoother conversation flow.
    """
    if not previous_text:
        return 1.0  # First message has perfect smoothness
    
    current_emb = _get_embedding(current_text, base_url)
    previous_emb = _get_embedding(previous_text, base_url)
    
    return _cosine_similarity(current_emb, previous_emb)


def _entropy_info_contribution(candidate_text: str, accepted_history_texts: List[str]) -> float:
    """
    Calculate Entropy-based Information Contribution.
    Measures how much new information the candidate adds.
    """
    if not candidate_text.strip():
        return 0.0
    
    # Tokenize candidate text
    candidate_words = re.findall(r'\b[a-z]+\b', candidate_text.lower())
    if not candidate_words:
        return 0.0
    
    # Build vocabulary from history
    history_text = " ".join(accepted_history_texts) if accepted_history_texts else ""
    history_words = re.findall(r'\b[a-z]+\b', history_text.lower())
    
    # Calculate word frequencies in history
    history_freq = Counter(history_words)
    total_history_words = len(history_words)
    
    if total_history_words == 0:
        return 1.0  # First message contributes maximum information
    
    # Calculate information contribution
    info_contribution = 0.0
    for word in set(candidate_words):  # Unique words only
        # Frequency in history (with smoothing)
        hist_freq = history_freq.get(word, 0)
        prob = (hist_freq + 1) / (total_history_words + len(history_freq))
        
        # Information content = -log(probability)
        info_content = -math.log2(prob)
        info_contribution += info_content
    
    # Normalize by candidate length
    unique_candidate_words = len(set(candidate_words))
    if unique_candidate_words == 0:
        return 0.0
    
    avg_info = info_contribution / unique_candidate_words
    
    # Scale to 0-1 range (empirically tuned)
    return min(1.0, avg_info / 10)


def _stability_score(tis_history: List[float], window: int = 5) -> float:
    """
    Calculate Stability Score based on TIS variance over recent window.
    Higher values indicate more stable conversation quality.
    """
    if len(tis_history) < 2:
        return 1.0  # Assume stable for short conversations
    
    # Use recent window
    recent_tis = tis_history[-window:] if len(tis_history) >= window else tis_history
    
    if len(recent_tis) < 2:
        return 1.0
    
    # Calculate coefficient of variation (lower is more stable)
    mean_tis = sum(recent_tis) / len(recent_tis)
    if mean_tis == 0:
        return 0.0
    
    variance = sum((x - mean_tis) ** 2 for x in recent_tis) / len(recent_tis)
    std_dev = math.sqrt(variance)
    cv = std_dev / mean_tis
    
    # Convert to stability score (lower CV = higher stability)
    stability = max(0.0, 1.0 - cv)
    return min(1.0, stability)


def _calculate_tis(sd: float, rc: float, is_val: float, eic: float, st: float, 
                   weights: TISWeights = tis_weights) -> float:
    """
    Calculate Tier-0 Interaction Score (TIS) from components.
    
    TIS = α·SD + β·RC + γ·IS + δ·EIC + ε·Sₜ
    """
    tis = (
        weights.alpha * sd +
        weights.beta * rc +
        weights.gamma * is_val +
        weights.delta * eic +
        weights.epsilon * st
    )
    
    return max(0.0, min(1.0, tis))  # Clamp to [0, 1]


# ----------------------------
# Main scoring function
# ----------------------------

def score_message(
    candidate_text: str,
    context_messages: List[str],
    accepted_history: List[str],
    previous_message: Optional[str] = None,
    tis_history: List[float] = None,
    latency_seconds: float = 0.0,
    model: str = "",
    base_url: str = None,  # NEW: Pass base_url for embeddings
    L_max: float = 8.0
) -> Dict[str, float]:
    """
    Comprehensive message scoring with Tier-0 TIS metrics.
    
    Returns dictionary with all metrics:
    - Legacy: R, N, A, IG, C, L, U
    - Tier-0: SD, RC, IS, EIC, St, TIS
    """
    if tis_history is None:
        tis_history = []
    
    # Legacy metrics
    R_emb = _relevance_embedding(candidate_text, context_messages, base_url)
    R_lex = _relevance_lexical(candidate_text, context_messages)
    R = max(R_emb, R_lex)  # Take maximum relevance
    
    N = _novelty_lex(candidate_text, accepted_history)
    A = _actionability(candidate_text)
    IG = _information_gain(candidate_text)
    
    # Cost and latency
    token_count = len(candidate_text.split())
    C = token_count / 1000.0  # Normalize to cost per 1000 tokens
    L = min(latency_seconds / L_max, 1.0)  # Normalize latency
    
    # Legacy utility (without latency to avoid indirect effects)
    U_raw = 0.50 * A + 0.45 * IG - 0.08 * C
    U = max(0.0, min(1.0, U_raw))
    
    # Tier-0 TIS components
    SD = _semantic_distance(candidate_text, accepted_history, base_url)
    RC = _reciprocal_coherence(candidate_text, context_messages, base_url)
    IS = _interaction_smoothness(candidate_text, previous_message, base_url)
    EIC = _entropy_info_contribution(candidate_text, accepted_history)
    St = _stability_score(tis_history)
    
    # Calculate TIS
    TIS = _calculate_tis(SD, RC, IS, EIC, St)
    
    return {
        # Legacy metrics
        "R": R,
        "N": N,
        "A": A,
        "IG": IG,
        "C": C,
        "L": L,
        "U": U,
        "tok": float(token_count),
        "L_raw": latency_seconds,
        
        # Tier-0 TIS components
        "SD": SD,
        "RC": RC,
        "IS": IS,
        "EIC": EIC,
        "St": St,
        "TIS": TIS
    }
