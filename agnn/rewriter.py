"""
AGNN Message Rewriter
Handles rewriting of rejected messages based on specific rejection reasons.
Enhanced with better protocol violation handling.
"""

from typing import List, Dict, Any
from .llm_client import chat_completion, LLMResponse

# System prompt for the rewriter (SIMPLIFIED FOR LOCAL MODELS)
REWRITER_SYSTEM_PROMPT = """Fix the rejected message by addressing the specific issues mentioned.

RULES:
- NO speaker labels or agent names
- NO task assignments to others
- NO meta-commentary about being an AI
- Keep the original meaning but fix the problems
- Be direct and helpful

Just rewrite the message to fix the issues."""

def _reasons_to_instructions(reasons: List[str]) -> str:
    """Convert rejection reasons to specific rewriting instructions"""
    instructions = []
    
    for reason in reasons:
        reason_lower = reason.lower()
        
        if "repetitive content" in reason_lower or "low novelty" in reason_lower:
            instructions.append("Add new information or a different perspective. Avoid repeating previous points.")
        
        elif "low relevance" in reason_lower or "coherence" in reason_lower:
            instructions.append("Make your response more directly relevant to the current discussion topic.")
        
        elif "low information" in reason_lower or "contribution" in reason_lower:
            instructions.append("Provide more specific, concrete information or actionable insights.")
        
        elif "poor conversation flow" in reason_lower or "smoothness" in reason_lower:
            instructions.append("Build more naturally on the previous message and maintain conversation flow.")
        
        elif "protocol violation" in reason_lower or "speaker label" in reason_lower:
            instructions.append("Remove any speaker labels, agent names, or task assignments. Focus on content only.")
        
        elif "too long" in reason_lower:
            instructions.append("Significantly shorten your response while keeping the key points.")
        
        elif "draft phase violation" in reason_lower:
            instructions.append("STOP researching/planning. WRITE the actual document NOW with concrete sections, headings, and detailed content.")
        
        elif "unstable conversation" in reason_lower:
            instructions.append("Provide a more measured, consistent response that maintains conversation quality.")
        
        else:
            instructions.append(f"Address this issue: {reason}")
    
    return " ".join(instructions)

def rewrite_message(
    original_message: str,
    rejection_reasons: List[str],
    context_messages: List[str],
    model: str,
    base_url: str,
    max_attempts: int = 2
) -> str:
    """
    Rewrite a rejected message to address the rejection reasons.
    
    Args:
        original_message: The original rejected message
        rejection_reasons: List of reasons why it was rejected
        context_messages: Recent conversation context
        model: Model to use for rewriting
        base_url: LM Studio base URL
        max_attempts: Maximum rewrite attempts
        
    Returns:
        Rewritten message text
    """
    if not rejection_reasons:
        return original_message
    
    # Create rewriting instructions
    instructions = _reasons_to_instructions(rejection_reasons)
    
    # Check if this is a draft phase rejection to increase token limit
    is_draft_fix = any("draft" in r.lower() or "artifact" in r.lower() for r in rejection_reasons)
    current_max_tokens = 1500 if is_draft_fix else 300
    
    # Build context
    context = ""
    if context_messages:
        recent_context = context_messages[-3:]  # Last 3 messages
        context = "\n\nRecent conversation:\n" + "\n".join(f"- {msg}" for msg in recent_context)
    
    # Create SIMPLE rewriting prompt for local models
    user_prompt = f"""Fix this message:
"{original_message}"

Problems: {', '.join(rejection_reasons)}

Fix: {instructions}

Rewrite it properly:"""

    try:
        response = chat_completion(
            system_prompt=REWRITER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            model=model,
            base_url=base_url,
            max_tokens=current_max_tokens,  # Increased for draft artifacts
            temperature=0.5,  # Lower temperature for more focused rewrites
            timeout=60.0 if is_draft_fix else 30.0     # Longer timeout for larger generations
        )
        
        rewritten = response.text.strip()
        
        # Basic validation
        if len(rewritten) < 10:
            return original_message  # Fallback if rewrite is too short
        
        return rewritten
        
    except Exception as e:
        print(f"Rewrite failed: {e}")
        return original_message  # Fallback to original on error
