"""
AGNN Post-Run Synthesizer

After all phases complete, takes the top accepted messages grouped by phase
and produces a single clean, consolidated Markdown document.

Uses multi-pass synthesis: if the content is too long for one LLM call,
it synthesizes section-by-section and stitches the results together.
This prevents output truncation regardless of document length.
"""

from __future__ import annotations

import os
import re
import json
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from .llm_client import chat_completion


_SYNTHESIS_SYSTEM_PROMPT = """You are a professional technical editor and document architect.
You will receive structured outputs from a multi-agent AI system where each branch investigated
a specific angle of the task independently.

YOUR ONLY JOB IS TO EDIT AND ORGANIZE — NOT TO GENERATE NEW CONTENT.

Critical Rules:
- You are an EDITOR, not a researcher. Do NOT invent facts, opinions, or recommendations
  that are not explicitly present in the branch outputs below.
- Each branch's contribution is labeled [[Branch: Role Name]]. Preserve and credit insights
  from EACH branch — do not drop any branch's unique content.
- Cross-reference branches explicitly where they overlap or contradict each other.
  For example: "The Compliance Officer noted X, which the Bias Analyst's finding Y reinforces."
- Remove all boilerplate phrases like "Here's a revised version", "Let me know if you need more".
- Remove all references to "AgentA", "AgentB", "AgentC", "AgentD" — use role names instead.
- Remove duplicate ideas across branches; keep the most detailed/specific version of each point.
- Structure with clear Markdown headings (##, ###).
- When branches give CONFLICTING values, note both explicitly: "Branch A states X; Branch B states Y."
- Write ALL sections completely. Do NOT stop early or truncate.
- Do NOT end mid-sentence or mid-section under any circumstances.
"""

_SYNTHESIS_USER_TEMPLATE = """Task: {user_prompt}

The following are the outputs from parallel agent branches, each investigating a different
aspect of the task. Your job is to synthesize these into ONE coherent final document.
Do NOT add information that is not present in the branches below.

{phase_content}

---
Synthesize into ONE clean, complete Markdown document.
Reference contributions from each branch by role name where relevant.
Cross-reference branches where they overlap or contradict.
Cover ALL sections of the original task. Do not stop until every section is complete."""

_SECTION_SYSTEM_PROMPT = """You are a technical editor.
Synthesize the provided multi-agent research into the ONE specific section requested.
You are an EDITOR — do NOT invent facts not present in the source material.
Be specific, use numbers and examples directly from the source. Write in clean Markdown.
Reference which branch provided which insight where relevant.
Remove any boilerplate, agent codenames (AgentA/B/C/D), or meta-commentary.
Do not write other sections — only the one requested."""

_SECTION_USER_TEMPLATE = """Original task: {user_prompt}

Section to write: **{section_name}**

Source material from parallel branches:
{content}

Write only the "{section_name}" section now, in clean Markdown with ### sub-headings as needed.
Only use information present in the source material above."""


def _chat_with_backoff(*, system_prompt: str, user_prompt: str, model: str, base_url: str,
                       timeout: float, max_tokens: int, temperature: float,
                       retries: int = 3, base_sleep: float = 1.0):
    """
    Wrapper around chat_completion with explicit retry/backoff for transient failures
    (especially HTTP 429 from cloud endpoints).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                base_url=base_url,
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            last_exc = exc
            err = str(exc)
            transient = any(x in err for x in ("429", "Too Many Requests", "timed out", "Request failed"))
            if attempt < retries - 1 and transient:
                time.sleep(base_sleep * (2 ** attempt))
                continue
            raise
    raise RuntimeError(f"Synthesis call failed: {last_exc}")


def _pick_best_messages_per_phase(
    accepted: List[Dict[str, Any]],
    top_n_per_phase: int = 6,
    min_tis: float = 0.68,
) -> Dict[str, List[str]]:
    """
    Group accepted messages by phase_id (DAG branch) and pick the top N by TIS.
    Returns {phase_type: [content1, content2, ...]}
    Each content block is prefixed with [[Branch: RoleName]] so the Synthesizer
    can cross-reference agent contributions explicitly.
    """
    phase_buckets: Dict[int, List[Dict]] = {}
    phase_type_map: Dict[int, str] = {}

    for msg in accepted:
        pid = msg.get("phase_id", 0)
        phase_type = msg.get("phase") or "general"
        if phase_type == "unknown":
            continue
        phase_buckets.setdefault(pid, []).append(msg)
        phase_type_map[pid] = phase_type

    result: Dict[str, List[str]] = {}
    for pid, msgs in phase_buckets.items():
        phase_type = phase_type_map.get(pid, "general")
        msgs.sort(key=lambda m: m.get("metrics", {}).get("TIS", 0.0), reverse=True)
        top = [m for m in msgs if m.get("metrics", {}).get("TIS", 0.0) >= min_tis][:top_n_per_phase]
        if top:
            tagged = []
            for m in top:
                role_name = m.get("role", m.get("agent_id", f"Branch-{pid}"))
                tag = f"[[Branch: {role_name}]]\n{m['content']}"
                tagged.append(tag)
            result.setdefault(phase_type, []).extend(tagged)

    return result


def _strip_meta_commentary(text: str) -> str:
    """Remove common LLM meta-phrases before passing to synthesizer."""
    patterns = [
        r"(?i)^(okay|alright|sure|certainly|absolutely)[,!.]?\s*",
        r"(?i)here['']?s\s+(a\s+)?revised\s+version\s+(of\s+the\s+message\s+)?addressing\s+the\s+issues[^:]*:?\s*",
        r"(?i)^let me know if you (have any further questions|need more details)[^\.]*\.\s*",
        r"(?i)\bPlease let me know\b.*?\.\s*",
        r"(?i)^to further refine the onboarding plan[,.]?\s*",
        r"---\s*$",
        # Remove "Changes Made:" meta-sections
        r"(?i)\n\*\*Changes Made:\*\*.*$",
        r"(?i)\nChanges Made:.*$",
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.MULTILINE | re.DOTALL).strip()
    return text


def _scrub_agent_names(text: str) -> str:
    """Replace internal agent codenames with neutral phrasing."""
    text = re.sub(r"\b(Agent[ABCDE])\b", "the team", text)
    text = re.sub(
        r"(?i)\n*(please let me know.*|let me know if.*|feel free to.*|"
        r"i hope this helps.*|this (plan|document) (is|provides|aims).*)\s*$",
        "",
        text,
        flags=re.DOTALL,
    ).strip()
    return text


def _infer_sections_from_prompt(user_prompt: str) -> List[str]:
    """
    Heuristically extract expected document sections from the user's task prompt.
    Used for multi-pass synthesis when the full content is too long.
    """
    # Look for comma/semicolon-separated lists after "Include:" or "Cover:"
    match = re.search(
        r"(?i)(?:include|cover|address|provide)[:\s]+(.+?)(?:\.|$)", user_prompt
    )
    if match:
        raw = match.group(1)
        items = re.split(r"[,;]", raw)
        sections = []
        for item in items:
            s = item.strip().strip("-").strip()
            if len(s) > 5:
                # Capitalise first word
                sections.append(s[0].upper() + s[1:])
        if len(sections) >= 3:
            return sections

    # Fallback: generic document sections
    return [
        "Executive Summary",
        "Market Analysis",
        "Strategic Recommendations",
        "Implementation Plan",
        "Risks and Mitigations",
        "Conclusion",
    ]


def _single_pass_synthesis(
    user_prompt: str,
    phase_content: str,
    base_url: str,
    model: str,
    max_tokens: int = 3000,
) -> Optional[str]:
    """Single LLM call synthesis. Used when content fits within context limits."""
    user_msg = _SYNTHESIS_USER_TEMPLATE.format(
        user_prompt=user_prompt,
        phase_content=phase_content,
    )
    try:
        response = _chat_with_backoff(
            system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=user_msg,
            model=model,
            base_url=base_url,
            timeout=180.0,
            max_tokens=max_tokens,
            temperature=0.25,
        )
        return response.text.strip()
    except Exception as e:
        print(f"[Synthesizer] Single-pass call failed: {e}")
        return None


def _multi_pass_synthesis(
    user_prompt: str,
    phase_content: str,
    base_url: str,
    model: str,
    max_tokens_per_section: int = 800,
) -> Optional[str]:
    """
    Section-by-section synthesis: each section gets its own LLM call.
    Guarantees complete output regardless of total document length.
    """
    sections = _infer_sections_from_prompt(user_prompt)
    print(f"[Synthesizer] Multi-pass: synthesising {len(sections)} sections individually")

    # Trim content per section call to avoid context overflow
    content_per_section = phase_content[:6000]

    section_outputs: List[str] = []
    for section in sections:
        user_msg = _SECTION_USER_TEMPLATE.format(
            user_prompt=user_prompt,
            section_name=section,
            content=content_per_section,
        )
        try:
            response = _chat_with_backoff(
                system_prompt=_SECTION_SYSTEM_PROMPT,
                user_prompt=user_msg,
                model=model,
                base_url=base_url,
                timeout=120.0,
                max_tokens=max_tokens_per_section,
                temperature=0.25,
            )
            text = response.text.strip()
            if text:
                # Ensure the section has a heading
                if not text.startswith("#"):
                    text = f"## {section}\n\n{text}"
                section_outputs.append(_scrub_agent_names(text))
                print(f"[Synthesizer]   ✓ {section} ({len(text.split())} words)")
        except Exception as e:
            print(f"[Synthesizer]   ✗ {section}: {e}")
            continue

    if not section_outputs:
        return None
    return "\n\n---\n\n".join(section_outputs)


def synthesize(
    user_prompt: str,
    accepted_messages: List[Dict[str, Any]],
    base_url: str,
    model: Optional[str] = None,
    output_dir: str = "agnn/outputs",
    slug: Optional[str] = None,
    handoff_packages: Optional[Dict] = None,
) -> Optional[str]:
    """
    Run the post-run synthesis pass.

    Strategy:
    - If total phase content ≤ 8000 chars → single LLM call (fast path)
    - If total phase content > 8000 chars → multi-pass section-by-section (safe path)

    Both paths are guarded against truncation.

    Args:
        handoff_packages: Optional dict {subgoal_id: HandoffPackage} with structured
                         workspace summaries. When provided, a structured overview
                         is prepended to help the synthesizer organize better.
    """
    if not accepted_messages:
        print("[Synthesizer] No accepted messages to synthesise.")
        return None

    # --- 1. Pick best messages per phase ---
    phase_map = _pick_best_messages_per_phase(accepted_messages, top_n_per_phase=6)
    if not phase_map:
        print("[Synthesizer] No qualifying messages (TIS >= 0.68) to synthesise.")
        return None

    # --- 2. Build phase content block ---
    phase_order = ["research", "analysis", "draft", "review", "general"]
    ordered_phases = sorted(
        phase_map.keys(),
        key=lambda p: phase_order.index(p) if p in phase_order else 99,
    )

    phase_blocks = []

    # Inject structured handoff summaries if available
    if handoff_packages:
        handoff_overview = ["### Structured Workspace Summaries\n"]
        for sid, pkg in handoff_packages.items():
            handoff_overview.append(f"**{pkg.subgoal_name}** ({pkg.phase_type}):")
            handoff_overview.append(f"  Summary: {pkg.what_was_done}")
            if pkg.what_matters_downstream:
                for point in pkg.what_matters_downstream[:3]:
                    handoff_overview.append(f"  • {point}")
            if pkg.uncertainties:
                for u in pkg.uncertainties[:2]:
                    handoff_overview.append(f"  ⚠ {u}")
            handoff_overview.append("")
        phase_blocks.append("\n".join(handoff_overview))

    for phase in ordered_phases:
        contents = phase_map[phase]
        cleaned = [_strip_meta_commentary(c) for c in contents]
        phase_blocks.append(
            f"### Phase: {phase.upper()}\n\n" + "\n\n---\n\n".join(cleaned)
        )

    phase_content = "\n\n".join(phase_blocks)
    used_model = model or ""

    # --- 3. Choose synthesis strategy ---
    print("[Synthesizer] Running final synthesis pass...")
    print(f"[Synthesizer] Content size: {len(phase_content):,} chars | "
          f"phases: {list(ordered_phases)}")

    if len(phase_content) <= 8000:
        print("[Synthesizer] Strategy: single-pass (content fits in one call)")
        synthesized_text = _single_pass_synthesis(
            user_prompt, phase_content, base_url, used_model, max_tokens=3000
        )
    else:
        print("[Synthesizer] Strategy: multi-pass (content too long for single call)")
        # Try single pass on a truncated version first (faster)
        truncated = phase_content[:8000] + "\n\n[...additional content available...]"
        synthesized_text = _single_pass_synthesis(
            user_prompt, truncated, base_url, used_model, max_tokens=3000
        )
        # Check if output looks truncated (ends mid-sentence or no final heading)
        if synthesized_text and (
            not synthesized_text.rstrip().endswith((".", ")", "-", "*"))
            or len(synthesized_text.split()) < 400
        ):
            print("[Synthesizer] Single-pass output appears truncated — switching to multi-pass")
            synthesized_text = _multi_pass_synthesis(
                user_prompt, phase_content, base_url, used_model
            )

    degraded = False
    if not synthesized_text:
        degraded = True
        print("[Synthesizer] Empty response — returning degraded synthesis notice.")
        synthesized_text = (
            "## Synthesis Degraded\n\n"
            "- Final synthesis could not be completed due to repeated model/API failures.\n"
            "- This output is intentionally marked degraded to avoid presenting unreliable merged content.\n"
            "- Please retry with a healthy model or lower request rate.\n"
        )

    # --- 3b. Post-processing scrub ---
    synthesized_text = _scrub_agent_names(synthesized_text)

    # --- 4. Task-Aware Output Routing ---
    # Detect what KIND of deliverable this task should produce
    # and save the correct file type instead of always defaulting to .md
    os.makedirs(output_dir, exist_ok=True)

    if not slug:
        prompt_clean = re.sub(r"[^a-z0-9\s]", "", user_prompt.lower())
        words = prompt_clean.split()[:8]
        slug = "-".join(words) or "output"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # --- Detect output type from prompt keywords and tool call content ---
    tool_calls_text = " ".join(
        m.get("content", "") for m in accepted_messages if "[TOOL]" in m.get("content", "")
    ).lower()
    prompt_lower = user_prompt.lower()

    # Extract code blocks from synthesized text
    code_blocks = re.findall(r"```(?:python|py)\n(.*?)```", synthesized_text, re.DOTALL)
    bat_blocks   = re.findall(r"```(?:batch|bat|cmd|powershell|ps1)\n(.*?)```", synthesized_text, re.DOTALL)

    def _is_task(keywords):
        return any(k in prompt_lower for k in keywords)

    output_type = "md"  # default

    if code_blocks and _is_task(["python script", "write a script", "create a script", ".py", "python file"]):
        output_type = "py"
    elif bat_blocks and _is_task(["batch", "bat file", "powershell", ".bat", ".ps1", "automation script"]):
        output_type = "bat"
    elif _is_task(["convert", "pdf", "create a pdf", "make a pdf", "save as pdf", "export pdf"]):
        output_type = "pdf_from_md"
    elif _is_task(["email", "letter", "memo", "plain text", "save as txt", ".txt"]):
        output_type = "txt"
    elif _is_task(["folder", "create folder", "make folder", "delete", "move files",
                   "copy files", "rename", "organize", "list files", "list folder",
                   "check disk", "system info", "audit", "analyse", "analyze"]) \
            and "write" not in prompt_lower and "script" not in prompt_lower:
        # Pure file-system / analysis tasks with no document output
        # Save a lean action log rather than a full report
        output_type = "log"

    # --- Save by detected type ---
    if output_type == "py":
        filename = f"{output_dir}/{slug}-{timestamp}.py"
        raw_code = code_blocks[0].strip()
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"# Generated by AGNN — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"# Task: {user_prompt}\n\n")
                f.write(raw_code)
            print(f"[Synthesizer] Python script saved to: {filename}")
        except Exception as e:
            print(f"[Synthesizer] Failed to save .py: {e}")
        # Also save full synthesis as companion .md
        md_companion = filename.replace(".py", "_report.md")
        with open(md_companion, "w", encoding="utf-8") as f:
            f.write(f"# {user_prompt}\n\n*Generated by AGNN — {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n---\n\n")
            f.write(synthesized_text)

    elif output_type == "bat":
        filename = f"{output_dir}/{slug}-{timestamp}.bat"
        raw_code = bat_blocks[0].strip()
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f":: Generated by AGNN — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f":: Task: {user_prompt}\n\n")
                f.write(raw_code)
            print(f"[Synthesizer] Batch script saved to: {filename}")
        except Exception as e:
            print(f"[Synthesizer] Failed to save .bat: {e}")

    elif output_type == "txt":
        filename = f"{output_dir}/{slug}-{timestamp}.txt"
        try:
            plain = re.sub(r"#{1,6}\s+", "", synthesized_text)
            plain = re.sub(r"\*\*(.+?)\*\*", r"\1", plain)
            plain = re.sub(r"\*(.+?)\*", r"\1", plain)
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"Generated by AGNN — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"Task: {user_prompt}\n{'─'*60}\n\n")
                f.write(plain)
            print(f"[Synthesizer] Text file saved to: {filename}")
        except Exception as e:
            print(f"[Synthesizer] Failed to save .txt: {e}")

    elif output_type == "log":
        filename = f"{output_dir}/{slug}-{timestamp}.log"
        try:
            tool_lines = [
                m.get("content", "") for m in accepted_messages
                if "[TOOL]" in m.get("content", "")
            ]
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"AGNN Action Log — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                f.write(f"Task: {user_prompt}\n{'─'*60}\n\n")
                for line in tool_lines:
                    f.write(line + "\n")
                f.write(f"\n{'─'*60}\nSummary:\n{synthesized_text[:800]}\n")
            print(f"[Synthesizer] Action log saved to: {filename}")
        except Exception as e:
            print(f"[Synthesizer] Failed to save .log: {e}")

    else:
        # Default: full markdown report
        filename = f"{output_dir}/{slug}-{timestamp}.md"
        header = (
            f"# {user_prompt}\n\n"
            f"*Generated by AGNN — {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n---\n\n"
        )
        if degraded:
            header += "> STATUS: COMPLETE_WITH_WARNINGS (degraded_synthesis)\n\n"
        full_output = header + synthesized_text
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(full_output)
            print(f"[Synthesizer] Final document saved to: {filename}")
        except Exception as e:
            print(f"[Synthesizer] Failed to save output: {e}")

    return synthesized_text
