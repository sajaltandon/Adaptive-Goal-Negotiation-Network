"""
AGNN CLI entry point.
Terminal-first autonomous flow:
1) user provides LM Studio URL + task
2) AGNN analyzes task
3) AGNN selects best model team
4) AGNN runs orchestration
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

from .live_display import LiveDisplay
from .orchestrator import Orchestrator
from .llm_client import list_models, chat_completion
from .task_analyzer import analyze_task
from .model_selector import select_models
from .tools import ToolRegistry


BANNER = r"""
============================================================
 AGNN - Adaptive Goal Negotiation Network (Autonomous CLI)
============================================================
"""

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "data", "run_profiles.json")


def _load_env_file() -> None:
    """Load simple KEY=VALUE pairs from a root .env file."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    env_path = os.path.join(repo_root, ".env")
    if not os.path.exists(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception as exc:
        print(f"Warning: Could not load .env file: {exc}")


def _print_task_analysis(analysis) -> None:
    data = analysis.to_dict()
    print("\n[1/4] Task Analysis")
    print(f"  Type        : {data['task_type']}")
    print(f"  Complexity  : {data['complexity']} ({data['complexity_score']})")
    print(f"  Team size   : {data['team_size']}")
    print(f"  Budget mode : {data['budget_profile']}")
    print(f"  Source      : {data.get('source', 'unknown')} | confidence={data.get('confidence', 0.0)}")
    print(f"  Phases      : {', '.join(data['phase_plan'])}")


def _print_selection(selection: dict) -> None:
    selected = selection.get('selected_models', [])
    rationale = selection.get('rationale', [])

    print("\n[2/4] Intelligent Model Selection")
    if not selected:
        print("  No models selected.")
        return

    print("  Selected models:")
    for i, model in enumerate(selected, 1):
        print(f"   {i}. {model}")

    if rationale:
        print("\n  Selection rationale:")
        for item in rationale:
            strengths = ', '.join(item.get('strengths', []))
            print(
                f"   - {item.get('model')}: score={item.get('score')} "
                f"| strengths=[{strengths}] "
                f"| rel={item.get('reliability')} | probe={item.get('probe_quality')}"
            )



def _choose_model_mode() -> str:
    while True:
        choice = input('\n[Mode] Model selection mode: Auto or Manual? [A/m]: ').strip().lower()
        if choice in ('', 'a', 'auto'):
            return 'auto'
        if choice in ('m', 'manual'):
            return 'manual'
        print('  Invalid choice. Enter A for Auto or M for Manual.')


def _manual_select_models(discovered: List[str], default_count: int) -> List[str]:
    print('\n[2/4] Manual Model Selection')
    print('  Available models:')
    for i, model in enumerate(discovered, 1):
        print(f'   {i}. {model}')

    while True:
        raw = input(
            f'  Enter model numbers (comma-separated) or press Enter for first {default_count}: '
        ).strip()
        if not raw:
            return discovered[:default_count]

        try:
            idxs = [int(x.strip()) for x in raw.split(',') if x.strip()]
        except ValueError:
            print('  Invalid input. Use numbers like: 1,2,4')
            continue

        unique = []
        for idx in idxs:
            if 1 <= idx <= len(discovered) and idx not in unique:
                unique.append(idx)

        if not unique:
            print('  No valid model numbers selected.')
            continue

        return [discovered[i - 1] for i in unique]


def _resolve_base_url(raw_url: str) -> str:
    # Keep host-level URL by default; endpoint family is auto-selected in llm_client.
    base = (raw_url or '').strip() or 'http://192.168.1.6:1234'
    return base.rstrip('/')


def _root_base_url(base_url: str) -> str:
    clean = (base_url or '').strip().rstrip('/')
    for suffix in ('/api/v1', '/v1'):
        if clean.endswith(suffix):
            return clean[: -len(suffix)].rstrip('/')
    return clean


def _resolve_embedding_url(base_url: str) -> str:
    # Embeddings remain OpenAI-compatible in current LM Studio builds.
    root = _root_base_url(base_url)
    return f"{root}/v1"


def _load_profiles() -> dict:
    if not os.path.exists(PROFILE_PATH):
        return {}
    try:
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_profiles(data: dict) -> None:
    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _preflight_models(base_url: str, models: List[str]) -> tuple[List[str], List[str]]:
    """Probe selected models and return (healthy, unhealthy)."""
    import os as _os
    _os.environ["_AGNN_PROBE_MODE"] = "1"  # silence error catalog boxes during probe
    print("\n[Preflight] Testing selected model endpoints...")
    healthy: List[str] = []
    unhealthy: List[str] = []
    registry = ToolRegistry()
    probe_prompt = "Call the list_dirs tool with path='.' and do not answer with prose."
    probe_tools = [
        registry.tools["read_file"].get_schema(),
        registry.tools["list_dirs"].get_schema(),
    ]
    for model in models:
        try:
            resp = chat_completion(
                system_prompt="You must use tools when the user explicitly requests one.",
                user_prompt=probe_prompt,
                model=model,
                base_url=base_url,
                timeout=90.0,
                max_tokens=80,
                temperature=0.0,
                tools=probe_tools,
                tool_choice="auto",
                retry_attempts=1,
            )
            tool_calls = resp.tool_calls or []
            used_list_dirs = any(
                isinstance(tc, dict) and tc.get("function", {}).get("name") == "list_dirs"
                for tc in tool_calls
            )
            if used_list_dirs or (resp.text or "").strip():
                healthy.append(model)
                print(f"  ✓ {model}")
            else:
                unhealthy.append(model)
                print(f"  ✗ {model} (unexpected response)")
        except Exception as exc:
            unhealthy.append(model)
            print(f"  ✗ {model} ({exc})")
    _os.environ.pop("_AGNN_PROBE_MODE", None)  # re-enable catalog
    print(f"[Preflight] Healthy: {len(healthy)}/{len(models)}")
    return healthy, unhealthy


def _manual_select_models_fast(discovered: List[str], suggested: List[str], default_count: int) -> List[str]:
    """
    Fast manual selector: one prompt, with profile-based suggestions.
    """
    print('\n[2/4] Manual Model Selection (Fast)')
    print('  Available models:')
    for i, model in enumerate(discovered, 1):
        marker = " (saved)" if model in suggested else ""
        print(f'   {i}. {model}{marker}')

    suggested_idxs = [str(i + 1) for i, m in enumerate(discovered) if m in suggested]
    fallback_idxs = [str(i) for i in range(1, min(len(discovered), default_count) + 1)]
    default_hint = ",".join(suggested_idxs) if suggested_idxs else ",".join(fallback_idxs)

    raw = input(f'  Enter model numbers (comma-separated) [{default_hint}]: ').strip()
    if not raw:
        raw = default_hint

    try:
        idxs = [int(x.strip()) for x in raw.split(',') if x.strip()]
    except ValueError:
        print('  Invalid input. Falling back to defaults.')
        idxs = [int(x) for x in (suggested_idxs or fallback_idxs)]

    unique = []
    for idx in idxs:
        if 1 <= idx <= len(discovered) and idx not in unique:
            unique.append(idx)
    if not unique:
        unique = [int(x) for x in fallback_idxs]
    return [discovered[i - 1] for i in unique]


def main() -> None:
    _load_env_file()
    parser = argparse.ArgumentParser(
        prog="python -m agnn",
        description="AGNN — Adaptive Goal Negotiation Network",
    )
    parser.add_argument(
        "--dashboard", "-d",
        action="store_true",
        default=False,
        help="Launch the rich live dashboard instead of the plain terminal view.",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["strict", "balanced", "fast"],
        default="balanced",
        help="Execution mode preset for thresholds and settings.",
    )
    parser.add_argument(
        "--quick-manual",
        action="store_true",
        default=False,
        help="Fast manual flow: fewer prompts, profile-backed model selection.",
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        default=False,
        help="Skip Gemini models this run (useful when Gemini API is slow/unavailable).",
    )
    parser.add_argument(
        "--no-groq",
        action="store_true",
        default=False,
        help="Skip Groq models this run.",
    )
    args, _ = parser.parse_known_args()

    os.system('cls' if os.name == 'nt' else 'clear')

    # ── Provider Status Banner ────────────────────────────────────────────────
    has_groq   = bool(os.environ.get("GROQ_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    provider_line = []
    if has_groq:   provider_line.append("Groq ✓")
    if has_gemini: provider_line.append("Gemini ✓")
    provider_line.append("LM Studio (auto)")

    # ── Display selection ─────────────────────────────────────────────────────
    # Dashboard temporarily disabled — using plain terminal for stability.
    display = LiveDisplay()

    print(BANNER)
    print(f"  Providers : {' | '.join(provider_line)}")
    print(f"  Command   : python -m agnn")
    print()

    try:
        profiles = _load_profiles()
        last = profiles.get("last_manual", {}) if isinstance(profiles.get("last_manual", {}), dict) else {}
        default_url = last.get("base_url", "http://192.168.1.6:1234")

        base_input = input(f'Enter LM Studio URL [{default_url}]: ').strip()
        base_url = _resolve_base_url(base_input)

        # ── Outer task loop ───────────────────────────────────────────────────
        # After each task completes, ask for the next. Ctrl+C or 'exit' to quit.
        while True:
            prompt = input('\nEnter task prompt (or "exit" to quit): ').strip()
            if not prompt or prompt.lower() in ('exit', 'quit', 'q'):
                print('Goodbye!')
                break

            print('\n[0/4] Discovering models...')
            import os as _os
            _os.environ["_AGNN_PROBE_MODE"] = "1"  # silence catalog during discovery
            discovered = list_models(base_url)
            _os.environ.pop("_AGNN_PROBE_MODE", None)

            # Filter out providers the user disabled for this run
            if args.no_gemini:
                discovered = [m for m in discovered if not m.startswith("models/gemini")]
                print("  [--no-gemini] Gemini models excluded.")
            if args.no_groq:
                discovered = [m for m in discovered if not m.startswith("groq/")]
                print("  [--no-groq] Groq models excluded.")

            if not discovered:
                print('No usable models found. Ensure LM Studio server is running and models are loaded.')
                continue  # ask for next task

            print(f"  Discovered {len(discovered)} models.")

            profiler_model = discovered[0] if discovered else None
            analysis = analyze_task(
                prompt,
                available_model_count=len(discovered),
                base_url=base_url,
                profiler_model=profiler_model,
                llm_passes=1,
            )
            _print_task_analysis(analysis)

            # ── Model selection + preflight loop ─────────────────────────────────
            selected_models: List[str] = []
            while not selected_models:
                default_n = max(2, min(analysis.team_size, len(discovered), 4))
                mode = "manual" if args.quick_manual else _choose_model_mode()
                selection = {'selected_models': [], 'rationale': []}

                if mode == 'manual':
                    if args.quick_manual:
                        suggested_models = [m for m in last.get("selected_models", []) if m in discovered]
                        selected_models = _manual_select_models_fast(discovered, suggested_models, default_n)
                    else:
                        selected_models = _manual_select_models(discovered, default_n)
                    selection = {'selected_models': selected_models, 'rationale': []}
                    print('\n  Manual selection complete.')
                    for i, model in enumerate(selected_models, 1):
                        print(f'   {i}. {model}')
                else:
                    selection = select_models(
                        base_url=base_url,
                        models=discovered,
                        analysis=analysis,
                        max_agents=max(2, min(analysis.team_size, 5)),
                        enable_probe=True,
                    )
                    selected_models = selection.get('selected_models', [])
                    if not selected_models:
                        selected_models = discovered[:default_n]
                        selection = {'selected_models': selected_models, 'rationale': []}
                    _print_selection(selection)

                if not args.quick_manual:
                    proceed = input('\nProceed with these models? [Y/n]: ').strip().lower()
                    if proceed == 'n':
                        print('Run cancelled by user.')
                        selected_models = []  # force re-select
                        break

                if selected_models:
                    healthy_models, unhealthy_models = _preflight_models(base_url, selected_models)
                    if unhealthy_models:
                        print(f"[Preflight] Unhealthy: {', '.join(unhealthy_models)} — dropping.")
                    selected_models = healthy_models

                if not selected_models:
                    print("\n[Preflight] No healthy models remain.")
                    retry = input("  Re-select models? [Y/n]: ").strip().lower()
                    if retry == 'n':
                        break  # back to task prompt
                    print("\n  Returning to model selection...\n")

            if not selected_models:
                print("[Skipping task — no healthy models selected.]")
                continue  # back to task prompt

            # Persist last successful manual preferences for quick runs.
            profiles["last_manual"] = {
                "base_url": base_url,
                "selected_models": selected_models,
                "updated_at": __import__("datetime").datetime.now().isoformat(),
            }
            _save_profiles(profiles)

            # ── Auto execution mode selection ────────────────────────────────────────
            # Rule: 2+ healthy models → always DAG (Tier-2). 1 model → Solo.
            # Complexity heuristics decide Hybrid vs pure DAG, but never force Solo
            # when the user explicitly selected 2+ models.
            if len(selected_models) >= 2:
                has_cloud_model = any("groq" in m or "gemini" in m for m in selected_models)
                has_local_model = any("groq" not in m and "gemini" not in m for m in selected_models)
                if has_cloud_model and has_local_model:
                    execution_mode = "Hybrid (Cloud Planner + Local Execution)"
                else:
                    execution_mode = "DAG (Multi-Agent Team)"
                enable_tier2 = True
            else:
                execution_mode = "Solo (Minimal Fast Execution)"
                enable_tier2 = False

            print(f"\n[Auto-Mode] Selected Execution Mode: {execution_mode}")
            if hasattr(display, 'set_execution_mode'):
                display.set_execution_mode(execution_mode)

            print('\n[3/4] Starting AGNN run...\n')
            orch = Orchestrator(
                base_url=base_url,
                models=selected_models,
                force_agent_mode=False,
                max_turns=50,
                debug=False,
                enable_tier2=enable_tier2,
                embedding_base_url=_resolve_embedding_url(base_url),
                task_analysis=analysis.__dict__,
                display=display,
                execution_mode=args.mode,
            )

            accepted = orch.run(prompt)

            print('\n[4/4] Run Summary')
            print(f"  Termination : {getattr(orch, 'termination_reason', 'unknown')}")
            print(f"  Accepted    : {len(accepted)}")
            print(f"  Rewrites    : {getattr(orch, 'rewrite_count', 0)}")
            print(f"  Rejected    : {getattr(orch, 'rejected_count', 0)}")

            if enable_tier2 and getattr(orch, 'phase_controller', None):
                s = orch.phase_controller.get_summary()
                print(f"  Phases      : {s['completed_phases']}/{s['total_phases']} completed")

            print("\n" + "─"*62)
            print("  ✓ Task complete. Paste next task or type 'exit'.")
            print("─"*62)
            # while True → next task prompt

    except KeyboardInterrupt:
        print('\nInterrupted by user.')
    except Exception as exc:
        print(f'\nError: {exc}')
        raise


if __name__ == '__main__':
    main()
