"""
LiveDisplay — thread-safe real-time terminal output for parallel workspaces.

Design:
  - For PARALLEL runs (multiple workspaces active): print turn-level summaries
    immediately when each turn completes, with a coloured workspace label.
  - For SOLO runs (one workspace): stream tokens as they arrive.
  - Never garbles output: a single threading.Lock() serialises all writes.

Usage:
    display = LiveDisplay()

    # Token streaming callback (solo mode):
    resp = stream_chat_completion(..., on_token=display.make_token_cb("Market Research"))

    # Turn-level broadcast (parallel mode):
    display.turn_accepted("Market Research", "research", turn=2,
                          snippet="TAM is estimated at $180B...", tis=0.74)
    display.workspace_done("Market Research", turns=4, elapsed=32.1)
"""

from __future__ import annotations
import sys
import threading
import time
from typing import Optional

# ANSI colour codes (degrade gracefully if terminal doesn't support them)
_COLOURS = ["\033[36m", "\033[33m", "\033[35m", "\033[32m",
            "\033[34m", "\033[91m", "\033[93m", "\033[96m"]
_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_DIM     = "\033[2m"


class LiveDisplay:
    """Thread-safe live terminal display for AGNN workspace execution."""

    def __init__(self, use_colour: bool = True):
        self._lock       = threading.Lock()
        self._use_colour = use_colour and sys.stdout.isatty()
        self._ws_colours: dict = {}
        self._colour_idx = 0
        self._streaming_ws: Optional[str] = None  # workspace currently streaming tokens

    # ── colour helpers ────────────────────────────────────────────────────────

    def _colour_for(self, ws_name: str) -> str:
        if not self._use_colour:
            return ""
        if ws_name not in self._ws_colours:
            self._ws_colours[ws_name] = _COLOURS[self._colour_idx % len(_COLOURS)]
            self._colour_idx += 1
        return self._ws_colours[ws_name]

    def _label(self, ws_name: str) -> str:
        c = self._colour_for(ws_name)
        r = _RESET if self._use_colour else ""
        b = _BOLD  if self._use_colour else ""
        return f"{c}{b}[{ws_name}]{r}"

    # ── run lifecycle ─────────────────────────────────────────────────────────

    def run_start(self, prompt: str, n_workspaces: int) -> None:
        with self._lock:
            line = "─" * 64
            print(f"\n{line}")
            print(f"  AGNN  |  {n_workspaces} workspace(s)  |  {time.strftime('%H:%M:%S')}")
            print(f"  Task: {prompt[:80]}{'...' if len(prompt)>80 else ''}")
            print(f"{line}\n")
            sys.stdout.flush()

    def run_complete(self, wall_secs: float, accepted: int,
                     rejected: int, output_path: str,
                     termination_reason: str = "completed") -> None:
        with self._lock:
            line = "─" * 64
            tr = (termination_reason or "completed").lower()
            if tr in ("completed", "success", "ok"):
                status = "COMPLETE"
            elif "verification" in tr or "failed" in tr:
                status = "FAILED_VERIFICATION"
            else:
                status = "COMPLETE_WITH_WARNINGS"
            print(f"\n{line}")
            print(f"  {status}  |  {wall_secs:.1f}s  |  "
                  f"{accepted} accepted  {rejected} rejected")
            print(f"  Output → {output_path}")
            print(f"  Termination → {termination_reason or 'unknown'}")
            print(f"{line}\n")
            sys.stdout.flush()

    # ── workspace lifecycle ───────────────────────────────────────────────────

    def workspace_start(self, ws_name: str, phase: str, subgoal_desc: str) -> None:
        with self._lock:
            label = self._label(ws_name)
            dim   = _DIM   if self._use_colour else ""
            reset = _RESET if self._use_colour else ""
            print(f"{label} START  {phase}  {dim}{subgoal_desc[:60]}{reset}")
            sys.stdout.flush()

    def workspace_done(self, ws_name: str, turns: int,
                       elapsed: float, hus: float = 0.0) -> None:
        with self._lock:
            label = self._label(ws_name)
            hus_s = f"HUS={hus:.2f}  " if hus > 0 else ""
            print(f"{label} DONE   {turns} turn(s)  {hus_s}{elapsed:.1f}s")
            sys.stdout.flush()

    def workspace_failed(self, ws_name: str, reason: str) -> None:
        with self._lock:
            label = self._label(ws_name)
            print(f"{label} FAIL   {reason[:80]}")
            sys.stdout.flush()

    # ── turn-level (parallel mode) ────────────────────────────────────────────

    def turn_accepted(self, ws_name: str, phase: str, turn: int,
                      snippet: str, tis: float) -> None:
        with self._lock:
            label = self._label(ws_name)
            dim   = _DIM   if self._use_colour else ""
            reset = _RESET if self._use_colour else ""
            tis_s = f"TIS={tis:.2f}" if tis else ""
            snip  = snippet.replace("\n", " ")[:70]
            print(f"{label} t{turn:<2}  {tis_s}  {dim}{snip}...{reset}")
            sys.stdout.flush()

    def turn_rejected(self, ws_name: str, turn: int, reason: str) -> None:
        with self._lock:
            label = self._label(ws_name)
            dim   = _DIM   if self._use_colour else ""
            reset = _RESET if self._use_colour else ""
            print(f"{label} t{turn:<2}  REJECT  {dim}{reason[:60]}{reset}")
            sys.stdout.flush()

    # ── token streaming (solo / low-parallelism mode) ─────────────────────────

    def make_token_cb(self, ws_name: str, turn: int = 1):
        """
        Return an on_token callback suitable for stream_chat_completion.

        First token prints the workspace header; subsequent tokens are
        printed inline.  A newline is printed when the stream ends via
        stream_end().
        """
        state = {"first": True}

        def _on_token(token: str) -> None:
            with self._lock:
                if state["first"]:
                    label = self._label(ws_name)
                    dim   = _DIM   if self._use_colour else ""
                    reset = _RESET if self._use_colour else ""
                    print(f"\n{label} t{turn}  {dim}", end="", flush=True)
                    state["first"] = False
                sys.stdout.write(token)
                sys.stdout.flush()

        return _on_token

    def stream_end(self, ws_name: str) -> None:
        """Print a newline after a streaming turn finishes."""
        with self._lock:
            reset = _RESET if self._use_colour else ""
            print(reset, flush=True)

    # ── bus messages ──────────────────────────────────────────────────────────

    def bus_event(self, event: str, ws_name: str, detail: str = "") -> None:
        """Print a message bus event (request / answer)."""
        with self._lock:
            label = self._label(ws_name)
            dim   = _DIM   if self._use_colour else ""
            reset = _RESET if self._use_colour else ""
            print(f"{label} BUS    {event}  {dim}{detail[:70]}{reset}")
            sys.stdout.flush()

    def tool_execution(self, ws_name: str, tool_name: str) -> None:
        """Display a tool execution event."""
        with self._lock:
            label = self._label(ws_name)
            dim   = _DIM   if self._use_colour else ""
            reset = _RESET if self._use_colour else ""
            print(f"{label} TOOL   {tool_name[:70]}{reset}")
            sys.stdout.flush()

    # ── synthesis / scoring ───────────────────────────────────────────────────

    def synthesis_start(self) -> None:
        with self._lock:
            print("\n  Synthesising final output...", flush=True)

    def score_display(self, scores: dict) -> None:
        """Pretty-print auto-scorer results."""
        with self._lock:
            print("\n  ── Auto-score ──────────────────────────────────")
            for k, v in scores.items():
                if k == "overall":
                    continue
                bar = "█" * int(v) + "░" * (10 - int(v))
                print(f"  {k:<15} {bar}  {v:.1f}/10")
            overall = scores.get("overall", 0)
            print(f"  {'OVERALL':<15} {'█' * int(overall)}{'░' * (10-int(overall))}  {overall:.1f}/10")
            print("  ────────────────────────────────────────────────")
            sys.stdout.flush()
