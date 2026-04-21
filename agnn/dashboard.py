"""
AGNN Rich Live Dashboard  ·  v3.0
─────────────────────────────────
Layout (mirrors the mockup):

  ┌──────────────────────────────────┬─────────────────────┐
  │  ██╗  ██╗ ██████╗ ███╗  ██╗███╗ │  Live Log Output    │
  │  ██║  ██║██╔════╝ ████╗ ██║╚══╝ │  • event …          │
  │  ███████║██║  ███╗██╔██╗██║     │  • event …          │
  │  ██╔══██║██║   ██║██║╚████║     │                     │
  │  ██║  ██║╚██████╔╝██║ ╚███║     │                     │
  │─ Module Status ────────────────  │                     │
  │  Core Engine   ACTIVE   12ms … │                     │
  │─ Active Agents ────────────────  │                     │
  │  AGENT-01  Research  68% ████▌  │                     │
  └──────────────────────────────────┴─────────────────────┘
  │ user@agnn:~$ Running task: <prompt>                    │
  └────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# ── Colour palette ──────────────────────────────────────────────────────────────
_C = {
    "cyan":    "bold cyan",
    "magenta": "bold magenta",
    "green":   "bold green",
    "yellow":  "bold yellow",
    "red":     "bold red",
    "dim":     "dim white",
}

_PHASE_COLOURS: Dict[str, str] = {
    "research":  "cyan",
    "analysis":  "yellow",
    "draft":     "magenta",
    "review":    "green",
    "synthesis": "blue",
    "general":   "white",
}

_STATUS_COLOUR: Dict[str, str] = {
    "pending":  "dim white",
    "running":  "bold cyan",
    "complete": "bold green",
    "failed":   "bold red",
}

_MODULE_STATUS_COLOUR: Dict[str, str] = {
    "ACTIVE":   "bold cyan",
    "RUNNING":  "bold magenta",
    "ONLINE":   "bold green",
    "SYNCED":   "bold cyan",
    "IDLE":     "dim white",
    "ERROR":    "bold red",
}

_TIS_COLOUR = lambda tis: (
    "bold green" if tis >= 0.80 else
    "yellow"     if tis >= 0.65 else
    "bold red"
)

# ── ASCII banner  ────────────────────────────────────────────────────────────────
_BANNER = """\
[bold cyan] █████╗  ██████╗ ███╗  ██╗███╗  ██╗[/bold cyan]
[bold magenta]██╔══██╗██╔════╝ ████╗ ██║████╗ ██║[/bold magenta]
[bold cyan]███████║██║  ███╗██╔██╗██║██╔██╗██║[/bold cyan]
[bold magenta]██╔══██║██║   ██║██║╚████║██║╚████║[/bold magenta]
[bold cyan]██║  ██║╚██████╔╝██║ ╚███║██║ ╚███║[/bold cyan]
[bold magenta]╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚══╝╚═╝  ╚══╝[/bold magenta]
[dim]  · Adaptive Goal Negotiation Network ·[/dim]"""


# ── Data holders ────────────────────────────────────────────────────────────────
class _ModuleRow:
    def __init__(self, name: str, status: str, tasks: int = 0,
                 latency_ms: float = 0.0, notes: str = ""):
        self.name       = name
        self.status     = status
        self.tasks      = tasks
        self.latency_ms = latency_ms
        self.notes      = notes


class _AgentRow:
    def __init__(self, ws_name: str, phase: str, desc: str, model: str = ""):
        self.ws_name   = ws_name
        self.phase     = phase
        self.desc      = desc[:38]
        self.model     = model
        self.turns     = 0
        self.tis       = 0.0
        self.rejections= 0
        self.status    = "running"
        self.start_t   = time.time()
        self.snippet   = ""
        self.progress  = 0    # 0–100

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_t


# ── Main dashboard class ─────────────────────────────────────────────────────────
class RichDashboard:
    """
    Drop-in replacement for LiveDisplay.
    Matches the cyberpunk two-column mockup with banner, module table,
    agent progress table, and live log panel.
    """

    def __init__(self, use_colour: bool = True):
        self._lock       = threading.Lock()
        self._agents: Dict[str, _AgentRow]  = {}
        self._modules: Dict[str, _ModuleRow] = {
            "Core Engine":   _ModuleRow("Core Engine",   "ACTIVE",  0,  0.0, "Processing"),
            "Agent Planner": _ModuleRow("Agent Planner", "ONLINE",  0, 15.0, "Goal Setting"),
            "Memory Bank":   _ModuleRow("Memory Bank",   "SYNCED",  0,  0.0, "Vector Memory"),
            "Tool Runner":   _ModuleRow("Tool Runner",   "IDLE",    0,  0.0, "Standby"),
        }
        self._log_lines: List[str] = []
        self._MAX_LOG   = 30
        self._run_prompt= ""
        self._run_start = time.time()
        self._accepted  = 0
        self._rejected  = 0
        self._output_path = ""
        self._exec_mode   = "Solo"

        self._console = Console(highlight=False)
        self._live: Optional[Live] = None
        self._live_thread: Optional[threading.Thread] = None
        self._running = False

    # ── Rendering helpers ────────────────────────────────────────────────────────

    def _build_module_table(self) -> Table:
        t = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold white",
            expand=True,
            border_style="bright_black",
            padding=(0, 1),
        )
        t.add_column("Module",  style="bold white",  min_width=16)
        t.add_column("Status",                       min_width=10)
        t.add_column("Tasks",   justify="right",     min_width=6)
        t.add_column("Latency", justify="right",     min_width=9)
        t.add_column("Notes",   style="dim",         min_width=14)

        for m in self._modules.values():
            sc = _MODULE_STATUS_COLOUR.get(m.status, "white")
            lat = f"{m.latency_ms:.0f} ms" if m.latency_ms else "< 1 ms"
            t.add_row(
                m.name,
                Text(m.status, style=sc),
                str(m.tasks) if m.tasks else "—",
                lat,
                m.notes,
            )
        return t

    def _build_agent_table(self) -> Table:
        t = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold white",
            expand=True,
            border_style="bright_black",
            padding=(0, 1),
        )
        t.add_column("Agent ID",    style="bold",     min_width=12)
        t.add_column("Description",                   min_width=22)
        t.add_column("Status",                        min_width=12)
        t.add_column("Progress",                      min_width=22)

        for row in self._agents.values():
            sc = _STATUS_COLOUR.get(row.status, "white")

            # Build coloured progress bar (like the mockup)
            pct = min(100, max(0, row.progress))
            bar_width = 12
            filled = int(bar_width * pct / 100)
            empty  = bar_width - filled
            bar_colour = "cyan" if row.status == "running" else ("green" if row.status == "complete" else "magenta")
            bar = Text()
            bar.append(f"{pct:3d}% ", style="bold white")
            bar.append("█" * filled, style=bar_colour)
            bar.append("░" * empty,  style="dim")

            # Status text
            status_label = {
                "running":  "Working...",
                "complete": "Complete",
                "failed":   "Failed",
                "pending":  "Pending",
            }.get(row.status, row.status.capitalize())

            t.add_row(
                Text(row.ws_name[:12], style=f"bold {bar_colour}"),
                row.desc[:22],
                Text(status_label, style=sc),
                bar,
            )
        return t

    def _build_left(self) -> Panel:
        from rich.console import Group
        from rich.text import Text as RText

        banner_text = Text.from_markup(_BANNER)
        subtitle    = Text(" Advanced Goal Negotiation Network", style="bold white", justify="center")

        # Module section
        mod_rule  = Rule(title="[bold white]System Modules[/bold white]", style="bright_black")
        agent_rule= Rule(title="[bold white]Active Agents[/bold white]",  style="bright_black")

        content = Group(
            banner_text,
            subtitle,
            mod_rule,
            self._build_module_table(),
            agent_rule,
            self._build_agent_table() if self._agents else Text.from_markup("  [dim]Waiting for agents…[/dim]"),
        )
        return Panel(
            content,
            border_style="cyan",
            padding=(0, 1),
        )

    def _build_log_panel(self) -> Panel:
        lines = self._log_lines[-self._MAX_LOG:]
        lines_markup = "\n".join(lines) if lines else "[dim]Waiting for events…[/dim]"
        return Panel(
            Text.from_markup(lines_markup),
            title="[bold white]Live Log Output[/bold white]",
            border_style="magenta",
            expand=True,
            padding=(1, 1),
        )

    def _build_footer(self) -> Panel:
        elapsed = time.time() - self._run_start
        prompt  = self._run_prompt[:80] + ("…" if len(self._run_prompt) > 80 else "")
        mode_col = "cyan" if "Solo" in self._exec_mode else ("magenta" if "Hybrid" in self._exec_mode else "green")
        content = Text.from_markup(
            f"[bold bright_black]user@agnn[/bold bright_black][white]:[/white]"
            f"[cyan]~[/cyan][white]$ [/white]"
            f"{prompt}   "
            f"[dim]│[/dim] [{mode_col}]{self._exec_mode}[/{mode_col}]   "
            f"[dim]│[/dim] ⏱ {elapsed:.0f}s   "
            f"[bold green]✓{self._accepted}[/bold green]  "
            f"[bold red]✗{self._rejected}[/bold red]"
        )
        return Panel(content, border_style="bright_black", height=3)

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="body",   ratio=9),
            Layout(name="footer", ratio=1, minimum_size=3),
        )
        layout["body"].split_row(
            Layout(name="left",  ratio=3),
            Layout(name="right", ratio=2),
        )
        layout["left"].update(self._build_left())
        layout["right"].update(self._build_log_panel())
        layout["footer"].update(self._build_footer())
        return layout

    # ── Live loop ────────────────────────────────────────────────────────────────

    def _start_live(self) -> None:
        self._running = True

        def _render_loop():
            with Live(
                self._build_layout(),
                console=self._console,
                refresh_per_second=6,
                screen=False,
            ) as live:
                self._live = live
                while self._running:
                    with self._lock:
                        live.update(self._build_layout())
                    time.sleep(0.16)

        self._live_thread = threading.Thread(target=_render_loop, daemon=True)
        self._live_thread.start()

    def _stop_live(self) -> None:
        self._running = False
        if self._live_thread:
            self._live_thread.join(timeout=2.0)

    def _log(self, msg: str) -> None:
        """Append a bullet-point line to the event log (call inside lock)."""
        self._log_lines.append(f"[bold cyan]•[/bold cyan] {msg}")
        if len(self._log_lines) > self._MAX_LOG:
            self._log_lines = self._log_lines[-self._MAX_LOG:]

    # ── Public API  (mirrors LiveDisplay exactly) ─────────────────────────────────

    def run_start(self, prompt: str, n_workspaces: int) -> None:
        with self._lock:
            self._run_prompt = prompt
            self._run_start  = time.time()
            self._modules["Core Engine"].status = "ACTIVE"
            self._modules["Core Engine"].tasks  = n_workspaces
            self._log(f"[bold cyan]Initializing AGNN Framework…[/bold cyan]")
            self._log(f"Launching [bold]{n_workspaces}[/bold] workspace(s)")
        self._start_live()

    def run_complete(self, wall_secs: float, accepted: int,
                     rejected: int, output_path: str,
                     termination_reason: str = "completed") -> None:
        with self._lock:
            self._accepted    = accepted
            self._rejected    = rejected
            self._output_path = output_path
            self._modules["Core Engine"].status = "SYNCED"
            tr = (termination_reason or "completed").lower()
            if tr in ("completed", "success", "ok"):
                self._log(f"[bold green]All tasks successfully completed![/bold green]")
                banner = "[bold green]✓ Run complete[/bold green]"
            elif "verification" in tr or "failed" in tr:
                self._log(f"[bold red]Run ended with verification failure.[/bold red]")
                banner = "[bold red]✗ Run failed verification[/bold red]"
            else:
                self._log(f"[bold yellow]Run completed with warnings.[/bold yellow]")
                banner = "[bold yellow]! Run complete with warnings[/bold yellow]"
            self._log(f"[dim]Ready for next command…[/dim]")
        self._stop_live()
        self._console.print(f"\n{banner}  {wall_secs:.1f}s")
        self._console.print(f"  Accepted: {accepted}  Rejected: {rejected}")
        self._console.print(f"  Termination: {termination_reason or 'unknown'}")
        if output_path:
            self._console.print(f"  Output → [cyan]{output_path}[/cyan]")

    def workspace_start(self, ws_name: str, phase: str, subgoal_desc: str) -> None:
        with self._lock:
            self._agents[ws_name] = _AgentRow(ws_name, phase, subgoal_desc)
            self._modules["Agent Planner"].status = "RUNNING"
            self._modules["Agent Planner"].tasks += 1
            self._log(f"[cyan]{ws_name}[/cyan] started  [{_PHASE_COLOURS.get(phase,'white')}]{phase}[/{_PHASE_COLOURS.get(phase,'white')}]")

    def workspace_done(self, ws_name: str, turns: int,
                       elapsed: float, hus: float = 0.0) -> None:
        with self._lock:
            if ws_name in self._agents:
                self._agents[ws_name].status   = "complete"
                self._agents[ws_name].turns    = turns
                self._agents[ws_name].progress = 100
            hus_s = f"  HUS={hus:.2f}" if hus > 0 else ""
            self._log(f"[green]DONE[/green]  {ws_name}  {turns}t  {elapsed:.0f}s{hus_s}")

    def workspace_failed(self, ws_name: str, reason: str) -> None:
        with self._lock:
            if ws_name in self._agents:
                self._agents[ws_name].status = "failed"
            self._log(f"[red]FAIL[/red]  {ws_name}  {reason[:60]}")

    def turn_accepted(self, ws_name: str, phase: str, turn: int,
                      snippet: str, tis: float) -> None:
        with self._lock:
            self._accepted += 1
            if ws_name in self._agents:
                row = self._agents[ws_name]
                row.turns    = turn
                row.tis      = tis
                row.snippet  = snippet.replace("\n", " ")
                # Simulate progress from turn count (caps at 95 — done() sets 100)
                row.progress = min(95, turn * 12)
            self._modules["Core Engine"].tasks += 1
            tis_c = _TIS_COLOUR(tis)
            self._log(
                f"{ws_name} t{turn}  "
                f"[{tis_c}]TIS={tis:.2f}[/{tis_c}]  "
                f"[dim]{snippet[:45].replace(chr(10),' ')}[/dim]"
            )

    def turn_rejected(self, ws_name: str, turn: int, reason: str) -> None:
        with self._lock:
            self._rejected += 1
            if ws_name in self._agents:
                self._agents[ws_name].rejections += 1
            self._log(f"[red]REJECT[/red]  {ws_name} t{turn}  [dim]{reason[:55]}[/dim]")

    def make_token_cb(self, ws_name: str, turn: int = 1):
        """Suppress raw token streaming — turns appear as accepted summaries."""
        return lambda token: None

    def stream_end(self, ws_name: str) -> None:
        pass  # no-op in dashboard mode

    def bus_event(self, event: str, ws_name: str, detail: str = "") -> None:
        with self._lock:
            self._log(f"[blue]BUS[/blue]  {ws_name}  {event}  [dim]{detail[:50]}[/dim]")

    def tool_execution(self, ws_name: str, tool_name: str) -> None:
        with self._lock:
            self._modules["Tool Runner"].status  = "RUNNING"
            self._modules["Tool Runner"].tasks  += 1
            self._log(f"[bold yellow]TOOL[/bold yellow]  {ws_name} → [yellow]{tool_name}[/yellow]")

    def synthesis_start(self) -> None:
        with self._lock:
            self._log("[bold magenta]SYNTHESIS[/bold magenta] — compiling final document…")

    def score_display(self, scores: dict) -> None:
        with self._lock:
            overall = scores.get("overall", 0)
            self._log(
                f"[bold]SCORES[/bold]  overall={overall:.1f}/10  "
                + "  ".join(
                    f"{k}={v:.1f}" for k, v in scores.items()
                    if k != "overall"
                )
            )

    def set_execution_mode(self, mode: str) -> None:
        """Called from __main__ to display execution mode in footer."""
        with self._lock:
            self._exec_mode = mode
