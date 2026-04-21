"""
Tier-0/1/2 Monitoring & Visualization Layer
Uses 'rich' library to render beautiful, scrolling terminal output.
Replaces dynamic dashboard with a clean, linear, high-visibility log stream.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.style import Style
from rich.markdown import Markdown
from rich.box import ROUNDED, HEAVY

class RichLogger:
    def __init__(self, debug: bool = False):
        self.console = Console()
        self.debug_mode = debug
        
    def print_system_header(self):
        """Print the startup banner"""
        grid = Table.grid(expand=True)
        grid.add_column(justify="center", ratio=1)
        grid.add_row(
            Panel(
                Text("AGNN: Adaptive Goal Negotiation Network", justify="center", style="bold white on blue"),
                style="blue",
                box=HEAVY
            )
        )
        self.console.print(grid)

    def print_phase_header(self, phase_name: str, description: str):
        """Print a major phase transition banner"""
        self.console.print()
        self.console.print(
            Panel(
                Text(description, justify="center", style="italic white"),
                title=f"[bold yellow]PHASE: {phase_name.upper()}[/]",
                border_style="yellow",
                box=ROUNDED,
                padding=(1, 2)
            )
        )

    def print_team(self, team_members: List[Dict]):
        """Print the formed team in a nice table"""
        table = Table(title="Tier-1 Team Composition", box=ROUNDED, expand=True, border_style="magenta")
        table.add_column("Agent ID", style="bold cyan")
        table.add_column("Primary Role", style="magenta")
        table.add_column("Secondary Role(s)", style="yellow")
        table.add_column("Confidence", justify="right", style="green")
        table.add_column("Capabilities", style="dim white")

        for m in team_members:
            # Handle primary role object (might be dict from asdict())
            role_data = m.get("role")
            primary_role = role_data.get("name") if isinstance(role_data, dict) else str(role_data)

            # Handle secondary roles (list of dicts or list of Role names)
            secondary_raw = m.get("secondary_roles", []) or []
            secondary_names = []
            for sr in secondary_raw:
                if isinstance(sr, dict):
                    secondary_names.append(sr.get("name", ""))
                else:
                    secondary_names.append(getattr(sr, "name", str(sr)))
            secondary_text = ", ".join([s for s in secondary_names if s]) if secondary_names else "-"

            # Handle capabilities list converting to string
            caps = ", ".join(m.get("capabilities", []) or [])[:50]

            table.add_row(
                str(m.get("agent_id")),
                primary_role,
                secondary_text,
                f"{m.get('confidence', 0):.2f}",
                caps
            )
        self.console.print(table)
        
    def print_subgoals(self, subgoals: List[Dict]):
        """Print the decomposition plan"""
        table = Table(title="Tier-2 Goal Decomposition", box=ROUNDED, expand=True, border_style="yellow")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Phase", style="bold yellow")
        table.add_column("Objective", style="white")

        for i, sg in enumerate(subgoals, 1):
            table.add_row(
                str(i),
                sg.get("name"),
                sg.get("description")
            )
        self.console.print(table)

    def print_team_plan(self, team_plan: Dict[str, Any]):
        """Print the team plan (turn order + responsibilities)"""
        if not team_plan:
            return

        # Turn order (optional)
        if team_plan.get("turn_order"):
            order_table = Table(title="Team Plan: Turn Order", box=ROUNDED, expand=True, border_style="cyan")
            order_table.add_column("Phase", style="bold cyan")
            order_table.add_column("Order", style="white")
            for phase, order in team_plan.get("turn_order", {}).items():
                order_table.add_row(str(phase), " -> ".join(order))
            self.console.print(order_table)

        # Responsibilities
        resp_table = Table(title="Team Plan: Responsibilities", box=ROUNDED, expand=True, border_style="green")
        resp_table.add_column("Agent", style="bold green")
        resp_table.add_column("Responsibilities", style="white")
        for agent_id, items in team_plan.get("responsibilities", {}).items():
            resp_table.add_row(str(agent_id), ", ".join(items))
        self.console.print(resp_table)

    def print_chat_turn(self, agent_id: str, role: str, content: str, metrics: Dict[str, float], status: str = "ACCEPTED"):
        """Print a chat message bubble"""
        
        # Color based on status
        border_style = "green" if status == "ACCEPTED" else "red"
        
        # Header info
        tis = metrics.get('TIS', 0.0)
        header = f"[bold cyan]{agent_id}[/] ([magenta]{role}[/]) | TIS: [bold {border_style}]{tis:.3f}[/]"
        
        # Metrics summary (mini footer)
        metrics_str = (
            f"SD: {metrics.get('SD',0):.2f} | RC: {metrics.get('RC',0):.2f} | "
            f"IS: {metrics.get('IS',0):.2f} | EIC: {metrics.get('EIC',0):.2f}"
            + (f" | [bold red]REJECTED[/]" if status != "ACCEPTED" else "")
        )

        # Use Text instead of Markdown to prevent truncation
        from rich.text import Text
        content_text = Text(content, style="white")
        
        self.console.print(
            Panel(
                content_text,  # Use Text instead of Markdown
                title=header,
                subtitle=metrics_str,
                border_style=border_style,
                box=ROUNDED,
                padding=(0, 1),
                expand=True  # Force full width to prevent truncation
            )
        )

    def log_rejection(self, agent_id: str, reasons: List[str], metrics: Dict[str, float]):
        """Log a rejection event in the console"""
        reason_text = ", ".join(reasons) if reasons else "unspecified"
        tis = metrics.get("TIS", 0.0)
        msg = f"[bold red]REJECTED[/] {agent_id} | TIS: {tis:.3f} | Reasons: {reason_text}"
        self.console.print(msg)

    def print_negotiation_event(self, event_type: str, details: Dict):
        """Print low-profile negotiation events (logs)"""
        # Only print if relevant or crucial
        if event_type == "Winner":
            text = f"[bold green]WINNER:[/] {details.get('agent')} (Bid: {details.get('bid'):.2f})"
            self.console.print(f"  Shared Goal Negotiation > {text}")
        elif event_type == "Bid" and self.debug_mode:
             self.console.print(f"  [dim]Bid: {details.get('agent')} offered {details.get('confidence'):.2f}[/dim]")

    def print_error(self, message: str):
        self.console.print(f"[bold red]ERROR:[/] {message}")

    def log_system(self, message: str):
        """Generic system log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[dim]{timestamp}[/] [bold white]SYSTEM:[/] {message}")
