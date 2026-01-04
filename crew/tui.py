"""Basic TUI skeleton for crew using Textual."""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual.app import App
from textual.binding import Binding
from textual.widgets import Tree


@dataclass
class TicketData:
    """Data associated with a ticket node in the tree."""

    id: str
    title: str
    status: str
    deps: list[str]
    wave: int  # -1 = blocked/cyclic, 0 = ready, 1+ = later waves


def compute_waves(tickets: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Compute wave numbers for tickets using topological sort (Kahn's algorithm).

    This reuses the logic from cmd_queue in cli.py.

    Args:
        tickets: Dict mapping ticket ID to ticket data (must have 'deps' and 'status')

    Returns:
        Dict mapping ticket ID to wave number:
        - Wave 0: Ready now (no open dependencies)
        - Wave 1+: Waiting on tickets in earlier waves
        - Wave -1: Blocked (cyclic dependencies)
    """
    # Only consider open tickets
    open_tickets = {tid: t for tid, t in tickets.items() if t.get("status") == "open"}

    if not open_tickets:
        return {}

    # Compute waves using Kahn's algorithm
    waves: dict[str, int] = {}
    remaining = set(open_tickets.keys())

    wave_num = 0
    while remaining:
        # Find tickets with no unresolved open dependencies
        ready_this_wave = []
        for tid in remaining:
            deps = open_tickets[tid].get("deps", [])
            # A ticket is ready if all its deps are either closed or in previous waves
            open_deps = [d for d in deps if d in open_tickets and d not in waves]
            if not open_deps:
                ready_this_wave.append(tid)

        if not ready_this_wave:
            # Remaining tickets are in a cycle - mark as blocked
            for tid in remaining:
                waves[tid] = -1
            break

        for tid in ready_this_wave:
            waves[tid] = wave_num
            remaining.discard(tid)

        wave_num += 1

    return waves


def fetch_tickets() -> list[dict[str, Any]]:
    """Fetch tickets from tk command.

    Returns:
        List of ticket dicts, or empty list on error.
    """
    try:
        result = subprocess.run(
            ["tk", "query", "--json"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return []
        return json.loads(result.stdout) if result.stdout.strip() else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []


class TicketTree(Tree[TicketData]):
    """A tree widget showing tickets organized by dependency waves."""

    DEFAULT_CSS = """
    TicketTree {
        width: 100%;
        height: 100%;
    }
    TicketTree > .tree--label {
        color: $text;
    }
    TicketTree > .tree--guides {
        color: $surface-lighten-2;
    }
    """

    def __init__(
        self,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        """Initialize the ticket tree.

        Args:
            name: The name of the widget.
            id: The ID of the widget in the DOM.
            classes: CSS classes for the widget.
        """
        super().__init__(
            "Tickets",
            data=None,
            name=name,
            id=id,
            classes=classes,
        )
        self.show_root = True

    def on_mount(self) -> None:
        """Load tickets when the widget is mounted."""
        self.refresh_tickets()

    def refresh_tickets(self) -> None:
        """Refresh the ticket tree from tk query."""
        self.clear()

        # Fetch tickets
        tickets_data = fetch_tickets()
        if not tickets_data:
            no_tickets = self.root.add("No tickets", data=None)
            no_tickets.allow_expand = False
            return

        # Build ticket lookup
        tickets: dict[str, dict[str, Any]] = {}
        for t in tickets_data:
            tid = t.get("id", "")
            if tid:
                tickets[tid] = t

        # Compute waves
        waves = compute_waves(tickets)

        # Group tickets by wave
        wave_tickets: dict[int, list[str]] = defaultdict(list)
        for tid, wave in waves.items():
            wave_tickets[wave].append(tid)

        # Also include closed tickets under a separate category
        closed_tickets = [tid for tid, t in tickets.items() if t.get("status") != "open"]

        # Ready now (wave 0)
        if 0 in wave_tickets:
            ready_node = self.root.add(
                Text("Ready", style="bold green"),
                data=None,
            )
            ready_node.expand()
            for tid in sorted(wave_tickets[0]):
                t = tickets[tid]
                title = t.get("title", tid)
                ticket_data = TicketData(
                    id=tid,
                    title=title,
                    status=t.get("status", ""),
                    deps=t.get("deps", []),
                    wave=0,
                )
                label = Text()
                label.append(tid, style="cyan")
                label.append(f" {title}")
                node = ready_node.add(label, data=ticket_data)
                node.allow_expand = False

        # Next (wave 1)
        if 1 in wave_tickets:
            next_node = self.root.add(
                Text("Next", style="bold yellow"),
                data=None,
            )
            next_node.expand()
            for tid in sorted(wave_tickets[1]):
                t = tickets[tid]
                title = t.get("title", tid)
                deps = t.get("deps", [])
                open_deps = [d for d in deps if d in tickets and tickets[d].get("status") == "open"]
                ticket_data = TicketData(
                    id=tid,
                    title=title,
                    status=t.get("status", ""),
                    deps=deps,
                    wave=1,
                )
                label = Text()
                label.append(tid, style="cyan")
                label.append(f" {title}")
                if open_deps:
                    label.append(f" (→ {', '.join(open_deps)})", style="dim")
                node = next_node.add(label, data=ticket_data)
                node.allow_expand = False

        # Later (waves 2+)
        later_waves = sorted([w for w in wave_tickets.keys() if w >= 2])
        if later_waves:
            later_node = self.root.add(
                Text("Later", style="bold blue"),
                data=None,
            )
            for wave in later_waves:
                for tid in sorted(wave_tickets[wave]):
                    t = tickets[tid]
                    title = t.get("title", tid)
                    deps = t.get("deps", [])
                    open_deps = [d for d in deps if d in tickets and tickets[d].get("status") == "open"]
                    ticket_data = TicketData(
                        id=tid,
                        title=title,
                        status=t.get("status", ""),
                        deps=deps,
                        wave=wave,
                    )
                    label = Text()
                    label.append(tid, style="cyan")
                    label.append(f" {title}")
                    if open_deps:
                        label.append(f" (→ {', '.join(open_deps)})", style="dim")
                    node = later_node.add(label, data=ticket_data)
                    node.allow_expand = False

        # Blocked (wave -1)
        if -1 in wave_tickets:
            blocked_node = self.root.add(
                Text("Blocked", style="bold red"),
                data=None,
            )
            for tid in sorted(wave_tickets[-1]):
                t = tickets[tid]
                title = t.get("title", tid)
                deps = t.get("deps", [])
                open_deps = [d for d in deps if d in tickets and tickets[d].get("status") == "open"]
                ticket_data = TicketData(
                    id=tid,
                    title=title,
                    status=t.get("status", ""),
                    deps=deps,
                    wave=-1,
                )
                label = Text()
                label.append(tid, style="cyan")
                label.append(f" {title}")
                if open_deps:
                    label.append(f" (→ {', '.join(open_deps)})", style="dim")
                node = blocked_node.add(label, data=ticket_data)
                node.allow_expand = False

        # Closed tickets (collapsed by default)
        if closed_tickets:
            closed_node = self.root.add(
                Text(f"Closed ({len(closed_tickets)})", style="dim"),
                data=None,
            )
            for tid in sorted(closed_tickets):
                t = tickets[tid]
                title = t.get("title", tid)
                ticket_data = TicketData(
                    id=tid,
                    title=title,
                    status="closed",
                    deps=t.get("deps", []),
                    wave=-2,  # Special value for closed
                )
                label = Text()
                label.append(tid, style="dim cyan")
                label.append(f" {title}", style="dim")
                node = closed_node.add(label, data=ticket_data)
                node.allow_expand = False

        # Expand root by default
        self.root.expand()


class CrewApp(App):
    """A Textual app for managing crew agents."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
    ]

    def compose(self):
        """Compose the app layout."""
        yield TicketTree(id="ticket-tree")

    def action_refresh(self) -> None:
        """Refresh the ticket tree."""
        tree = self.query_one("#ticket-tree", TicketTree)
        tree.refresh_tickets()


def main():
    """Run the TUI application."""
    app = CrewApp()
    app.run()


if __name__ == "__main__":
    main()
