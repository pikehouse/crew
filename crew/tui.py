"""TUI for crew using Textual.

This module provides both agent status view (DataTable) and ticket queue view (Tree).
"""

from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, Center, Middle
from textual.screen import ModalScreen
from textual.widgets import DataTable, Header, Footer, Tree, Static, Button, Label

from crew.runner import shutdown_agent, find_claude_process
from crew.state import load_state, save_state
from crew.git import run_git, GitError
from crew.crew_logging import read_log_tail

if TYPE_CHECKING:
    from crew.cli import BackgroundRunner


@dataclass
class TicketData:
    """Data associated with a ticket node in the tree."""

    id: str
    title: str
    status: str
    deps: list[str]
    wave: int  # -1 = blocked/cyclic, 0 = ready, 1+ = later waves
    description: str = ""
    assignee: str = ""
    created: str = ""
    ticket_type: str = ""
    priority: int = 0


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
            ["tk", "query"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return []
        # tk query outputs JSONL (one JSON object per line)
        if not result.stdout.strip():
            return []
        lines = result.stdout.strip().split('\n')
        return [json.loads(line) for line in lines if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def make_ticket_data(ticket: dict[str, Any], wave: int) -> TicketData:
    """Create a TicketData from a ticket dictionary.

    Args:
        ticket: Ticket dict from tk query.
        wave: Wave number for the ticket.

    Returns:
        TicketData instance with all fields populated.
    """
    return TicketData(
        id=ticket.get("id", ""),
        title=ticket.get("title", ""),
        status=ticket.get("status", ""),
        deps=ticket.get("deps", []),
        wave=wave,
        description=ticket.get("description", ""),
        assignee=ticket.get("assignee", ""),
        created=ticket.get("created", ""),
        ticket_type=ticket.get("type", ""),
        priority=ticket.get("priority", 0),
    )


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

    def refresh_tickets(self, zoomed_ticket: str | None = None) -> None:
        """Refresh the ticket tree from tk query.

        Args:
            zoomed_ticket: If set, show the dependency tree for this ticket only.
        """
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

        # If zoomed, show the ticket's dependency tree
        if zoomed_ticket is not None:
            self._show_zoomed_view(zoomed_ticket, tickets)
            return

        # Otherwise show the normal wave-based view
        self._show_wave_view(tickets)

    def _show_zoomed_view(
        self, zoomed_ticket: str, tickets: dict[str, dict[str, Any]]
    ) -> None:
        """Show the zoomed dependency tree for a specific ticket.

        Args:
            zoomed_ticket: The ticket ID to focus on.
            tickets: All tickets data.
        """
        if zoomed_ticket not in tickets:
            no_ticket = self.root.add(
                Text(f"Ticket {zoomed_ticket} not found", style="red"),
                data=None,
            )
            no_ticket.allow_expand = False
            return

        t = tickets[zoomed_ticket]
        title = t.get("title", zoomed_ticket)
        status = t.get("status", "")
        deps = t.get("deps", [])

        # Change root label to indicate zoomed mode
        self.root.set_label(Text(f"Zoomed: {zoomed_ticket}", style="bold magenta"))

        # Create the focused ticket as the main node
        status_style = "green" if status == "open" else "dim"
        main_label = Text()
        main_label.append(zoomed_ticket, style=f"bold cyan")
        main_label.append(f" {title}", style="bold")
        main_label.append(f" [{status}]", style=status_style)

        main_data = make_ticket_data(t, wave=0)
        main_node = self.root.add(main_label, data=main_data)
        main_node.expand()

        # Add dependencies (tickets this one depends on)
        if deps:
            deps_node = main_node.add(
                Text("Dependencies (blocks this)", style="yellow"),
                data=None,
            )
            deps_node.expand()
            for dep_id in sorted(deps):
                if dep_id in tickets:
                    dep_t = tickets[dep_id]
                    dep_title = dep_t.get("title", dep_id)
                    dep_status = dep_t.get("status", "")
                    dep_deps = dep_t.get("deps", [])

                    dep_style = "dim" if dep_status != "open" else ""
                    label = Text()
                    label.append(dep_id, style=f"cyan {dep_style}".strip())
                    label.append(f" {dep_title}", style=dep_style)
                    label.append(f" [{dep_status}]", style="dim")

                    dep_data = make_ticket_data(dep_t, wave=0)
                    node = deps_node.add(label, data=dep_data)
                    node.allow_expand = False
                else:
                    # Dependency not found (external or deleted)
                    label = Text()
                    label.append(dep_id, style="dim red")
                    label.append(" (not found)", style="dim")
                    node = deps_node.add(label, data=None)
                    node.allow_expand = False

        # Find dependents (tickets that depend on this one)
        dependents = []
        for tid, ticket in tickets.items():
            if zoomed_ticket in ticket.get("deps", []):
                dependents.append(tid)

        if dependents:
            dependents_node = main_node.add(
                Text("Dependents (blocked by this)", style="blue"),
                data=None,
            )
            dependents_node.expand()
            for dep_id in sorted(dependents):
                dep_t = tickets[dep_id]
                dep_title = dep_t.get("title", dep_id)
                dep_status = dep_t.get("status", "")

                dep_style = "dim" if dep_status != "open" else ""
                label = Text()
                label.append(dep_id, style=f"cyan {dep_style}".strip())
                label.append(f" {dep_title}", style=dep_style)
                label.append(f" [{dep_status}]", style="dim")

                dep_data = make_ticket_data(dep_t, wave=0)
                node = dependents_node.add(label, data=dep_data)
                node.allow_expand = False

        # Expand root
        self.root.expand()

    def _show_wave_view(self, tickets: dict[str, dict[str, Any]]) -> None:
        """Show the normal wave-based ticket view.

        Args:
            tickets: All tickets data.
        """
        # Reset root label
        self.root.set_label("Tickets")

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
                ticket_data = make_ticket_data(t, wave=0)
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
                ticket_data = make_ticket_data(t, wave=1)
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
                    ticket_data = make_ticket_data(t, wave=wave)
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
                ticket_data = make_ticket_data(t, wave=-1)
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
                ticket_data = make_ticket_data(t, wave=-2)
                label = Text()
                label.append(tid, style="dim cyan")
                label.append(f" {title}", style="dim")
                node = closed_node.add(label, data=ticket_data)
                node.allow_expand = False

        # Expand root by default
        self.root.expand()


class EventLog(Static):
    """A widget to display recent runner events."""

    DEFAULT_CSS = """
    EventLog {
        height: 5;
        width: 100%;
        border-top: solid $primary;
        padding: 0 1;
        background: $surface;
    }
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._events: list[str] = []
        self._max_events = 4

    def add_event(self, event_text: str) -> None:
        """Add an event to the log."""
        self._events.append(event_text)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        self._update_display()

    def _update_display(self) -> None:
        """Update the display with current events."""
        if self._events:
            self.update("\n".join(self._events))
        else:
            self.update("[dim]No recent events[/dim]")


class AgentDetailPanel(Static):
    """A panel showing git status, diff stat, and log tail for the selected agent."""

    DEFAULT_CSS = """
    AgentDetailPanel {
        width: 100%;
        height: 100%;
        border-left: solid $primary;
        padding: 0 1;
        background: $surface;
        overflow-y: auto;
    }
    """

    def __init__(self, project_root: Path | None = None, **kwargs) -> None:
        """Initialize the agent detail panel.

        Args:
            project_root: Project root path for git and log operations.
        """
        super().__init__(**kwargs)
        self._project_root = project_root or Path.cwd()
        self._current_agent: str | None = None

    def update_agent(self, agent_name: str | None, worktree: Path | None) -> None:
        """Update the panel to show details for the given agent.

        Args:
            agent_name: Name of the agent to display, or None to clear.
            worktree: Path to the agent's worktree.
        """
        self._current_agent = agent_name

        if not agent_name or not worktree:
            self.update(Text("[dim]No agent selected[/dim]"))
            return

        # Build the detail content
        content = Text()

        # Header
        content.append(f"Agent: {agent_name}\n", style="bold cyan")
        content.append("\n")

        # Git status section
        content.append("Git Status\n", style="bold underline")
        try:
            status = run_git("status", "--short", cwd=worktree)
            if status:
                for line in status.split("\n")[:10]:  # Limit to 10 lines
                    # Color code by status
                    if line.startswith("M"):
                        content.append(line + "\n", style="yellow")
                    elif line.startswith("A"):
                        content.append(line + "\n", style="green")
                    elif line.startswith("D"):
                        content.append(line + "\n", style="red")
                    elif line.startswith("??"):
                        content.append(line + "\n", style="dim")
                    else:
                        content.append(line + "\n")
            else:
                content.append("[dim]No changes[/dim]\n")
        except GitError as e:
            content.append(f"[red]Error: {e}[/red]\n")
        content.append("\n")

        # Diff stat section
        content.append("Diff Stat\n", style="bold underline")
        try:
            diff_stat = run_git("diff", "--stat", "HEAD", cwd=worktree)
            if diff_stat:
                lines = diff_stat.split("\n")
                # Show file changes (limit to 8 lines)
                for line in lines[:8]:
                    if "|" in line:
                        # File change line - color insertions green, deletions red
                        parts = line.split("|")
                        if len(parts) == 2:
                            content.append(parts[0] + "|")
                            stat_part = parts[1]
                            # Color the +/- indicators
                            colored_stat = ""
                            for char in stat_part:
                                if char == "+":
                                    content.append("+", style="green")
                                elif char == "-":
                                    content.append("-", style="red")
                                else:
                                    content.append(char)
                            content.append("\n")
                        else:
                            content.append(line + "\n")
                    else:
                        content.append(line + "\n")
                # Show summary if present
                if lines and ("insertion" in lines[-1] or "deletion" in lines[-1]):
                    summary = lines[-1]
                    content.append(summary + "\n", style="bold")
            else:
                content.append("[dim]No uncommitted changes[/dim]\n")
        except GitError as e:
            content.append(f"[red]Error: {e}[/red]\n")
        content.append("\n")

        # Log tail section
        content.append("Recent Output\n", style="bold underline")
        log_content = read_log_tail(agent_name, lines=15, project_root=self._project_root)
        if log_content:
            # Truncate long lines and limit content
            log_lines = log_content.split("\n")
            for line in log_lines[-15:]:
                # Truncate long lines
                if len(line) > 80:
                    line = line[:77] + "..."
                content.append(line + "\n", style="dim")
        else:
            content.append("[dim]No logs available[/dim]\n")

        self.update(content)

    def refresh_current(self) -> None:
        """Refresh the display for the current agent."""
        if self._current_agent:
            # Need to get the worktree from state
            state = load_state(self._project_root)
            agent = state.agents.get(self._current_agent)
            if agent:
                self.update_agent(agent.name, agent.worktree)


class HelpOverlay(Static):
    """A modal overlay showing all keybindings."""

    DEFAULT_CSS = """
    HelpOverlay {
        width: 50;
        height: auto;
        background: $surface;
        border: round $primary;
        padding: 1 2;
    }
    """

    def __init__(self, bindings: list[tuple[str, str]], **kwargs) -> None:
        """Initialize the help overlay.

        Args:
            bindings: List of (key, description) tuples.
        """
        super().__init__(**kwargs)
        self._bindings = bindings

    def compose(self) -> ComposeResult:
        """Compose the help content."""
        yield Static(self._render_help())

    def _render_help(self) -> Text:
        """Render the help text."""
        text = Text()
        text.append("Keybindings\n", style="bold underline")
        text.append("\n")
        for key, description in self._bindings:
            text.append(f"  {key:>10}", style="bold cyan")
            text.append(f"  {description}\n")
        text.append("\n")
        text.append("Press ", style="dim")
        text.append("?", style="bold cyan")
        text.append(" to close", style="dim")
        return text


class MergeConfirmScreen(ModalScreen[bool]):
    """Modal screen for confirming agent merge."""

    BINDINGS = [
        Binding("y", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
        Binding("escape", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    MergeConfirmScreen {
        align: center middle;
    }

    MergeConfirmScreen > Vertical {
        width: 60;
        height: auto;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    MergeConfirmScreen > Vertical > Label {
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }

    MergeConfirmScreen > Vertical > Horizontal {
        width: 100%;
        height: auto;
        align: center middle;
    }

    MergeConfirmScreen > Vertical > Horizontal > Button {
        margin: 0 1;
    }
    """

    def __init__(self, agent_name: str, task_id: str | None = None) -> None:
        """Initialize the confirmation screen.

        Args:
            agent_name: Name of the agent to merge.
            task_id: Optional task ID the agent was working on.
        """
        super().__init__()
        self.agent_name = agent_name
        self.task_id = task_id

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog."""
        task_info = f" ({self.task_id})" if self.task_id else ""
        with Vertical():
            yield Label(f"Merge agent [bold]{self.agent_name}[/bold]{task_info}?")
            yield Label("This will merge the agent's branch to main and remove the worktree.")
            with Horizontal():
                yield Button("Yes (y)", variant="primary", id="yes")
                yield Button("No (n)", variant="default", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "yes":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_confirm(self) -> None:
        """Confirm the merge."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Cancel the merge."""
        self.dismiss(False)


class ConfirmKillScreen(ModalScreen[bool]):
    """Modal screen to confirm killing an agent."""

    DEFAULT_CSS = """
    ConfirmKillScreen {
        align: center middle;
    }

    ConfirmKillScreen > Vertical {
        width: 50;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: thick $error;
    }

    ConfirmKillScreen > Vertical > Label {
        width: 100%;
        content-align: center middle;
        margin-bottom: 1;
    }

    ConfirmKillScreen > Vertical > #agent-name {
        text-style: bold;
        color: $warning;
    }

    ConfirmKillScreen > Vertical > Center {
        width: 100%;
        height: auto;
    }

    ConfirmKillScreen > Vertical > Center > Horizontal {
        width: auto;
        height: auto;
    }

    ConfirmKillScreen Button {
        margin: 0 1;
    }
    """

    BINDINGS = [
        Binding("y", "confirm", "Yes"),
        Binding("n", "cancel", "No"),
        Binding("escape", "cancel", "Cancel"),
    ]

    def __init__(self, agent_name: str) -> None:
        """Initialize the confirm screen.

        Args:
            agent_name: Name of the agent to kill.
        """
        super().__init__()
        self.agent_name = agent_name

    def compose(self) -> ComposeResult:
        """Compose the confirmation dialog."""
        with Vertical():
            yield Label("Kill agent?")
            yield Label(self.agent_name, id="agent-name")
            yield Label("This will stop the agent process.")
            with Center():
                with Horizontal():
                    yield Button("Yes (y)", variant="error", id="yes")
                    yield Button("No (n)", variant="primary", id="no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press."""
        if event.button.id == "yes":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_confirm(self) -> None:
        """Confirm the action."""
        self.dismiss(True)

    def action_cancel(self) -> None:
        """Cancel the action."""
        self.dismiss(False)


class TicketDetailsScreen(ModalScreen[None]):
    """Modal screen displaying full ticket details."""

    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("enter", "close", "Close"),
        Binding("q", "close", "Close"),
    ]

    DEFAULT_CSS = """
    TicketDetailsScreen {
        align: center middle;
    }

    TicketDetailsScreen > Vertical {
        width: 70;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $primary;
        padding: 1 2;
    }

    TicketDetailsScreen > Vertical > Static {
        width: 100%;
        margin-bottom: 1;
    }

    TicketDetailsScreen > Vertical > #ticket-title {
        text-style: bold;
        color: $text;
        margin-bottom: 1;
    }

    TicketDetailsScreen > Vertical > #ticket-meta {
        color: $text-muted;
    }

    TicketDetailsScreen > Vertical > #ticket-description {
        margin-top: 1;
        padding: 1;
        background: $surface-darken-1;
    }

    TicketDetailsScreen > Vertical > #close-hint {
        margin-top: 1;
        text-align: center;
        color: $text-muted;
    }
    """

    def __init__(self, ticket_data: TicketData) -> None:
        """Initialize the ticket details screen.

        Args:
            ticket_data: The ticket data to display.
        """
        super().__init__()
        self.ticket_data = ticket_data

    def compose(self) -> ComposeResult:
        """Compose the ticket details dialog."""
        t = self.ticket_data
        with Vertical():
            # Title
            yield Static(f"[bold cyan]{t.id}[/bold cyan] {t.title}", id="ticket-title")

            # Metadata
            meta_lines = []
            meta_lines.append(f"[bold]Status:[/bold] {t.status}")
            if t.deps:
                meta_lines.append(f"[bold]Dependencies:[/bold] {', '.join(t.deps)}")
            else:
                meta_lines.append("[bold]Dependencies:[/bold] None")

            yield Static("\n".join(meta_lines), id="ticket-meta")

            # Close hint
            yield Static("[dim]Press Enter, Esc, or q to close[/dim]", id="close-hint")

    def action_close(self) -> None:
        """Close the details screen."""
        self.dismiss(None)


class CrewApp(App):
    """A Textual app for managing crew agents."""

    BINDINGS = [
        Binding("q", "quit_or_unzoom", "Quit/Unzoom"),
        Binding("r", "run_runner", "Run"),
        Binding("s", "stop_runner", "Stop"),
        Binding("R", "refresh", "Refresh"),
        Binding("t", "toggle_view", "Toggle View"),
        Binding("m", "merge_agent", "Merge"),
        Binding("k", "kill_agent", "Kill"),
        Binding("enter", "zoom_in", "Zoom In", show=False),
        Binding("question_mark", "toggle_help", "Help"),
    ]

    CSS = """
    #main-container {
        height: 1fr;
    }
    #ticket-tree {
        width: 40%;
        height: 100%;
        border-right: solid $primary;
    }
    #right-panel {
        width: 60%;
        height: 100%;
    }
    #agents-container {
        height: 50%;
        width: 100%;
    }
    #agents {
        width: 100%;
        height: 100%;
    }
    #agent-detail {
        height: 50%;
        width: 100%;
        border-top: solid $primary;
    }
    #help-overlay-container {
        width: 100%;
        height: 100%;
        align: center middle;
        display: none;
    }
    #help-overlay-container.visible {
        display: block;
    }
    """

    def __init__(
        self,
        runner: BackgroundRunner | None = None,
        project_root: Path | None = None,
    ) -> None:
        """Initialize the app.

        Args:
            runner: Optional BackgroundRunner instance to poll for events.
            project_root: Project root path for merge operations.
        """
        super().__init__()
        self._show_agents = True  # Start with agents view
        self._zoomed_ticket: str | None = None  # Ticket ID when zoomed
        self._runner = runner
        self._project_root = project_root or Path.cwd()
        self._poll_timer = None
        self._help_visible = False

    def compose(self) -> ComposeResult:
        """Compose the app layout with split view."""
        yield Header()
        with Horizontal(id="main-container"):
            yield TicketTree(id="ticket-tree")
            with Vertical(id="right-panel"):
                with Vertical(id="agents-container"):
                    yield DataTable(id="agents")
                yield AgentDetailPanel(project_root=self._project_root, id="agent-detail")
        yield EventLog(id="event-log")
        yield Footer()
        # Help overlay (hidden by default)
        bindings = [
            ("?", "Show/hide this help"),
            ("q", "Quit (or unzoom if zoomed)"),
            ("r", "Run the background runner"),
            ("s", "Stop the background runner"),
            ("R", "Refresh views"),
            ("t", "Toggle agents/tickets view"),
            ("m", "Merge selected agent"),
            ("k", "Kill selected agent"),
            ("Enter", "Show ticket details / zoom in"),
        ]
        with Center(id="help-overlay-container"):
            with Middle():
                yield HelpOverlay(bindings, id="help-overlay")

    def on_mount(self) -> None:
        """Set up the app when it mounts."""
        # Set up agents table
        table = self.query_one("#agents", DataTable)
        table.add_columns("Name", "Task", "Status", "Proc", "Steps", "Cost")
        self._refresh_agents(table)

        # Initialize event log
        event_log = self.query_one("#event-log", EventLog)
        event_log._update_display()

        # Start polling for runner events if runner is provided
        if self._runner:
            self._poll_timer = self.set_interval(0.5, self._poll_runner_events)

        # Update detail panel with first agent if any exist
        self._update_detail_panel_for_current_selection()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Handle when a row is highlighted (cursor moved) in the DataTable."""
        self._update_detail_panel_for_current_selection()

    def _update_detail_panel_for_current_selection(self) -> None:
        """Update the agent detail panel based on current table selection."""
        table = self.query_one("#agents", DataTable)
        detail_panel = self.query_one("#agent-detail", AgentDetailPanel)

        cursor_row = table.cursor_row
        if cursor_row is None or cursor_row < 0:
            detail_panel.update_agent(None, None)
            return

        state = load_state(self._project_root)
        agents_list = list(state.agents.values())

        if cursor_row >= len(agents_list):
            detail_panel.update_agent(None, None)
            return

        agent = agents_list[cursor_row]
        detail_panel.update_agent(agent.name, agent.worktree)

    def _poll_runner_events(self) -> None:
        """Poll for events from BackgroundRunner and update UI."""
        if not self._runner:
            return

        events = self._runner.drain_events()
        if not events:
            return

        event_log = self.query_one("#event-log", EventLog)
        needs_refresh = False

        for event in events:
            # Format event for display
            if event.type == "step":
                event_text = f"[cyan]↻[/cyan] {event.agent_name}: {event.message}"
                needs_refresh = True
            elif event.type == "done":
                event_text = f"[green]✓[/green] {event.agent_name} completed {event.message}"
                needs_refresh = True
            elif event.type == "merged":
                event_text = f"[green]⎇[/green] {event.agent_name} merged {event.message}"
                needs_refresh = True
            elif event.type == "assigned":
                event_text = f"[yellow]→[/yellow] {event.agent_name} assigned {event.message}"
                needs_refresh = True
            elif event.type == "error":
                event_text = f"[red]✗[/red] {event.agent_name}: {event.message}"
                needs_refresh = True
            elif event.type == "stopped":
                event_text = f"[blue]■[/blue] Runner stopped: {event.message}"
            else:
                event_text = f"{event.type}: {event.agent_name} {event.message}"

            event_log.add_event(event_text)

        # Refresh views if agent state changed
        if needs_refresh:
            table = self.query_one("#agents", DataTable)
            self._refresh_agents(table)
            # Also refresh the agent detail panel
            self._update_detail_panel_for_current_selection()
            # Also refresh tickets on done/merged since task might be closed
            if any(e.type in ("done", "merged") for e in events):
                tree = self.query_one("#ticket-tree", TicketTree)
                tree.refresh_tickets()

    def _refresh_agents(self, table: DataTable) -> None:
        """Refresh the agents table with current state."""
        table.clear()
        state = load_state()
        for agent in state.agents.values():
            cost_str = f"${agent.total_cost_usd:.2f}" if agent.total_cost_usd > 0 else "-"
            # Check if Claude process is actually running for this agent
            proc_indicator = "-"
            if agent.status in ("ready", "working"):
                pid = find_claude_process(agent)
                if pid:
                    proc_indicator = Text("●", style="green")  # Process running
                else:
                    proc_indicator = Text("⚠", style="bold red")  # Dead process warning
            table.add_row(
                agent.name,
                agent.task or "-",
                agent.status,
                proc_indicator,
                str(agent.step_count),
                cost_str,
            )

    def action_refresh(self) -> None:
        """Refresh all views."""
        table = self.query_one("#agents", DataTable)
        tree = self.query_one("#ticket-tree", TicketTree)
        self._refresh_agents(table)
        self._update_detail_panel_for_current_selection()
        tree.refresh_tickets(zoomed_ticket=self._zoomed_ticket)

    def action_run_runner(self) -> None:
        """Start the BackgroundRunner if available and not running."""
        if not self._runner:
            event_log = self.query_one("#event-log", EventLog)
            event_log.add_event("[yellow]![/yellow] No runner available")
            return

        if self._runner.is_running:
            event_log = self.query_one("#event-log", EventLog)
            event_log.add_event("[dim]Already running[/dim]")
            return

        started = self._runner.start()
        event_log = self.query_one("#event-log", EventLog)
        if started:
            event_log.add_event("[green]▶[/green] Runner started")
        else:
            event_log.add_event("[yellow]![/yellow] Failed to start runner")

    def action_stop_runner(self) -> None:
        """Stop the BackgroundRunner if running."""
        if not self._runner:
            event_log = self.query_one("#event-log", EventLog)
            event_log.add_event("[yellow]![/yellow] No runner available")
            return

        if not self._runner.is_running:
            event_log = self.query_one("#event-log", EventLog)
            event_log.add_event("[dim]Not running[/dim]")
            return

        self._runner.stop()
        event_log = self.query_one("#event-log", EventLog)
        event_log.add_event("[blue]■[/blue] Runner stopped")

    def action_toggle_view(self) -> None:
        """Toggle between agents and tickets view."""
        right_panel = self.query_one("#right-panel", Vertical)
        tree = self.query_one("#ticket-tree", TicketTree)
        self._show_agents = not self._show_agents

        if self._show_agents:
            right_panel.display = True
            tree.display = False
        else:
            right_panel.display = False
            tree.display = True
            tree.refresh_tickets(zoomed_ticket=self._zoomed_ticket)

    def action_zoom_in(self) -> None:
        """Show details for the selected ticket and zoom into its dependency tree."""
        if self._show_agents:
            return  # Only works in ticket view

        tree = self.query_one("#ticket-tree", TicketTree)
        cursor_node = tree.cursor_node
        if cursor_node is None:
            return

        # Get the ticket data from the selected node
        ticket_data = cursor_node.data
        if ticket_data is None:
            return  # Category node selected, not a ticket

        # Show the ticket details screen
        self.push_screen(TicketDetailsScreen(ticket_data))

        # Also zoom into this ticket
        self._zoomed_ticket = ticket_data.id
        tree.refresh_tickets(zoomed_ticket=self._zoomed_ticket)

    def action_toggle_help(self) -> None:
        """Toggle the help overlay visibility."""
        container = self.query_one("#help-overlay-container")
        self._help_visible = not self._help_visible
        if self._help_visible:
            container.add_class("visible")
        else:
            container.remove_class("visible")

    def action_merge_agent(self) -> None:
        """Merge the selected agent's branch to main."""
        if not self._show_agents:
            return  # Only works in agents view

        table = self.query_one("#agents", DataTable)
        cursor_row = table.cursor_row

        # Get the agent name from the selected row
        if cursor_row is None or cursor_row < 0:
            return

        state = load_state(self._project_root)
        agents_list = list(state.agents.values())

        if cursor_row >= len(agents_list):
            return

        agent = agents_list[cursor_row]

        # Only allow merging agents that are done or have work to merge
        if agent.status not in ("done", "working", "ready"):
            event_log = self.query_one("#event-log", EventLog)
            event_log.add_event(f"[yellow]![/yellow] Cannot merge {agent.name}: status is {agent.status}")
            return

        if not agent.branch:
            event_log = self.query_one("#event-log", EventLog)
            event_log.add_event(f"[yellow]![/yellow] Cannot merge {agent.name}: no branch")
            return

        # Show confirmation dialog
        self.push_screen(
            MergeConfirmScreen(agent.name, agent.task),
            self._on_merge_confirm,
        )

    def _on_merge_confirm(self, confirmed: bool) -> None:
        """Handle merge confirmation result.

        Args:
            confirmed: True if user confirmed the merge.
        """
        if not confirmed:
            return

        # Get the selected agent again (state may have changed)
        table = self.query_one("#agents", DataTable)
        cursor_row = table.cursor_row

        if cursor_row is None or cursor_row < 0:
            return

        state = load_state(self._project_root)
        agents_list = list(state.agents.values())

        if cursor_row >= len(agents_list):
            return

        agent = agents_list[cursor_row]
        event_log = self.query_one("#event-log", EventLog)

        # Perform the merge
        try:
            from crew.runner import cleanup_agent
            cleanup_agent(agent, state, merge=True, project_root=self._project_root)
            event_log.add_event(f"[green]✓[/green] Merged {agent.name} ({agent.branch})")
            # Refresh views
            self._refresh_agents(table)
            tree = self.query_one("#ticket-tree", TicketTree)
            tree.refresh_tickets(zoomed_ticket=self._zoomed_ticket)
        except Exception as e:
            event_log.add_event(f"[red]✗[/red] Merge failed for {agent.name}: {e}")

    def action_kill_agent(self) -> None:
        """Kill the selected agent with confirmation."""
        if not self._show_agents:
            return  # Only works in agents view

        table = self.query_one("#agents", DataTable)
        cursor_row = table.cursor_row

        if cursor_row is None or cursor_row < 0:
            self.notify("No agent selected", severity="warning")
            return

        state = load_state(self._project_root)
        agents_list = list(state.agents.values())

        if cursor_row >= len(agents_list):
            return

        agent = agents_list[cursor_row]

        # Show confirmation dialog
        self.push_screen(
            ConfirmKillScreen(agent.name),
            self._on_kill_confirm,
        )

    def _on_kill_confirm(self, confirmed: bool) -> None:
        """Handle kill confirmation result.

        Args:
            confirmed: True if user confirmed the kill.
        """
        if not confirmed:
            return

        # Get the selected agent again (state may have changed)
        table = self.query_one("#agents", DataTable)
        cursor_row = table.cursor_row

        if cursor_row is None or cursor_row < 0:
            return

        state = load_state(self._project_root)
        agents_list = list(state.agents.values())

        if cursor_row >= len(agents_list):
            return

        agent = agents_list[cursor_row]
        event_log = self.query_one("#event-log", EventLog)

        # Shutdown the agent
        try:
            work_status = shutdown_agent(agent, state)
            save_state(state, self._project_root)

            # Notify user about the result
            if work_status == "done":
                event_log.add_event(f"[green]✓[/green] Killed {agent.name} (work completed)")
            elif work_status == "partial":
                event_log.add_event(f"[yellow]![/yellow] Killed {agent.name} (partial work saved)")
            else:
                event_log.add_event(f"[dim]○[/dim] Killed {agent.name} (no work done)")

            # Refresh the UI
            self._refresh_agents(table)
            tree = self.query_one("#ticket-tree", TicketTree)
            tree.refresh_tickets(zoomed_ticket=self._zoomed_ticket)
        except Exception as e:
            event_log.add_event(f"[red]✗[/red] Kill failed for {agent.name}: {e}")

    def action_quit_or_unzoom(self) -> None:
        """Unzoom if zoomed, otherwise quit."""
        if self._help_visible:
            # Close help if visible
            self.action_toggle_help()
        elif self._zoomed_ticket is not None:
            # Unzoom
            self._zoomed_ticket = None
            tree = self.query_one("#ticket-tree", TicketTree)
            tree.refresh_tickets(zoomed_ticket=None)
        else:
            # Quit
            self.exit()


def main(
    runner: BackgroundRunner | None = None,
    project_root: Path | None = None,
):
    """Run the TUI application.

    Args:
        runner: Optional BackgroundRunner instance for live event updates.
        project_root: Project root path for merge operations.
    """
    app = CrewApp(runner=runner, project_root=project_root)
    app.run()


if __name__ == "__main__":
    main()
