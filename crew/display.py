"""Display formatting for crew."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

from crew.agent import Agent
from crew.state import State

console = Console()


def print_banner() -> None:
    """Print the crew banner."""
    banner = Text()
    banner.append("crew", style="bold blue")
    banner.append(" — multi-agent orchestrator", style="dim")
    console.print(Panel(banner, border_style="blue"))


def print_status(state: State) -> None:
    """Print the current status of all agents and work queue."""
    # Agents table
    if state.agents:
        table = Table(title="Agents", show_header=True, header_style="bold")
        table.add_column("", width=2)
        table.add_column("Name")
        table.add_column("Task")
        table.add_column("Status")
        table.add_column("Steps")
        table.add_column("Time")

        for agent in state.agents.values():
            status_icon = get_status_icon(agent.status)
            status_style = get_status_style(agent.status)

            table.add_row(
                status_icon,
                agent.name,
                agent.task or "-",
                Text(agent.status, style=status_style),
                str(agent.step_count),
                agent.elapsed,
            )

        console.print(table)
    else:
        console.print("[dim]No agents.[/dim]")

    console.print()


def get_status_icon(status: str) -> str:
    """Get icon for agent status."""
    icons = {
        "idle": "[dim]○[/dim]",
        "ready": "[yellow]◐[/yellow]",
        "working": "[green]●[/green]",
        "done": "[blue]✓[/blue]",
        "stuck": "[red]![/red]",
    }
    return icons.get(status, "?")


def get_status_style(status: str) -> str:
    """Get style for agent status."""
    styles = {
        "idle": "dim",
        "ready": "yellow",
        "working": "green",
        "done": "blue",
        "stuck": "red",
    }
    return styles.get(status, "")


def print_agent_created(agent: Agent) -> None:
    """Print agent creation message."""
    console.print(f"[green]✓[/green] Created agent [bold]{agent.name}[/bold]", end="")
    if agent.task:
        console.print(f" on [cyan]{agent.task}[/cyan]")
    else:
        console.print()


def print_agent_step(agent: Agent, output_preview: str = "") -> None:
    """Print agent step message."""
    console.print(f"[blue]→[/blue] Stepped [bold]{agent.name}[/bold] (step {agent.step_count})")
    if output_preview:
        console.print(f"  [dim]{output_preview[:80]}...[/dim]")


def print_agent_done(agent: Agent) -> None:
    """Print agent completion message."""
    console.print(f"[green]✓[/green] [bold]{agent.name}[/bold] completed [cyan]{agent.task}[/cyan]")


def print_agent_merged(agent: Agent) -> None:
    """Print agent merge message."""
    console.print(f"[green]✓[/green] Merged [bold]{agent.branch}[/bold] to main")


def print_error(message: str) -> None:
    """Print error message."""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    console.print(f"[yellow]![/yellow] {message}")


def print_info(message: str) -> None:
    """Print info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def print_peek(agent_name: str, content: str) -> None:
    """Print peek output."""
    console.print(Panel(
        content,
        title=f"[bold]{agent_name}[/bold] latest output",
        border_style="dim",
    ))


def print_help() -> None:
    """Print help message."""
    help_text = """
[bold]Commands:[/bold]

  [cyan]status[/cyan], [cyan]s[/cyan]          Show all agents and queue
  [cyan]spawn[/cyan] <name> [task]   Create worker (optionally with task)
  [cyan]step[/cyan] <name>       Run one step for agent
  [cyan]run[/cyan]              Start agents in background
  [cyan]stop[/cyan]             Stop background runner (Ctrl-C also works)
  [cyan]ps[/cyan]               Show running claude processes
  [cyan]watch[/cyan]            Show live file changes in worktrees
  [cyan]peek[/cyan] <name>       Show agent's recent output (after step completes)
  [cyan]sniff[/cyan]            Show tail of ALL agents' output (after steps complete)
  [cyan]logs[/cyan] <name>       Show log directory
  [cyan]kill[/cyan] <name>       Stop agent, keep worktree
  [cyan]cleanup[/cyan] <name>    Remove agent and worktree
  [cyan]merge[/cyan] <name>      Merge agent's branch to main

  [cyan]ready[/cyan], [cyan]r[/cyan]          Show ready tickets (tk ready)
  [cyan]new[/cyan] <title> [--dep <id>...]   Create ticket with optional dependencies
  [cyan]dep[/cyan] <ticket> <depends-on>    Add dependency (ticket blocked by depends-on)
  [cyan]assign[/cyan] <name> <id> Assign ticket to agent

  [cyan]reset[/cyan] <commit>    [red]Nuclear option[/red]: reset everything to commit

  [cyan]help[/cyan]             Show this help
  [cyan]quit[/cyan], [cyan]q[/cyan]          Exit
"""
    console.print(help_text)
