"""TUI for crew using Textual."""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import DataTable, Header, Footer

from crew.state import load_state


class CrewApp(App):
    """A Textual app for managing crew agents."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    CSS = """
    DataTable {
        height: 100%;
    }
    """

    def compose(self) -> ComposeResult:
        """Compose the app layout."""
        yield Header()
        yield DataTable(id="agents")
        yield Footer()

    def on_mount(self) -> None:
        """Set up the agents table when the app mounts."""
        table = self.query_one("#agents", DataTable)
        table.add_columns("Name", "Task", "Status", "Steps", "Cost")
        self._refresh_agents(table)

    def _refresh_agents(self, table: DataTable) -> None:
        """Refresh the agents table with current state."""
        table.clear()
        state = load_state()
        for agent in state.agents.values():
            cost_str = f"${agent.total_cost_usd:.2f}" if agent.total_cost_usd > 0 else "-"
            table.add_row(
                agent.name,
                agent.task or "-",
                agent.status,
                str(agent.step_count),
                cost_str,
            )


def main():
    """Run the TUI application."""
    app = CrewApp()
    app.run()


if __name__ == "__main__":
    main()
