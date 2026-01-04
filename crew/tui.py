"""Basic TUI skeleton for crew using Textual."""

from textual.app import App
from textual.binding import Binding


class CrewApp(App):
    """A Textual app for managing crew agents."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
    ]

    def compose(self):
        """Compose the app layout."""
        yield from ()


def main():
    """Run the TUI application."""
    app = CrewApp()
    app.run()


if __name__ == "__main__":
    main()
