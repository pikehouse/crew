"""State management for crew."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from crew.agent import Agent

CREW_DIR = ".crew"
STATE_FILE = "state.json"
STATE_VERSION = 1


@dataclass
class State:
    """Global crew state."""

    agents: dict[str, Agent] = field(default_factory=dict)
    version: int = STATE_VERSION

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "agents": {name: agent.to_dict() for name, agent in self.agents.items()},
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "State":
        """Deserialize from dictionary."""
        return cls(
            agents={name: Agent.from_dict(agent_data) for name, agent_data in data.get("agents", {}).items()},
            version=data.get("version", STATE_VERSION),
        )

    @property
    def active_agents(self) -> list[Agent]:
        """List of agents currently working."""
        return [a for a in self.agents.values() if a.is_active]

    def get_agent(self, name: str) -> Agent | None:
        """Get agent by name."""
        return self.agents.get(name)

    def add_agent(self, agent: Agent) -> None:
        """Add or update an agent."""
        self.agents[agent.name] = agent

    def remove_agent(self, name: str) -> None:
        """Remove an agent by name."""
        self.agents.pop(name, None)


def get_crew_dir(project_root: Path | None = None) -> Path:
    """Get the .crew directory path."""
    root = project_root or Path.cwd()
    return root / CREW_DIR


def ensure_crew_dir(project_root: Path | None = None) -> Path:
    """Ensure .crew directory exists, return path."""
    crew_dir = get_crew_dir(project_root)
    crew_dir.mkdir(parents=True, exist_ok=True)
    (crew_dir / "logs").mkdir(exist_ok=True)
    return crew_dir


def load_state(project_root: Path | None = None) -> State:
    """Load state from .crew/state.json, or create empty state."""
    crew_dir = get_crew_dir(project_root)
    state_path = crew_dir / STATE_FILE

    if state_path.exists():
        with open(state_path) as f:
            data = json.load(f)
        return State.from_dict(data)

    return State()


def save_state(state: State, project_root: Path | None = None) -> None:
    """Save state to .crew/state.json."""
    crew_dir = ensure_crew_dir(project_root)
    state_path = crew_dir / STATE_FILE

    with open(state_path, "w") as f:
        json.dump(state.to_dict(), f, indent=2)
