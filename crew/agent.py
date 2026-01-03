"""Agent data structures and operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

AgentStatus = Literal["idle", "ready", "working", "done", "stuck"]


@dataclass
class Agent:
    """Represents a Claude Code agent working in a git worktree."""

    name: str
    session: str
    worktree: Path | None
    branch: str
    task: str | None = None
    status: AgentStatus = "idle"
    started_at: datetime = field(default_factory=datetime.now)
    step_count: int = 0
    last_step_at: datetime | None = None

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage."""
        return {
            "name": self.name,
            "session": self.session,
            "worktree": str(self.worktree) if self.worktree else None,
            "branch": self.branch,
            "task": self.task,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "step_count": self.step_count,
            "last_step_at": self.last_step_at.isoformat() if self.last_step_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Agent":
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            session=data["session"],
            worktree=Path(data["worktree"]) if data.get("worktree") else None,
            branch=data["branch"],
            task=data.get("task"),
            status=data.get("status", "idle"),
            started_at=datetime.fromisoformat(data["started_at"]),
            step_count=data.get("step_count", 0),
            last_step_at=datetime.fromisoformat(data["last_step_at"]) if data.get("last_step_at") else None,
        )

    @property
    def is_active(self) -> bool:
        """True if agent can be stepped (ready, working, or idle)."""
        return self.status in ("ready", "working", "idle")

    @property
    def is_done(self) -> bool:
        """True if agent has completed its task."""
        return self.status == "done"

    @property
    def elapsed(self) -> str:
        """Human-readable elapsed time since agent started."""
        delta = datetime.now() - self.started_at
        minutes = int(delta.total_seconds() // 60)
        if minutes < 60:
            return f"{minutes}m"
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h{mins}m"
