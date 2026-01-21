"""Property-based tests for crew state management.

Tests that invariants hold after any sequence of operations, including
random sequences and chaos testing with random failures.

Uses simple randomized testing (hypothesis optional but recommended).
"""

from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Callable
from unittest.mock import patch

import pytest

from crew.agent import Agent
from crew.runner import (
    spawn_worker,
    assign_task,
    step_agent,
    complete_task,
)
from crew.state import State, save_state, load_state
from tests.fault_injection.invariants import (
    assert_invariants,
    check_all_invariants,
    InvariantViolation,
)
from tests.fixtures.mock_claude import MockClaude


# Try to import hypothesis for property-based testing
try:
    from hypothesis import given, settings, strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


def _commit_work_in_worktree(worktree: Path, filename: str = "work.py") -> None:
    """Helper to create a commit in the worktree."""
    file_path = worktree / filename
    file_path.write_text(f"# Work in {filename}\n")
    subprocess.run(["git", "add", filename], cwd=worktree, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", f"Add {filename}"],
        cwd=worktree,
        capture_output=True,
        check=True,
    )


class OperationRunner:
    """Runs operations on a crew state and tracks history."""

    def __init__(self, git_project: Path):
        self.project = git_project
        self.state = State()
        self.history: list[str] = []
        self.agent_counter = 0
        self.task_counter = 0
        self.mock_claude = MockClaude()
        self.mock_claude.configure("success", responses=["Working...", "More work...", "DONE"])

    def spawn(self) -> str | None:
        """Spawn a new agent."""
        self.agent_counter += 1
        name = f"agent-{self.agent_counter}"
        try:
            spawn_worker(name, self.state, self.project)
            self.history.append(f"spawn({name})")
            return name
        except Exception as e:
            self.history.append(f"spawn({name}) FAILED: {e}")
            return None

    def assign(self, agent_name: str) -> bool:
        """Assign a task to an agent."""
        agent = self.state.get_agent(agent_name)
        if not agent or agent.status != "idle":
            self.history.append(f"assign({agent_name}) SKIPPED: not idle")
            return False

        self.task_counter += 1
        task_id = f"t-{self.task_counter:04d}"

        # Create ticket file
        ticket_dir = self.project / ".tickets"
        ticket_dir.mkdir(exist_ok=True)
        (ticket_dir / f"{task_id}.md").write_text(
            f"---\nid: {task_id}\nstatus: open\n---\n# Task {task_id}\n"
        )

        try:
            with patch("crew.runner.get_task_description") as mock_desc:
                mock_desc.return_value = f"Work on {task_id}"
                assign_task(agent, task_id, self.state, self.project)
            self.history.append(f"assign({agent_name}, {task_id})")
            return True
        except Exception as e:
            self.history.append(f"assign({agent_name}) FAILED: {e}")
            return False

    def step(self, agent_name: str) -> bool:
        """Step an agent."""
        agent = self.state.get_agent(agent_name)
        if not agent or agent.status not in ("ready", "working"):
            self.history.append(f"step({agent_name}) SKIPPED: invalid status")
            return False

        try:
            with patch("crew.runner.run_claude", self.mock_claude):
                with patch("crew.runner._refresh_agent_summary"):
                    step_agent(agent, self.state, project_root=self.project)
            self.history.append(f"step({agent_name}) -> {agent.status}")
            return True
        except Exception as e:
            self.history.append(f"step({agent_name}) FAILED: {e}")
            return False

    def complete(self, agent_name: str) -> bool:
        """Complete an agent's task."""
        agent = self.state.get_agent(agent_name)
        if not agent or agent.status != "done":
            self.history.append(f"complete({agent_name}) SKIPPED: not done")
            return False

        # Make some work to merge
        if agent.worktree and agent.worktree.exists():
            try:
                _commit_work_in_worktree(agent.worktree)
            except Exception:
                pass

        try:
            with patch("crew.runner.close_ticket", return_value=True):
                success, _ = complete_task(agent, self.state, self.project)
            self.history.append(f"complete({agent_name}) -> {success}")
            return success
        except Exception as e:
            self.history.append(f"complete({agent_name}) FAILED: {e}")
            return False

    def check_invariants(self) -> list[InvariantViolation]:
        """Check all invariants."""
        return check_all_invariants(self.state, self.project)


class TestRandomOperationSequences:
    """Tests with random sequences of operations."""

    def test_random_operations_maintain_invariants(self, git_project: Path):
        """Random sequence of operations maintains invariants."""
        runner = OperationRunner(git_project)

        # Run 50 random operations
        for i in range(50):
            # Choose operation
            op = random.choice(["spawn", "assign", "step", "complete"])

            if op == "spawn":
                runner.spawn()
            elif op == "assign":
                # Pick random idle agent
                idle_agents = [
                    a.name for a in runner.state.agents.values()
                    if a.status == "idle"
                ]
                if idle_agents:
                    runner.assign(random.choice(idle_agents))
            elif op == "step":
                # Pick random steppable agent
                steppable = [
                    a.name for a in runner.state.agents.values()
                    if a.status in ("ready", "working")
                ]
                if steppable:
                    runner.step(random.choice(steppable))
            elif op == "complete":
                # Pick random done agent
                done_agents = [
                    a.name for a in runner.state.agents.values()
                    if a.status == "done"
                ]
                if done_agents:
                    runner.complete(random.choice(done_agents))

            # Check invariants after each operation
            violations = runner.check_invariants()
            if violations:
                pytest.fail(
                    f"Invariant violations after operation {i}:\n"
                    f"History: {runner.history}\n"
                    f"Violations: {violations}"
                )

    def test_full_lifecycle_multiple_agents(self, git_project: Path):
        """Multiple agents completing full lifecycle maintains invariants."""
        runner = OperationRunner(git_project)

        # Spawn 5 agents
        agents = []
        for _ in range(5):
            name = runner.spawn()
            if name:
                agents.append(name)

        runner.check_invariants()

        # Assign tasks to all
        for name in agents:
            runner.assign(name)

        violations = runner.check_invariants()
        assert not violations, f"After assign: {violations}"

        # Step all to done
        for _ in range(10):  # Max 10 steps each
            for name in agents:
                agent = runner.state.get_agent(name)
                if agent and agent.status in ("ready", "working"):
                    runner.step(name)

        violations = runner.check_invariants()
        assert not violations, f"After steps: {violations}"

        # Complete all done agents
        for name in agents:
            agent = runner.state.get_agent(name)
            if agent and agent.status == "done":
                runner.complete(name)

        violations = runner.check_invariants()
        assert not violations, f"After complete: {violations}"


class TestChaosFailures:
    """Tests with random failures injected."""

    def test_random_failures_during_operations(self, git_project: Path):
        """Random failures during operations don't corrupt state."""
        runner = OperationRunner(git_project)

        # Spawn some agents
        for _ in range(3):
            runner.spawn()

        # Assign tasks
        for agent in list(runner.state.agents.values()):
            if agent.status == "idle":
                runner.assign(agent.name)

        # Run operations with random failures
        for i in range(30):
            # Pick operation
            op = random.choice(["step", "complete"])

            # Inject random failure
            should_fail = random.random() < 0.2  # 20% failure rate

            if op == "step":
                steppable = [
                    a.name for a in runner.state.agents.values()
                    if a.status in ("ready", "working")
                ]
                if steppable:
                    agent_name = random.choice(steppable)
                    agent = runner.state.get_agent(agent_name)

                    if should_fail:
                        # Simulate timeout
                        mock = MockClaude()
                        mock.configure("timeout")
                        try:
                            with patch("crew.runner.run_claude", mock):
                                step_agent(agent, runner.state, project_root=runner.project)
                        except TimeoutError:
                            runner.history.append(f"step({agent_name}) TIMEOUT")
                    else:
                        runner.step(agent_name)

            elif op == "complete":
                done_agents = [
                    a.name for a in runner.state.agents.values()
                    if a.status == "done"
                ]
                if done_agents:
                    agent_name = random.choice(done_agents)
                    if should_fail:
                        runner.history.append(f"complete({agent_name}) SKIPPED (inject)")
                    else:
                        runner.complete(agent_name)

            # Check invariants
            violations = runner.check_invariants()
            if violations:
                pytest.fail(
                    f"Invariant violations after operation {i}:\n"
                    f"History: {runner.history}\n"
                    f"Violations: {violations}"
                )


class TestIdempotencyProperties:
    """Tests verifying idempotency of operations."""

    def test_spawn_is_idempotent_with_different_names(self, git_project: Path):
        """Spawning different agents is idempotent (no side effects)."""
        state = State()

        agent1 = spawn_worker("agent-1", state, git_project)
        violations1 = check_all_invariants(state, git_project)

        agent2 = spawn_worker("agent-2", state, git_project)
        violations2 = check_all_invariants(state, git_project)

        # Both agents exist
        assert state.get_agent("agent-1") is not None
        assert state.get_agent("agent-2") is not None

        # No violations
        assert not violations1
        assert not violations2

    def test_assign_retry_is_safe(self, git_project: Path):
        """Retrying assign after partial failure is safe."""
        state = State()
        agent = spawn_worker("test-agent", state, git_project)

        # Create ticket
        ticket_dir = git_project / ".tickets"
        (ticket_dir / "t-retry.md").write_text(
            "---\nid: t-retry\nstatus: open\n---\n# Retry task\n"
        )

        # First attempt with mock
        with patch("crew.runner.get_task_description") as mock_desc:
            mock_desc.return_value = "Retry task"
            assign_task(agent, "t-retry", state, git_project)

        first_session = agent.session
        first_worktree = agent.worktree

        violations = check_all_invariants(state, git_project)
        assert not violations

        # Agent is now ready, can't reassign
        with pytest.raises(RuntimeError, match="not idle"):
            with patch("crew.runner.get_task_description") as mock_desc:
                mock_desc.return_value = "Retry task"
                assign_task(agent, "t-retry", state, git_project)

        # State unchanged
        assert agent.session == first_session
        assert agent.worktree == first_worktree


# Hypothesis tests (only if available)
if HYPOTHESIS_AVAILABLE:
    class TestHypothesisPropertyBased:
        """Property-based tests using Hypothesis."""

        @given(st.lists(
            st.sampled_from(["spawn", "assign", "step", "complete"]),
            min_size=1,
            max_size=50,
        ))
        @settings(max_examples=20, deadline=None)
        def test_any_operation_sequence(self, git_project: Path, operations: list[str]):
            """Any sequence of operations maintains invariants."""
            runner = OperationRunner(git_project)

            for op in operations:
                if op == "spawn":
                    runner.spawn()
                elif op == "assign":
                    idle_agents = [
                        a.name for a in runner.state.agents.values()
                        if a.status == "idle"
                    ]
                    if idle_agents:
                        runner.assign(random.choice(idle_agents))
                elif op == "step":
                    steppable = [
                        a.name for a in runner.state.agents.values()
                        if a.status in ("ready", "working")
                    ]
                    if steppable:
                        runner.step(random.choice(steppable))
                elif op == "complete":
                    done_agents = [
                        a.name for a in runner.state.agents.values()
                        if a.status == "done"
                    ]
                    if done_agents:
                        runner.complete(random.choice(done_agents))

                violations = runner.check_invariants()
                assert not violations, f"Violations after {op}: {violations}"
