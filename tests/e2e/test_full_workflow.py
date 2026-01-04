"""End-to-end tests for full crew workflow.

Tests spawn workers, assign tasks, run to completion using fake_claude.
Tests merge flow and stuck detection using sample_project fixture.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crew.agent import Agent
from crew.runner import (
    assign_task,
    complete_task,
    is_done,
    run_claude,
    spawn_worker,
    step_agent,
)
from crew.state import State, ensure_crew_dir, save_state, load_state
from crew.git import run_git, create_worktree, remove_worktree, merge_branch, delete_branch


# Path to fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
SAMPLE_PROJECT_DIR = FIXTURES_DIR / "sample_project"
FAKE_CLAUDE_PATH = FIXTURES_DIR / "fake_claude.py"
RESPONSES_DIR = FIXTURES_DIR / "responses"


@pytest.fixture
def sample_project(tmp_path: Path) -> Path:
    """Create a sample project with git repo and tickets for testing.

    Copies the sample_project fixture to a temp directory and initializes git.
    """
    # Copy sample_project to temp directory
    project_dir = tmp_path / "project"
    shutil.copytree(SAMPLE_PROJECT_DIR, project_dir)

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=project_dir, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=project_dir, capture_output=True, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=project_dir, capture_output=True, check=True)

    # Add all files and create initial commit
    subprocess.run(["git", "add", "-A"], cwd=project_dir, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=project_dir, capture_output=True, check=True)

    # Ensure crew directory exists
    ensure_crew_dir(project_dir)

    return project_dir


@pytest.fixture
def clean_fake_claude_sessions():
    """Clean up fake_claude session state before and after tests."""
    session_file = Path("/tmp/fake_claude_sessions.json")
    if session_file.exists():
        session_file.unlink()
    yield
    if session_file.exists():
        session_file.unlink()


def mock_run_claude_response(step: int, done: bool = False) -> dict:
    """Create a mock response for run_claude."""
    result_text = "DONE\n\nTask completed." if done else f"Working on step {step}..."
    return {
        "result": result_text,
        "input_tokens": 100 + step * 50,
        "output_tokens": 50 + step * 25,
        "cost_usd": 0.005 + step * 0.002,
        "stderr": "",
    }


class TestSpawnWorkersE2E:
    """E2E tests for spawn_worker functionality."""

    def test_spawn_single_worker(self, sample_project: Path):
        """Test spawning a single worker creates an idle agent."""
        state = State()

        agent = spawn_worker("worker-1", state, project_root=sample_project)

        assert agent.name == "worker-1"
        assert agent.status == "idle"
        assert agent.worktree is None
        assert agent.task is None
        assert agent.session == ""
        assert state.get_agent("worker-1") is agent

    def test_spawn_multiple_workers(self, sample_project: Path):
        """Test spawning multiple workers."""
        state = State()

        agents = []
        for i in range(3):
            agent = spawn_worker(f"worker-{i}", state, project_root=sample_project)
            agents.append(agent)

        assert len(state.agents) == 3
        for i, agent in enumerate(agents):
            assert agent.name == f"worker-{i}"
            assert agent.status == "idle"

    def test_spawn_worker_persists_state(self, sample_project: Path):
        """Test that spawned workers are persisted to state file."""
        state = State()
        spawn_worker("persistent-worker", state, project_root=sample_project)

        # Load state from disk
        loaded_state = load_state(sample_project)

        assert loaded_state.get_agent("persistent-worker") is not None
        assert loaded_state.get_agent("persistent-worker").status == "idle"


class TestAssignTasksE2E:
    """E2E tests for task assignment workflow."""

    def test_assign_task_to_idle_worker(self, sample_project: Path):
        """Test assigning a task to an idle worker."""
        state = State()
        agent = spawn_worker("task-worker", state, project_root=sample_project)

        # Mock create_worktree and get_task_description
        with patch("crew.runner.create_worktree") as mock_wt, \
             patch("crew.runner.get_task_description") as mock_desc:

            worktree_path = sample_project / "agents" / "task-worker-t-0002"
            worktree_path.mkdir(parents=True, exist_ok=True)
            mock_wt.return_value = worktree_path
            mock_desc.return_value = "# Add unit tests\n\nCreate unit tests for the sample module."

            assign_task(agent, "t-0002", state, project_root=sample_project)

        assert agent.status == "ready"
        assert agent.task == "t-0002"
        assert agent.branch == "agent/task-worker-t-0002"
        assert agent.session != ""  # Should have a session ID

    def test_assign_task_creates_claude_md(self, sample_project: Path):
        """Test that assigning a task creates CLAUDE.md in worktree."""
        state = State()
        agent = spawn_worker("claude-md-worker", state, project_root=sample_project)

        with patch("crew.runner.create_worktree") as mock_wt, \
             patch("crew.runner.get_task_description") as mock_desc:

            worktree_path = sample_project / "agents" / "claude-md-worker-t-0003"
            worktree_path.mkdir(parents=True, exist_ok=True)
            mock_wt.return_value = worktree_path
            mock_desc.return_value = "# Add documentation\n\nWrite API docs."

            assign_task(agent, "t-0003", state, project_root=sample_project)

        claude_md = worktree_path / "CLAUDE.md"
        assert claude_md.exists()
        content = claude_md.read_text()
        assert "claude-md-worker" in content
        assert "t-0003" in content

    def test_assign_task_fails_for_non_idle_worker(self, sample_project: Path):
        """Test that assigning a task to a non-idle worker fails."""
        state = State()
        agent = spawn_worker("busy-worker", state, project_root=sample_project)
        agent.status = "working"

        with pytest.raises(RuntimeError, match="not idle"):
            assign_task(agent, "t-0002", state, project_root=sample_project)

    def test_assign_multiple_tasks_to_different_workers(self, sample_project: Path):
        """Test assigning different tasks to multiple workers."""
        state = State()
        worker1 = spawn_worker("multi-1", state, project_root=sample_project)
        worker2 = spawn_worker("multi-2", state, project_root=sample_project)

        with patch("crew.runner.create_worktree") as mock_wt, \
             patch("crew.runner.get_task_description") as mock_desc:

            # First worker
            wt1 = sample_project / "agents" / "multi-1-t-0002"
            wt1.mkdir(parents=True, exist_ok=True)
            mock_wt.return_value = wt1
            mock_desc.return_value = "Task 1"
            assign_task(worker1, "t-0002", state, project_root=sample_project)

            # Second worker
            wt2 = sample_project / "agents" / "multi-2-t-0003"
            wt2.mkdir(parents=True, exist_ok=True)
            mock_wt.return_value = wt2
            mock_desc.return_value = "Task 2"
            assign_task(worker2, "t-0003", state, project_root=sample_project)

        assert worker1.task == "t-0002"
        assert worker2.task == "t-0003"
        assert worker1.status == "ready"
        assert worker2.status == "ready"


class TestRunToCompletionE2E:
    """E2E tests for running agents to completion using fake_claude."""

    def test_step_agent_to_completion(self, sample_project: Path):
        """Test stepping an agent until it outputs DONE."""
        state = State()
        agent = spawn_worker("step-worker", state, project_root=sample_project)

        # Set up agent as if task was assigned
        agent.status = "ready"
        agent.task = "t-0002"
        agent.session = "test-session-step"
        agent.worktree = sample_project
        agent.branch = "agent/step-worker-t-0002"

        # Mock responses: 3 steps before DONE
        responses = [
            mock_run_claude_response(0),
            mock_run_claude_response(1),
            mock_run_claude_response(2, done=True),
        ]

        with patch("crew.runner.run_claude") as mock_claude, \
             patch("crew.runner.write_log"):

            mock_claude.side_effect = responses

            # Step until done
            for _ in range(3):
                step_agent(agent, state, project_root=sample_project)
                if agent.status == "done":
                    break

        assert agent.status == "done"
        assert agent.step_count == 3

    def test_step_agent_accumulates_tokens(self, sample_project: Path):
        """Test that stepping accumulates token counts correctly."""
        state = State()
        agent = spawn_worker("token-worker", state, project_root=sample_project)

        agent.status = "ready"
        agent.task = "t-0002"
        agent.session = "test-session-tokens"
        agent.worktree = sample_project
        agent.branch = "agent/token-worker-t-0002"

        responses = [
            {"result": "Step 1", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.005, "stderr": ""},
            {"result": "Step 2", "input_tokens": 150, "output_tokens": 75, "cost_usd": 0.007, "stderr": ""},
            {"result": "DONE", "input_tokens": 80, "output_tokens": 40, "cost_usd": 0.003, "stderr": ""},
        ]

        with patch("crew.runner.run_claude") as mock_claude, \
             patch("crew.runner.write_log"):

            mock_claude.side_effect = responses

            for _ in range(3):
                step_agent(agent, state, project_root=sample_project)

        assert agent.total_input_tokens == 330  # 100 + 150 + 80
        assert agent.total_output_tokens == 165  # 50 + 75 + 40
        assert agent.total_cost_usd == pytest.approx(0.015)  # 0.005 + 0.007 + 0.003

    def test_multiple_agents_run_to_completion(self, sample_project: Path):
        """Test running multiple agents to completion."""
        state = State()

        agents = []
        for i in range(2):
            agent = spawn_worker(f"parallel-{i}", state, project_root=sample_project)
            agent.status = "ready"
            agent.task = f"t-000{i+2}"
            agent.session = f"session-parallel-{i}"
            agent.worktree = sample_project
            agent.branch = f"agent/parallel-{i}-t-000{i+2}"
            agents.append(agent)

        # Each agent takes 2 steps to complete
        with patch("crew.runner.run_claude") as mock_claude, \
             patch("crew.runner.write_log"):

            call_count = [0]
            def response_generator(*args, **kwargs):
                call_count[0] += 1
                # Every other call is DONE
                if call_count[0] % 2 == 0:
                    return mock_run_claude_response(1, done=True)
                return mock_run_claude_response(0)

            mock_claude.side_effect = response_generator

            # Step all agents
            for agent in agents:
                while agent.status != "done":
                    step_agent(agent, state, project_root=sample_project)

        for agent in agents:
            assert agent.status == "done"


class TestMergeFlowE2E:
    """E2E tests for the merge workflow after task completion."""

    def test_complete_task_resets_agent_to_idle(self, sample_project: Path):
        """Test that complete_task resets agent to idle state."""
        state = State()
        agent = spawn_worker("merge-worker", state, project_root=sample_project)

        # Set up as if agent completed work
        agent.status = "done"
        agent.task = "t-0002"
        agent.session = "merge-session"
        agent.worktree = sample_project / "agents" / "merge-worker-t-0002"
        agent.branch = "agent/merge-worker-t-0002"
        agent.step_count = 5
        agent.total_input_tokens = 500
        agent.total_output_tokens = 250

        with patch("crew.runner.close_ticket") as mock_close, \
             patch("crew.runner.remove_worktree") as mock_remove, \
             patch("crew.runner.run_git") as mock_git, \
             patch("crew.runner.merge_branch") as mock_merge, \
             patch("crew.runner.delete_branch") as mock_delete:

            complete_task(agent, state, project_root=sample_project)

        assert agent.status == "idle"
        assert agent.task is None
        assert agent.worktree is None
        assert agent.branch == ""
        assert agent.session == ""
        assert agent.step_count == 0

    def test_complete_task_merges_branch(self, sample_project: Path):
        """Test that complete_task merges the agent's branch."""
        state = State()
        agent = spawn_worker("branch-merge-worker", state, project_root=sample_project)

        agent.status = "done"
        agent.task = "t-0002"
        agent.session = "branch-merge-session"
        agent.worktree = sample_project / "agents" / "branch-merge-worker-t-0002"
        agent.branch = "agent/branch-merge-worker-t-0002"

        with patch("crew.runner.close_ticket"), \
             patch("crew.runner.remove_worktree"), \
             patch("crew.runner.run_git"), \
             patch("crew.runner.merge_branch") as mock_merge, \
             patch("crew.runner.delete_branch"):

            complete_task(agent, state, project_root=sample_project)

            mock_merge.assert_called_once()
            call_args = mock_merge.call_args
            assert "agent/branch-merge-worker-t-0002" in call_args[0]

    def test_complete_task_closes_ticket(self, sample_project: Path):
        """Test that complete_task closes the ticket."""
        state = State()
        agent = spawn_worker("ticket-close-worker", state, project_root=sample_project)

        agent.status = "done"
        agent.task = "t-0002"
        agent.session = "ticket-close-session"
        agent.worktree = sample_project / "agents" / "ticket-close-worker-t-0002"
        agent.branch = "agent/ticket-close-worker-t-0002"

        with patch("crew.runner.close_ticket") as mock_close, \
             patch("crew.runner.remove_worktree"), \
             patch("crew.runner.run_git"), \
             patch("crew.runner.merge_branch"), \
             patch("crew.runner.delete_branch"):

            complete_task(agent, state, project_root=sample_project)

            mock_close.assert_called_once_with("t-0002")

    def test_complete_task_removes_worktree(self, sample_project: Path):
        """Test that complete_task removes the worktree."""
        state = State()
        agent = spawn_worker("worktree-remove-worker", state, project_root=sample_project)

        worktree = sample_project / "agents" / "worktree-remove-worker-t-0002"
        agent.status = "done"
        agent.task = "t-0002"
        agent.session = "worktree-remove-session"
        agent.worktree = worktree
        agent.branch = "agent/worktree-remove-worker-t-0002"

        with patch("crew.runner.close_ticket"), \
             patch("crew.runner.remove_worktree") as mock_remove, \
             patch("crew.runner.run_git"), \
             patch("crew.runner.merge_branch"), \
             patch("crew.runner.delete_branch"):

            complete_task(agent, state, project_root=sample_project)

            mock_remove.assert_called_once_with(worktree)

    def test_complete_task_deletes_branch(self, sample_project: Path):
        """Test that complete_task deletes the merged branch."""
        state = State()
        agent = spawn_worker("branch-delete-worker", state, project_root=sample_project)

        agent.status = "done"
        agent.task = "t-0002"
        agent.session = "branch-delete-session"
        agent.worktree = sample_project / "agents" / "branch-delete-worker-t-0002"
        agent.branch = "agent/branch-delete-worker-t-0002"

        with patch("crew.runner.close_ticket"), \
             patch("crew.runner.remove_worktree"), \
             patch("crew.runner.run_git"), \
             patch("crew.runner.merge_branch"), \
             patch("crew.runner.delete_branch") as mock_delete:

            complete_task(agent, state, project_root=sample_project)

            mock_delete.assert_called_once_with("agent/branch-delete-worker-t-0002", cwd=sample_project)


class TestStuckDetectionE2E:
    """E2E tests for stuck agent detection."""

    def test_agent_becomes_stuck_after_20_steps(self, sample_project: Path):
        """Test that an agent is marked stuck after 20 steps without completion."""
        state = State()
        agent = spawn_worker("stuck-worker", state, project_root=sample_project)

        agent.status = "ready"
        agent.task = "t-0002"
        agent.session = "stuck-session"
        agent.worktree = sample_project
        agent.branch = "agent/stuck-worker-t-0002"
        agent.step_count = 19  # One step away from stuck

        with patch("crew.runner.run_claude") as mock_claude, \
             patch("crew.runner.write_log"):

            # Response that doesn't contain DONE
            mock_claude.return_value = {
                "result": "Still working...",
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.005,
                "stderr": "",
            }

            step_agent(agent, state, project_root=sample_project)

        assert agent.step_count == 20
        assert agent.status == "stuck"

    def test_agent_not_stuck_if_done_on_step_20(self, sample_project: Path):
        """Test that an agent completing on step 20 is not marked stuck."""
        state = State()
        agent = spawn_worker("almost-stuck-worker", state, project_root=sample_project)

        agent.status = "ready"
        agent.task = "t-0002"
        agent.session = "almost-stuck-session"
        agent.worktree = sample_project
        agent.branch = "agent/almost-stuck-worker-t-0002"
        agent.step_count = 19

        with patch("crew.runner.run_claude") as mock_claude, \
             patch("crew.runner.write_log"):

            # Response with DONE
            mock_claude.return_value = {
                "result": "Finally!\nDONE",
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.005,
                "stderr": "",
            }

            step_agent(agent, state, project_root=sample_project)

        assert agent.step_count == 20
        assert agent.status == "done"  # Done, not stuck

    def test_agent_not_stuck_before_20_steps(self, sample_project: Path):
        """Test that an agent is not marked stuck before 20 steps."""
        state = State()
        agent = spawn_worker("not-stuck-worker", state, project_root=sample_project)

        agent.status = "ready"
        agent.task = "t-0002"
        agent.session = "not-stuck-session"
        agent.worktree = sample_project
        agent.branch = "agent/not-stuck-worker-t-0002"
        agent.step_count = 18

        with patch("crew.runner.run_claude") as mock_claude, \
             patch("crew.runner.write_log"):

            mock_claude.return_value = {
                "result": "Still working...",
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.005,
                "stderr": "",
            }

            step_agent(agent, state, project_root=sample_project)

        assert agent.step_count == 19
        assert agent.status == "working"  # Not stuck yet


class TestFullWorkflowE2E:
    """E2E tests for the complete spawn -> assign -> run -> merge workflow."""

    def test_full_single_agent_workflow(self, sample_project: Path):
        """Test complete workflow: spawn, assign, step to completion, merge."""
        state = State()

        # 1. Spawn worker
        agent = spawn_worker("full-workflow-agent", state, project_root=sample_project)
        assert agent.status == "idle"

        # 2. Assign task
        with patch("crew.runner.create_worktree") as mock_wt, \
             patch("crew.runner.get_task_description") as mock_desc:

            worktree_path = sample_project / "agents" / "full-workflow-agent-t-0002"
            worktree_path.mkdir(parents=True, exist_ok=True)
            mock_wt.return_value = worktree_path
            mock_desc.return_value = "Add unit tests"

            assign_task(agent, "t-0002", state, project_root=sample_project)

        assert agent.status == "ready"
        assert agent.task == "t-0002"

        # 3. Step to completion
        responses = [
            {"result": "Starting...", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.005, "stderr": ""},
            {"result": "Working...", "input_tokens": 150, "output_tokens": 75, "cost_usd": 0.007, "stderr": ""},
            {"result": "DONE\n\nCompleted!", "input_tokens": 80, "output_tokens": 40, "cost_usd": 0.003, "stderr": ""},
        ]

        with patch("crew.runner.run_claude") as mock_claude, \
             patch("crew.runner.write_log"):

            mock_claude.side_effect = responses

            while agent.status not in ("done", "stuck"):
                step_agent(agent, state, project_root=sample_project)

        assert agent.status == "done"
        assert agent.step_count == 3

        # 4. Complete task (merge and cleanup)
        with patch("crew.runner.close_ticket"), \
             patch("crew.runner.remove_worktree"), \
             patch("crew.runner.run_git"), \
             patch("crew.runner.merge_branch"), \
             patch("crew.runner.delete_branch"):

            complete_task(agent, state, project_root=sample_project)

        assert agent.status == "idle"
        assert agent.task is None

    def test_full_multi_agent_workflow(self, sample_project: Path):
        """Test complete workflow with multiple agents working in parallel."""
        state = State()

        # Spawn and assign tasks to multiple agents
        agents = []
        for i in range(2):
            agent = spawn_worker(f"multi-agent-{i}", state, project_root=sample_project)

            with patch("crew.runner.create_worktree") as mock_wt, \
                 patch("crew.runner.get_task_description") as mock_desc:

                worktree = sample_project / "agents" / f"multi-agent-{i}-t-000{i+2}"
                worktree.mkdir(parents=True, exist_ok=True)
                mock_wt.return_value = worktree
                mock_desc.return_value = f"Task {i}"

                assign_task(agent, f"t-000{i+2}", state, project_root=sample_project)

            agents.append(agent)

        # All agents should be ready
        for agent in agents:
            assert agent.status == "ready"

        # Step all agents to completion
        with patch("crew.runner.run_claude") as mock_claude, \
             patch("crew.runner.write_log"):

            step_counter = [0]
            def mock_response(*args, **kwargs):
                step_counter[0] += 1
                if step_counter[0] % 2 == 0:
                    return {"result": "DONE", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.005, "stderr": ""}
                return {"result": "Working...", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.005, "stderr": ""}

            mock_claude.side_effect = mock_response

            # Step until all done
            max_iterations = 20
            iteration = 0
            while any(a.status not in ("done", "stuck") for a in agents) and iteration < max_iterations:
                for agent in agents:
                    if agent.status not in ("done", "stuck"):
                        step_agent(agent, state, project_root=sample_project)
                iteration += 1

        # All agents should be done
        for agent in agents:
            assert agent.status == "done"

        # Complete all tasks
        for agent in agents:
            with patch("crew.runner.close_ticket"), \
                 patch("crew.runner.remove_worktree"), \
                 patch("crew.runner.run_git"), \
                 patch("crew.runner.merge_branch"), \
                 patch("crew.runner.delete_branch"):

                complete_task(agent, state, project_root=sample_project)

        # All agents should be idle
        for agent in agents:
            assert agent.status == "idle"

    def test_workflow_with_stuck_recovery(self, sample_project: Path):
        """Test workflow where one agent gets stuck while another completes."""
        state = State()

        # Spawn two agents
        good_agent = spawn_worker("good-agent", state, project_root=sample_project)
        stuck_agent = spawn_worker("stuck-agent", state, project_root=sample_project)

        # Set up both agents
        for agent, task_id in [(good_agent, "t-0002"), (stuck_agent, "t-0003")]:
            with patch("crew.runner.create_worktree") as mock_wt, \
                 patch("crew.runner.get_task_description") as mock_desc:

                worktree = sample_project / "agents" / f"{agent.name}-{task_id}"
                worktree.mkdir(parents=True, exist_ok=True)
                mock_wt.return_value = worktree
                mock_desc.return_value = f"Task for {agent.name}"

                assign_task(agent, task_id, state, project_root=sample_project)

        # Step agents with different outcomes
        with patch("crew.runner.run_claude") as mock_claude, \
             patch("crew.runner.write_log"):

            def mock_response(prompt, cwd, *args, **kwargs):
                # Good agent finishes quickly
                if "good-agent" in str(cwd):
                    return {"result": "DONE", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.005, "stderr": ""}
                # Stuck agent never says DONE
                return {"result": "Still trying...", "input_tokens": 100, "output_tokens": 50, "cost_usd": 0.005, "stderr": ""}

            mock_claude.side_effect = mock_response

            # Step good agent
            step_agent(good_agent, state, project_root=sample_project)
            assert good_agent.status == "done"

            # Step stuck agent 20 times
            stuck_agent.step_count = 19
            step_agent(stuck_agent, state, project_root=sample_project)
            assert stuck_agent.status == "stuck"

        # Complete good agent's task
        with patch("crew.runner.close_ticket"), \
             patch("crew.runner.remove_worktree"), \
             patch("crew.runner.run_git"), \
             patch("crew.runner.merge_branch"), \
             patch("crew.runner.delete_branch"):

            complete_task(good_agent, state, project_root=sample_project)

        assert good_agent.status == "idle"
        assert stuck_agent.status == "stuck"


class TestIsDoneFunction:
    """Tests for the is_done helper function."""

    def test_done_on_own_line(self):
        """DONE on its own line returns True."""
        assert is_done("Some output\nDONE\nMore output") is True

    def test_done_case_insensitive(self):
        """is_done is case insensitive."""
        assert is_done("done") is True
        assert is_done("Done") is True
        assert is_done("DONE") is True

    def test_done_with_whitespace(self):
        """is_done handles whitespace."""
        assert is_done("  DONE  ") is True
        assert is_done("\nDONE\n") is True

    def test_done_embedded_returns_false(self):
        """DONE embedded in text returns False."""
        assert is_done("Not DONE yet") is False
        assert is_done("UNDONE") is False

    def test_no_done_returns_false(self):
        """No DONE returns False."""
        assert is_done("Still working") is False
        assert is_done("") is False


class TestTicketCloseCommittedWithMerge:
    """E2E tests verifying ticket close is committed as part of merge flow.

    When complete_task() runs, tk close modifies .tickets/<id>.md.
    These tests ensure this change is properly committed as part of the
    merge flow, not left as uncommitted changes in main.
    """

    def test_ticket_close_is_committed_after_merge(self, sample_project: Path):
        """Test that ticket close change is committed after agent merge.

        After complete_task() merges an agent's work:
        1. The ticket file should show status=closed
        2. The change should be committed (not dirty/uncommitted)
        """
        state = State()

        # Create agent and set up as if it completed work
        agent = spawn_worker("close-commit-worker", state, project_root=sample_project)

        # Create a real branch for the agent with actual commits
        branch_name = "agent/close-commit-worker-t-0002"
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=sample_project,
            capture_output=True,
            check=True,
        )

        # Make a commit on the agent branch (simulating agent work)
        test_file = sample_project / "agent_work.txt"
        test_file.write_text("Work done by agent")
        subprocess.run(
            ["git", "add", "agent_work.txt"],
            cwd=sample_project,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Agent work commit"],
            cwd=sample_project,
            capture_output=True,
            check=True,
        )

        # Switch back to main so we can set up the worktree mock
        subprocess.run(
            ["git", "checkout", "main"],
            cwd=sample_project,
            capture_output=True,
            check=True,
        )

        # Set up agent state as "done"
        agent.status = "done"
        agent.task = "t-0002"
        agent.session = "close-commit-session"
        agent.worktree = sample_project / "agents" / "close-commit-worker-t-0002"
        agent.branch = branch_name

        # Create a mock close_ticket that simulates tk close behavior
        # by directly modifying the ticket file to set status=closed
        def mock_close_ticket(task_id: str) -> bool:
            ticket_path = sample_project / ".tickets" / f"{task_id}.md"
            if ticket_path.exists():
                content = ticket_path.read_text()
                # Replace status: open with status: closed
                content = content.replace("status: open", "status: closed")
                ticket_path.write_text(content)
                return True
            return False

        # Run complete_task with mocked components
        with patch("crew.runner.close_ticket", side_effect=mock_close_ticket), \
             patch("crew.runner.remove_worktree"), \
             patch("crew.runner.run_tests_in_worktree", return_value=(True, None)):

            complete_task(agent, state, project_root=sample_project)

        # Verify 1: Ticket file shows status=closed
        ticket_path = sample_project / ".tickets" / "t-0002.md"
        ticket_content = ticket_path.read_text()
        assert "status: closed" in ticket_content, \
            f"Ticket should show status: closed, got:\n{ticket_content}"

        # Verify 2: Git status should be clean for tracked files (no uncommitted changes)
        # Note: .crew/ is untracked and expected (it's state data)
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=sample_project,
            capture_output=True,
            text=True,
        )

        # Filter out untracked files (lines starting with ??)
        uncommitted_changes = [
            line for line in result.stdout.strip().split("\n")
            if line and not line.startswith("??")
        ]
        assert uncommitted_changes == [], \
            f"Ticket close should be committed, but found uncommitted changes:\n{uncommitted_changes}"

        # Verify 3: The ticket change should be in the commit history
        result = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            cwd=sample_project,
            capture_output=True,
            text=True,
        )
        # There should be a merge commit that includes the ticket close
        assert "agent/close-commit-worker-t-0002" in result.stdout or \
               "close-commit-worker" in result.stdout.lower(), \
            f"Merge commit should reference the agent branch, got:\n{result.stdout}"

    def test_ticket_file_shows_closed_status_in_git_log(self, sample_project: Path):
        """Test that the ticket file with status=closed appears in git log.

        This verifies the ticket modification is part of the git history,
        not just staged or in working directory.
        """
        state = State()

        agent = spawn_worker("log-verify-worker", state, project_root=sample_project)

        # Create agent branch with commits
        branch_name = "agent/log-verify-worker-t-0003"
        subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=sample_project,
            capture_output=True,
            check=True,
        )

        # Make a commit on agent branch
        work_file = sample_project / "log_verify_work.txt"
        work_file.write_text("Log verify work")
        subprocess.run(["git", "add", "."], cwd=sample_project, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Log verify agent work"],
            cwd=sample_project,
            capture_output=True,
            check=True,
        )

        # Return to main
        subprocess.run(["git", "checkout", "main"], cwd=sample_project, capture_output=True, check=True)

        # Set up agent state
        agent.status = "done"
        agent.task = "t-0003"
        agent.session = "log-verify-session"
        agent.worktree = sample_project / "agents" / "log-verify-worker-t-0003"
        agent.branch = branch_name

        def mock_close_ticket(task_id: str) -> bool:
            ticket_path = sample_project / ".tickets" / f"{task_id}.md"
            if ticket_path.exists():
                content = ticket_path.read_text()
                content = content.replace("status: open", "status: closed")
                ticket_path.write_text(content)
                return True
            return False

        with patch("crew.runner.close_ticket", side_effect=mock_close_ticket), \
             patch("crew.runner.remove_worktree"), \
             patch("crew.runner.run_tests_in_worktree", return_value=(True, None)):

            complete_task(agent, state, project_root=sample_project)

        # Verify that the ticket file in HEAD shows status: closed
        # This checks the committed content, not the working directory
        result = subprocess.run(
            ["git", "show", "HEAD:.tickets/t-0003.md"],
            cwd=sample_project,
            capture_output=True,
            text=True,
        )

        # The committed version of the ticket should show status: closed
        assert "status: closed" in result.stdout, \
            f"Committed ticket should show status: closed, got:\n{result.stdout}"

    def test_sequential_agent_completions_preserve_ticket_closes(self, sample_project: Path):
        """Test that sequential agent completions properly preserve all ticket closes.

        When multiple agents complete sequentially, each ticket close should be
        committed and not lost when subsequent agents merge their work.
        """
        state = State()

        # First agent: complete and verify ticket is closed and committed
        agent1 = spawn_worker("seq-agent-1", state, project_root=sample_project)

        branch1 = "agent/seq-agent-1-t-0002"
        subprocess.run(
            ["git", "checkout", "-b", branch1],
            cwd=sample_project,
            capture_output=True,
            check=True,
        )

        work_file1 = sample_project / "seq_agent_1_work.txt"
        work_file1.write_text("Work by agent 1")
        subprocess.run(["git", "add", "."], cwd=sample_project, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Agent 1 work"],
            cwd=sample_project,
            capture_output=True,
            check=True,
        )

        subprocess.run(["git", "checkout", "main"], cwd=sample_project, capture_output=True, check=True)

        agent1.status = "done"
        agent1.task = "t-0002"
        agent1.session = "seq-agent-1-session"
        agent1.worktree = sample_project / "agents" / "seq-agent-1-t-0002"
        agent1.branch = branch1

        def mock_close_ticket_1(task_id: str) -> bool:
            ticket_path = sample_project / ".tickets" / f"{task_id}.md"
            if ticket_path.exists():
                content = ticket_path.read_text()
                content = content.replace("status: open", "status: closed")
                ticket_path.write_text(content)
                return True
            return False

        with patch("crew.runner.close_ticket", side_effect=mock_close_ticket_1), \
             patch("crew.runner.remove_worktree"), \
             patch("crew.runner.run_tests_in_worktree", return_value=(True, None)):

            complete_task(agent1, state, project_root=sample_project)

        # Verify first ticket is closed and committed
        result = subprocess.run(
            ["git", "show", "HEAD:.tickets/t-0002.md"],
            cwd=sample_project,
            capture_output=True,
            text=True,
        )
        assert "status: closed" in result.stdout, \
            f"First ticket should be committed as closed, got:\n{result.stdout}"

        # Second agent: complete and verify both tickets are closed and committed
        agent2 = spawn_worker("seq-agent-2", state, project_root=sample_project)

        branch2 = "agent/seq-agent-2-t-0003"
        subprocess.run(
            ["git", "checkout", "-b", branch2],
            cwd=sample_project,
            capture_output=True,
            check=True,
        )

        work_file2 = sample_project / "seq_agent_2_work.txt"
        work_file2.write_text("Work by agent 2")
        subprocess.run(["git", "add", "."], cwd=sample_project, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Agent 2 work"],
            cwd=sample_project,
            capture_output=True,
            check=True,
        )

        subprocess.run(["git", "checkout", "main"], cwd=sample_project, capture_output=True, check=True)

        agent2.status = "done"
        agent2.task = "t-0003"
        agent2.session = "seq-agent-2-session"
        agent2.worktree = sample_project / "agents" / "seq-agent-2-t-0003"
        agent2.branch = branch2

        def mock_close_ticket_2(task_id: str) -> bool:
            ticket_path = sample_project / ".tickets" / f"{task_id}.md"
            if ticket_path.exists():
                content = ticket_path.read_text()
                content = content.replace("status: open", "status: closed")
                ticket_path.write_text(content)
                return True
            return False

        with patch("crew.runner.close_ticket", side_effect=mock_close_ticket_2), \
             patch("crew.runner.remove_worktree"), \
             patch("crew.runner.run_tests_in_worktree", return_value=(True, None)):

            complete_task(agent2, state, project_root=sample_project)

        # Verify second ticket is closed and committed
        result = subprocess.run(
            ["git", "show", "HEAD:.tickets/t-0003.md"],
            cwd=sample_project,
            capture_output=True,
            text=True,
        )
        assert "status: closed" in result.stdout, \
            f"Second ticket should be committed as closed, got:\n{result.stdout}"

        # Verify first ticket is STILL closed (not reverted by second merge)
        result = subprocess.run(
            ["git", "show", "HEAD:.tickets/t-0002.md"],
            cwd=sample_project,
            capture_output=True,
            text=True,
        )
        assert "status: closed" in result.stdout, \
            f"First ticket should still be closed after second merge, got:\n{result.stdout}"

        # Git status should be clean
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=sample_project,
            capture_output=True,
            text=True,
        )
        uncommitted_changes = [
            line for line in result.stdout.strip().split("\n")
            if line and not line.startswith("??")
        ]
        assert uncommitted_changes == [], \
            f"All ticket closes should be committed, but found:\n{uncommitted_changes}"
