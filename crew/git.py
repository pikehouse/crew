"""Git operations for crew."""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitError(Exception):
    """Git operation failed."""

    pass


def run_git(*args: str, cwd: Path | None = None) -> str:
    """Run a git command and return output."""
    cmd = ["git", *args]
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise GitError(f"git {' '.join(args)} failed: {e.stderr.strip()}") from e


def create_worktree(name: str, agents_dir: Path | None = None) -> Path:
    """Create a git worktree for an agent.

    Args:
        name: Agent name (used for directory and branch)
        agents_dir: Directory to create worktrees in (default: ./agents)

    Returns:
        Path to the created worktree
    """
    agents_dir = agents_dir or Path("agents")
    worktree_path = agents_dir / name
    branch_name = f"agent/{name}"

    if worktree_path.exists():
        raise GitError(f"Worktree already exists: {worktree_path}")

    # Create worktree with new branch
    # Try to create with new branch first, fall back to existing branch
    try:
        run_git("worktree", "add", str(worktree_path), "-b", branch_name)
    except GitError:
        # Branch might already exist, try without -b
        run_git("worktree", "add", str(worktree_path), branch_name)

    return worktree_path


def remove_worktree(worktree_path: Path) -> None:
    """Remove a git worktree."""
    if not worktree_path.exists():
        return

    run_git("worktree", "remove", str(worktree_path), "--force")


def get_current_branch(cwd: Path | None = None) -> str:
    """Get the current branch name."""
    return run_git("branch", "--show-current", cwd=cwd)


def merge_branch(branch: str, message: str | None = None) -> None:
    """Merge a branch into the current branch."""
    args = ["merge", "--no-ff", branch]
    if message:
        args.extend(["-m", message])
    run_git(*args)


def delete_branch(branch: str) -> None:
    """Delete a local branch."""
    run_git("branch", "-d", branch)


def has_uncommitted_changes(cwd: Path | None = None) -> bool:
    """Check if there are uncommitted changes."""
    try:
        run_git("diff", "--quiet", cwd=cwd)
        run_git("diff", "--cached", "--quiet", cwd=cwd)
        return False
    except GitError:
        return True


def get_worktree_list() -> list[dict]:
    """Get list of all worktrees."""
    output = run_git("worktree", "list", "--porcelain")
    worktrees = []
    current: dict = {}

    for line in output.split("\n"):
        if line.startswith("worktree "):
            if current:
                worktrees.append(current)
            current = {"path": line[9:]}
        elif line.startswith("HEAD "):
            current["head"] = line[5:]
        elif line.startswith("branch "):
            current["branch"] = line[7:]
        elif line == "bare":
            current["bare"] = True
        elif line == "detached":
            current["detached"] = True

    if current:
        worktrees.append(current)

    return worktrees
