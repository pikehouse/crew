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


def create_worktree(name: str, agents_dir: Path | None = None, force_clean: bool = False) -> Path:
    """Create a git worktree for an agent.

    Args:
        name: Agent name (used for directory and branch)
        agents_dir: Directory to create worktrees in (default: ./agents)
        force_clean: If True, remove existing worktree before creating

    Returns:
        Path to the created worktree
    """
    agents_dir = agents_dir or Path("agents")
    worktree_path = agents_dir / name
    branch_name = f"agent/{name}"

    if worktree_path.exists():
        if force_clean:
            # Clean up stale worktree
            remove_worktree(worktree_path)
        else:
            raise GitError(f"Worktree already exists: {worktree_path}")

    # Create worktree with new branch
    # Try to create with new branch first, fall back to existing branch
    # Prune stale worktree entries first (e.g. from manually deleted dirs)
    run_git("worktree", "prune")

    try:
        run_git("worktree", "add", str(worktree_path), "-b", branch_name)
    except GitError:
        # Branch might already exist, try without -b
        run_git("worktree", "add", str(worktree_path), branch_name)

    return worktree_path


def remove_worktree(worktree_path: Path) -> None:
    """Remove a git worktree."""
    if not worktree_path.exists():
        # Directory gone but git might still have it registered - prune stale entries
        run_git("worktree", "prune")
        return

    run_git("worktree", "remove", str(worktree_path), "--force")


def get_current_branch(cwd: Path | None = None) -> str:
    """Get the current branch name."""
    return run_git("branch", "--show-current", cwd=cwd)


def merge_branch(branch: str, message: str | None = None, cwd: Path | None = None) -> None:
    """Merge a branch into the current branch."""
    args = ["merge", "--no-ff", branch]
    if message:
        args.extend(["-m", message])
    run_git(*args, cwd=cwd)


def delete_branch(branch: str, cwd: Path | None = None) -> None:
    """Delete a local branch."""
    run_git("branch", "-d", branch, cwd=cwd)


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


def is_merge_in_progress(cwd: Path | None = None) -> bool:
    """Check if a merge is currently in progress."""
    cwd = cwd or Path.cwd()
    git_dir = cwd / ".git"
    # Handle worktrees where .git is a file pointing to the actual git dir
    if git_dir.is_file():
        content = git_dir.read_text().strip()
        if content.startswith("gitdir: "):
            git_dir = Path(content[8:])
    return (git_dir / "MERGE_HEAD").exists()


def get_conflicted_files(cwd: Path | None = None) -> list[str]:
    """Get list of files with merge conflicts.

    Returns list of file paths that have unmerged changes (conflict markers).
    """
    try:
        status = run_git("status", "--porcelain", cwd=cwd)
        conflicted = []
        for line in status.split("\n"):
            if line.startswith("UU ") or line.startswith("AA "):
                # UU = both modified, AA = both added
                conflicted.append(line[3:])
        return conflicted
    except GitError:
        return []


def abort_merge(cwd: Path | None = None) -> None:
    """Abort the current merge operation."""
    run_git("merge", "--abort", cwd=cwd)
