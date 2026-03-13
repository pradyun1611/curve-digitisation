"""Path helpers for consistent project-root-relative file resolution."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def repo_path(*parts: str) -> Path:
    """Build an absolute path anchored at the repository root."""
    return PROJECT_ROOT.joinpath(*parts).resolve()


def resolve_repo_path(path_like: str | Path, *, base: Path | None = None) -> Path:
    """Resolve absolute paths directly and relative paths from repo root (or base)."""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path.resolve()
    anchor = base if base is not None else PROJECT_ROOT
    return (anchor / path).resolve()
