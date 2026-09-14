"""Dependency-free windowed file reading.

Separated from ``tools.py`` (which needs langchain) so the benchmark can measure
real window-vs-full-file token costs without any framework installed.
"""

from __future__ import annotations

from pathlib import Path

from .config import BUDGET


def read_window(
    path: str | Path, line: int, radius: int = BUDGET.source_window_lines
) -> str:
    """Return +/- ``radius`` lines around ``line`` with 1-based line numbers."""
    file_path = Path(path)
    if not file_path.is_file():
        return f"[read_window] not found: {path}"

    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    start = max(0, line - 1 - radius)
    end = min(len(lines), line + radius)
    numbered = [f"{i + 1:>5} | {lines[i]}" for i in range(start, end)]
    header = f"{file_path.as_posix()} lines {start + 1}-{end} of {len(lines)}"
    return f"{header}\n" + "\n".join(numbered)


def outline(path: str | Path) -> str:
    """Top-level defs/classes with line numbers -- a cheap map of a file."""
    file_path = Path(path)
    if not file_path.is_file():
        return f"[outline] not found: {path}"
    out: list[str] = []
    for number, text in enumerate(
        file_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
    ):
        stripped = text.lstrip()
        if stripped.startswith(("def ", "class ", "async def ")):
            indent = len(text) - len(stripped)
            out.append(f"{number:>5} | {' ' * indent}{stripped.rstrip(':')}")
    return "\n".join(out) or "[outline] no top-level definitions"


def full_read(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    return file_path.read_text(encoding="utf-8", errors="replace")
