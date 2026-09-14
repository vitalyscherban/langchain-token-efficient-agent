"""Stage 3: tools that return windows, not whole files.

The single biggest token sink in a coding agent is ``read_file`` on a 2,000-line
module to look at one function. Every tool here is designed to return the
smallest payload that still answers the question.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from .tokens import count_tokens
from .windows import full_read, outline, read_window

SKIP_DIRS = {".venv", "venv", "node_modules", "__pycache__", ".git"}


@tool
def read_source_window(path: str, line: int) -> str:
    """Read source around a specific line. Use the line from the stack trace.

    Prefer this over reading a whole file: it costs ~10-30x fewer tokens.
    """
    return read_window(path, line)


@tool
def grep_symbol(root: str, symbol: str, max_hits: int = 10) -> str:
    """Find where a symbol is defined or used. Returns file:line:text only.

    Returns locations, never file bodies -- follow up with read_source_window.
    """
    hits: list[str] = []
    for file_path in Path(root).rglob("*.py"):
        if SKIP_DIRS.intersection(file_path.parts):
            continue
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if symbol not in content:
            continue
        for number, text in enumerate(content.splitlines(), start=1):
            if symbol in text:
                hits.append(f"{file_path.as_posix()}:{number}: {text.strip()[:160]}")
                if len(hits) >= max_hits:
                    return "\n".join(hits)
    return "\n".join(hits) or f"[grep_symbol] no hits for {symbol!r}"


@tool
def file_outline(path: str) -> str:
    """List top-level defs/classes with line numbers -- a cheap map of a file.

    Costs a few percent of the tokens of reading the file and is usually enough
    to pick the right line for read_source_window.
    """
    return outline(path)


def token_cost_of_full_read(path: str | Path) -> int:
    """Used by the benchmark to show what a naive full read would have cost."""
    return count_tokens(full_read(path))


TOOLS = [read_source_window, grep_symbol, file_outline]
