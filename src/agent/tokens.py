"""Token accounting.

Uses tiktoken when available, otherwise falls back to a ~4-chars-per-token
heuristic so the benchmark and tests run in an offline CI container.
"""

from __future__ import annotations

from typing import Iterable

try:  # pragma: no cover - depends on environment
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover
    _ENCODING = None


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    return max(1, len(text) // 4)


def count_message_tokens(messages: Iterable) -> int:
    """Approximate prompt tokens for a list of LangChain messages.

    The +4 per message covers role/delimiter overhead in the chat format.
    """
    total = 0
    for message in messages:
        content = getattr(message, "content", message)
        if isinstance(content, list):  # multimodal content blocks
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        total += count_tokens(str(content)) + 4
    return total


def savings(before: int, after: int) -> str:
    if before <= 0:
        return "n/a"
    return f"{(1 - after / before) * 100:.1f}%"
