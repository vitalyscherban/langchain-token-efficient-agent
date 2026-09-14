"""Stage 2: keep the message history bounded.

Two complementary levers:

* ``trim_messages`` (langchain-core) -- token-aware, drops whole turns from the
  front while always preserving the system message and leaving the window
  starting on a HumanMessage so tool-call pairs stay valid.
* ``compact_tool_result`` -- replaces a large raw tool payload with a short
  structured digest *before* it ever enters history. Trimming fights symptoms;
  compaction fights the cause.
"""

from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)

from .config import BUDGET
from .prompts import SUMMARY_PROMPT
from .tokens import count_tokens


def trim_history(
    messages: list[BaseMessage], max_tokens: int = BUDGET.max_history_tokens
) -> list[BaseMessage]:
    """Token-aware trim that never orphans a tool call from its result."""
    return trim_messages(
        messages,
        max_tokens=max_tokens,
        strategy="last",
        token_counter=count_tokens_for_trim,
        include_system=True,
        allow_partial=False,
        start_on="human",
    )


def count_tokens_for_trim(messages: list[BaseMessage]) -> int:
    return sum(count_tokens(str(m.content)) + 4 for m in messages)


def compact_tool_result(
    name: str, payload: str, limit: int = BUDGET.tool_result_char_limit
) -> str:
    """Head/tail truncation with an explicit elision marker.

    Truncating in the middle beats truncating the tail: the head carries the
    command/context and the tail carries the error, while the middle is filler.
    """
    payload = payload.strip()
    if len(payload) <= limit:
        return payload
    head = payload[: limit // 2]
    tail = payload[-limit // 2 :]
    dropped = len(payload) - limit
    return f"{head}\n... [{name}: {dropped} chars elided] ...\n{tail}"


def summarize_old_turns(llm, messages: list[BaseMessage]) -> list[BaseMessage]:
    """Fold everything older than ``keep_recent_turns`` into one summary message.

    Runs on the cheap model: summarizing is extraction, not reasoning.
    """
    system = [m for m in messages if isinstance(m, SystemMessage)]
    body = [m for m in messages if not isinstance(m, SystemMessage)]

    if len(body) <= BUDGET.keep_recent_turns:
        return messages

    old, recent = body[: -BUDGET.keep_recent_turns], body[-BUDGET.keep_recent_turns :]
    transcript = "\n".join(f"{m.type}: {m.content}" for m in old)
    summary = llm.invoke([SystemMessage(content=SUMMARY_PROMPT), AIMessage(content=transcript)])
    state_block = SystemMessage(content=f"[COMPACTED STATE]\n{summary.content}")
    return [*system, state_block, *recent]


def strip_stale_tool_payloads(
    messages: list[BaseMessage], keep_last: int = 2
) -> list[BaseMessage]:
    """Once a tool result has been reasoned over, its raw text is dead weight.

    Keeps the newest ``keep_last`` tool payloads intact and reduces older ones to
    a one-line receipt so the model still knows the step happened.
    """
    tool_indexes = [i for i, m in enumerate(messages) if isinstance(m, ToolMessage)]
    stale = set(tool_indexes[:-keep_last]) if keep_last else set(tool_indexes)

    out: list[BaseMessage] = []
    for i, message in enumerate(messages):
        if i in stale and isinstance(message, ToolMessage):
            out.append(
                ToolMessage(
                    content=f"[{message.name}: result consumed, "
                    f"{count_tokens(str(message.content))} tokens reclaimed]",
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                )
            )
        else:
            out.append(message)
    return out
