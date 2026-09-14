"""The LangGraph agent, wiring every stage together.

Graph shape::

    triage ──> agent ──(tool calls)──> tools ──> compact ──┐
                 ^                                          │
                 └──────────────────────────────────────────┘
                 │
                 └──(no tool calls)──> END

Token discipline is enforced at two edges, not inside the model:
* ``compact`` shrinks tool output the moment it is produced.
* ``agent`` trims history right before every model call.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .cache import enable_response_cache, stable_system_prompt
from .context import compact_tool_result, strip_stale_tool_payloads, trim_history
from .log_pruner import filter_diff, prune_ci_log, render_failures
from .tokens import count_message_tokens
from .tools import TOOLS


class TriageState(TypedDict):
    # add_messages (not operator.add) is load-bearing: it replaces a message
    # when an update reuses its id, which is how the compact node rewrites a
    # bulky tool result in place instead of appending a second copy.
    messages: Annotated[list[BaseMessage], add_messages]
    raw_log: str
    raw_diff: str
    prompt_tokens: Annotated[int, operator.add]


def build_graph(llm):
    """Compile the agent. ``llm`` must be a chat model supporting tool calling."""
    enable_response_cache()
    model = llm.bind_tools(TOOLS)
    tool_node = ToolNode(TOOLS)

    def triage(state: TriageState) -> dict:
        """Zero-token stage: turn a 40k-line log into a ~60-line report."""
        failures = prune_ci_log(state["raw_log"])
        report = render_failures(failures)
        diff = filter_diff(state.get("raw_diff", ""))

        parts = [f"FAILURE REPORT ({len(failures)} failing tests)\n{report}"]
        if diff.strip():
            parts.append(f"CHANGED CODE (generated files removed)\n{diff}")

        return {
            "messages": [
                stable_system_prompt(),
                HumanMessage(content="\n\n".join(parts)),
            ]
        }

    def agent(state: TriageState) -> dict:
        history = strip_stale_tool_payloads(list(state["messages"]))
        history = trim_history(history)
        response = model.invoke(history)
        return {
            "messages": [response],
            "prompt_tokens": count_message_tokens(history),
        }

    def compact(state: TriageState) -> dict:
        """Rewrite the tool results just produced into bounded digests."""
        rewritten: list[BaseMessage] = []
        for message in reversed(list(state["messages"])):
            if not isinstance(message, ToolMessage):
                break
            rewritten.append(
                ToolMessage(
                    content=compact_tool_result(
                        message.name or "tool", str(message.content)
                    ),
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                    id=message.id,
                )
            )
        return {"messages": list(reversed(rewritten))} if rewritten else {}

    def should_continue(state: TriageState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    graph = StateGraph(TriageState)
    graph.add_node("triage", triage)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_node("compact", compact)

    graph.set_entry_point("triage")
    graph.add_edge("triage", "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "compact")
    graph.add_edge("compact", "agent")

    return graph.compile()


def run(llm, raw_log: str, raw_diff: str = "") -> TriageState:
    app = build_graph(llm)
    return app.invoke(
        {"messages": [], "raw_log": raw_log, "raw_diff": raw_diff, "prompt_tokens": 0},
        config={"recursion_limit": 12},
    )
