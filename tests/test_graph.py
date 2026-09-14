"""End-to-end graph test with a scripted fake model -- no API key, no network.

Verifies the token-discipline invariants that actually matter:
  * the raw 27k-token log never reaches the model;
  * tool payloads are compacted before they re-enter history;
  * the prompt stays under budget even after several tool round-trips.
"""

from typing import Any, Optional

import pytest
from fixtures import SAMPLE_DIFF, make_ci_log

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.config import BUDGET
from agent.graph import build_graph
from agent.tokens import count_message_tokens


class ScriptedModel(BaseChatModel):
    """Replays a fixed script and records every prompt it was handed."""

    script: list[AIMessage] = []
    seen_prompts: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.seen_prompts.append(list(messages))
        # Advance on evidence, not on call count: a warm response cache can skip
        # _generate entirely, and a counter would then loop forever.
        step = 1 if any(m.type == "tool" for m in messages) else 0
        index = min(step, len(self.script) - 1)
        return ChatResult(generations=[ChatGeneration(message=self.script[index])])

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001 - test double
        return self


@pytest.fixture
def model(tmp_path):
    target = tmp_path / "refund_policy.py"
    target.write_text("\n".join(f"line_{i} = {i}" for i in range(400)), encoding="utf-8")
    scripted = ScriptedModel(
        cache=False,  # the real agent caches responses; the double must not
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_source_window",
                        "args": {"path": str(target), "line": 42},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="ROOT CAUSE: naive datetime. CONFIDENCE: 0.9"),
        ],
        seen_prompts=[],
    )
    return scripted


def test_graph_reaches_a_verdict(model):
    result = build_graph(model).invoke(
        {
            "messages": [],
            "raw_log": make_ci_log(),
            "raw_diff": SAMPLE_DIFF,
            "prompt_tokens": 0,
        },
        config={"recursion_limit": 12},
    )
    assert "ROOT CAUSE" in result["messages"][-1].content


def test_raw_log_never_reaches_the_model(model):
    build_graph(model).invoke(
        {
            "messages": [],
            "raw_log": make_ci_log(),
            "raw_diff": SAMPLE_DIFF,
            "prompt_tokens": 0,
        },
        config={"recursion_limit": 12},
    )
    for prompt in model.seen_prompts:
        joined = "\n".join(str(m.content) for m in prompt)
        assert "PASSED" not in joined, "passing-test noise leaked into the prompt"
        assert "package-lock.json" not in joined, "lockfile diff leaked into the prompt"


def test_prompt_stays_within_budget(model):
    build_graph(model).invoke(
        {
            "messages": [],
            "raw_log": make_ci_log(),
            "raw_diff": SAMPLE_DIFF,
            "prompt_tokens": 0,
        },
        config={"recursion_limit": 12},
    )
    for prompt in model.seen_prompts:
        assert count_message_tokens(prompt) <= BUDGET.max_history_tokens * 1.2


def test_tool_payload_is_compacted(model):
    build_graph(model).invoke(
        {
            "messages": [],
            "raw_log": make_ci_log(),
            "raw_diff": SAMPLE_DIFF,
            "prompt_tokens": 0,
        },
        config={"recursion_limit": 12},
    )
    final_prompt = model.seen_prompts[-1]
    tool_texts = [str(m.content) for m in final_prompt if m.type == "tool"]
    assert tool_texts, "expected a tool result in history"
    assert all(len(text) <= BUDGET.tool_result_char_limit + 120 for text in tool_texts)
