"""Central knobs for the token budget.

Every number here is a lever that trades tokens for recall. They are kept in one
place so a benchmark run can sweep them without touching agent code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenBudget:
    # Hard ceiling on the message history handed to the model each turn.
    max_history_tokens: int = 3_000
    # Turns kept verbatim before older ones are folded into a summary block.
    keep_recent_turns: int = 6
    # Lines of source read around a stack-frame line number.
    source_window_lines: int = 40
    # Max characters of a single tool result kept verbatim before compaction.
    tool_result_char_limit: int = 800
    # Retrieved chunks kept after reranking.
    top_k_after_rerank: int = 3
    # Candidate chunks pulled from the vector store before reranking.
    top_k_before_rerank: int = 20


@dataclass(frozen=True)
class Models:
    cheap: str = os.getenv("CHEAP_MODEL", "gpt-4o-mini")
    smart: str = os.getenv("SMART_MODEL", "gpt-4o")


BUDGET = TokenBudget()
MODELS = Models()

# Paths that never carry signal for a test failure but cost thousands of tokens.
DIFF_DENYLIST = (
    "package-lock.json",
    "yarn.lock",
    "poetry.lock",
    "Pipfile.lock",
    "go.sum",
    ".min.js",
    ".min.css",
    "/dist/",
    "/build/",
    "/vendor/",
    "/node_modules/",
    "__snapshots__/",
    ".pb.go",
    "_pb2.py",
)
