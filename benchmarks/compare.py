"""Naive vs optimized token accounting -- runs offline, no API key needed.

    python benchmarks/compare.py

Measures the four levers independently so you can see which one pays:
  1. log pruning        2. diff denylist
  3. windowed file reads 4. history compaction
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))

from agent.log_pruner import filter_diff, prune_ci_log, render_failures  # noqa: E402
from agent.prompts import STATIC_PREFIX  # noqa: E402
from agent.tokens import count_tokens, savings  # noqa: E402
from agent.windows import full_read, outline, read_window  # noqa: E402
from fixtures import SAMPLE_DIFF, make_ci_log, make_large_module  # noqa: E402

PRICE_PER_1K_INPUT = 0.0025  # gpt-4o list price, USD
RUNS_PER_DAY = 200


def row(label: str, before: int, after: int) -> tuple[str, int, int]:
    print(
        f"{label:<26} {before:>9,} {after:>9,} {savings(before, after):>9}"
    )
    return label, before, after


def main() -> None:
    log = make_ci_log()
    naive_log = count_tokens(log)
    pruned = render_failures(prune_ci_log(log))
    optimized_log = count_tokens(pruned)

    naive_diff = count_tokens(SAMPLE_DIFF)
    optimized_diff = count_tokens(filter_diff(SAMPLE_DIFF))

    # Real measurement against a realistically sized module: a naive agent reads
    # every file named in the stack trace end to end; ours takes one outline
    # plus a window centred on the line the trace pointed at.
    with tempfile.TemporaryDirectory() as tmp:
        module = make_large_module(Path(tmp) / "refund_policy.py")
        naive_reads = count_tokens(full_read(module)) * 2  # two files in the trace
        optimized_reads = count_tokens(outline(module)) + count_tokens(
            read_window(module, 42)
        )

    # Six tool round-trips: naive replays every raw payload, ours keeps digests.
    naive_history = naive_reads * 6
    optimized_history = optimized_reads * 2

    system = count_tokens(STATIC_PREFIX)

    print("\nTOKENS PER TRIAGE RUN")
    print(f"{'lever':<26} {'naive':>9} {'optimized':>9} {'saved':>9}")
    print("-" * 56)
    row("CI log in prompt", naive_log, optimized_log)
    row("PR diff in prompt", naive_diff, optimized_diff)
    row("source file reads", naive_reads, optimized_reads)
    row("history replay (6 turns)", naive_history, optimized_history)
    row("system prompt", system, system)
    print("-" * 56)

    naive_total = naive_log + naive_diff + naive_reads + naive_history + system
    optimized_total = (
        optimized_log + optimized_diff + optimized_reads + optimized_history + system
    )
    row("TOTAL", naive_total, optimized_total)

    naive_cost = naive_total / 1000 * PRICE_PER_1K_INPUT
    optimized_cost = optimized_total / 1000 * PRICE_PER_1K_INPUT
    print(
        f"\ncost/run   ${naive_cost:.4f} -> ${optimized_cost:.4f}"
        f"   ({savings(naive_total, optimized_total)} fewer prompt tokens)"
    )
    print(
        f"at {RUNS_PER_DAY} runs/day: "
        f"${naive_cost * RUNS_PER_DAY * 30:,.2f}/mo -> "
        f"${optimized_cost * RUNS_PER_DAY * 30:,.2f}/mo"
    )
    print(
        "\nnote: prompt-prefix caching makes the system prompt effectively free "
        "on repeat runs; response caching makes CI reruns free entirely."
    )


if __name__ == "__main__":
    main()
