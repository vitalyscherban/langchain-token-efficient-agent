"""CLI: triage a CI log with the token-efficient agent.

    python run_triage.py --log ci.log --diff pr.diff
    python run_triage.py --demo          # synthetic log, no files needed

Requires OPENAI_API_KEY. Use ``python benchmarks/compare.py`` for the
token accounting, which needs no key at all.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "benchmarks"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, help="raw CI log file")
    parser.add_argument("--diff", type=Path, help="unified diff of the PR")
    parser.add_argument("--demo", action="store_true", help="use the bundled synthetic log")
    parser.add_argument("--model", default=os.getenv("SMART_MODEL", "gpt-4o"))
    args = parser.parse_args()

    if args.demo:
        from fixtures import SAMPLE_DIFF, make_ci_log

        raw_log, raw_diff = make_ci_log(), SAMPLE_DIFF
    elif args.log:
        raw_log = args.log.read_text(encoding="utf-8", errors="replace")
        raw_diff = args.diff.read_text(encoding="utf-8", errors="replace") if args.diff else ""
    else:
        parser.error("pass --log PATH or --demo")

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Try: python benchmarks/compare.py", file=sys.stderr)
        return 2

    from langchain_openai import ChatOpenAI

    from agent.graph import run
    from agent.tokens import count_tokens

    result = run(ChatOpenAI(model=args.model, temperature=0), raw_log, raw_diff)

    print(result["messages"][-1].content)
    print("\n--- token report ---")
    print(f"raw log tokens        {count_tokens(raw_log):>8,}")
    print(f"prompt tokens sent    {result['prompt_tokens']:>8,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
