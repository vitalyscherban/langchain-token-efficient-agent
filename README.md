# Token-Efficient LangChain Agent

A LangGraph CI-triage agent built to answer one question: **how few tokens can you
spend and still fix the bug?**

It takes a raw CI log (~27k tokens of mostly noise) plus a PR diff, and returns a
root cause, a patch, and a confidence score — while sending the model **~95% fewer
prompt tokens** than the obvious implementation.

```
TOKENS PER TRIAGE RUN
lever                          naive optimized     saved
--------------------------------------------------------
CI log in prompt              26,936       417     98.5%
PR diff in prompt             10,381       111     98.9%
source file reads             23,398     2,849     87.8%
history replay (6 turns)     140,388     5,698     95.9%
system prompt                    130       130      0.0%
--------------------------------------------------------
TOTAL                        201,233     9,205     95.4%

cost/run   $0.5031 -> $0.0230
at 200 runs/day: $3,018.49/mo -> $138.07/mo
```

Reproduce it yourself — no API key required:

```bash
pip install tiktoken
python benchmarks/compare.py
```

## The five levers

| # | Lever | Where | Idea |
|---|-------|-------|------|
| 1 | **Deterministic pruning** | `log_pruner.py` | Extract failure blocks and stack frames with regex. Costs *zero* tokens — never pay a model to do what `re` can do. |
| 2 | **Denylist filtering** | `log_pruner.filter_diff` | Drop lockfiles, `dist/`, snapshots, generated protobufs before they reach the prompt. |
| 3 | **Windowed reads** | `windows.py`, `tools.py` | `file_outline` to map a file, then `read_source_window(path, line)` for ±40 lines. Never `read_file`. |
| 4 | **History compaction** | `context.py` | `trim_messages` for a token ceiling, summary folding for old turns, and tool-payload digests so consumed results stop being re-sent every turn. |
| 5 | **Caching** | `cache.py` | `SQLiteCache` for exact-match reruns, plus a static-first system prompt so provider-side *prefix* caching actually hits. |

Lever 4 is the one people miss. In a 6-turn agent loop, every raw tool result is
re-sent on every subsequent turn — so a single 3k-token file read costs 18k tokens,
not 3k. Compacting at the moment of production beats trimming later.

## Architecture

```
triage ──> agent ──(tool calls)──> tools ──> compact ──┐
             ^                                          │
             └──────────────────────────────────────────┘
             │
             └──(no tool calls)──> END
```

* **`triage`** — zero-token stage. 40k lines in, ~60 lines out.
* **`agent`** — strips stale tool payloads, trims to budget, then calls the model.
* **`tools`** — outline / window / grep. All return locations or slices, never bodies.
* **`compact`** — rewrites each fresh tool result into a bounded digest *in place*.

> The `compact` node depends on LangGraph's `add_messages` reducer, which replaces a
> message when an update reuses its `id`. With a plain `operator.add` reducer the
> compacted copy is *appended* instead and nothing shrinks — a silent failure the
> test suite pins down (`test_tool_payload_is_compacted`).

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env          # add OPENAI_API_KEY

python run_triage.py --demo               # synthetic failing build
python run_triage.py --log ci.log --diff pr.diff
python benchmarks/compare.py              # token accounting, no key needed
pytest                                    # 16 tests, no network
```

## Tuning

Every knob lives in `config.py`:

```python
max_history_tokens    = 3_000   # ceiling handed to the model each turn
keep_recent_turns     = 6       # verbatim before summary folding kicks in
source_window_lines   = 40      # radius around a stack-frame line
tool_result_char_limit= 800     # digest size for a consumed tool result
top_k_before_rerank   = 20      # candidates pulled from the vector store
top_k_after_rerank    = 3       # chunks that survive compression
```

These are recall-vs-cost trades. Tighten them until the test suite's fidelity
assertions (`test_finds_every_failure`, `test_culprit_frame_skips_site_packages`)
start failing — that's your floor.

## Measuring in production

`benchmarks/compare.py` is the offline proxy. For live numbers, set
`LANGCHAIN_TRACING_V2=true` and read prompt tokens per run from LangSmith. The
graph also accumulates `prompt_tokens` in its own state, so you can assert on it
in CI and fail the build when a change regresses the budget.

## Retrieval

`retrieval.py` wraps any vector retriever in split → dedupe → relevance → extract:

```python
retriever = build_compressed_retriever(vectorstore.as_retriever(), embeddings, llm)
```

The order matters. Cheap embedding stages cut 20 candidates to 3 before the
expensive `LLMChainExtractor` ever runs.

## Caveats

* Token counts use `tiktoken` (`cl100k_base`), falling back to a 4-chars-per-token
  heuristic when it isn't installed — good enough for ratios, not for billing.
* The pruner's regexes target pytest output. Other runners need new patterns; the
  `Failure`/`StackFrame` shape stays the same.
* Aggressive pruning trades recall for cost. Keep the fidelity tests honest.
