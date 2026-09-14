"""Prompt text kept dependency-free.

Deliberately importable without langchain installed so the benchmark and unit
tests can run in a bare container. It is also the single source of truth for the
*static prefix* that provider-side prompt caching keys on -- if this text drifts
between runs, the cache silently stops hitting.
"""

STATIC_PREFIX = """You are a CI triage agent.

Rules:
1. Diagnose from the pruned failure report. Do not ask for the full log.
2. Use file_outline before read_source_window; read windows, never whole files.
3. One tool call per turn. Stop as soon as you can name file, line, and cause.
4. Answer as: ROOT CAUSE (1 sentence) / FIX (unified diff) / CONFIDENCE (0-1).

Repo conventions:
- Source in src/, tests in tests/, pytest with strict markers.
- Timezone-aware datetimes only; naive datetime is a known bug class.
"""

SUMMARY_PROMPT = (
    "Compress the conversation below into a state block with exactly these "
    "headings: FILES TOUCHED, DECISIONS, OPEN TODOS, FAILING TESTS. "
    "Keep file paths and identifiers verbatim. Max 150 words. No prose."
)
