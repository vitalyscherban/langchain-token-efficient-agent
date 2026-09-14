"""Stage 4: caching.

Two different caches, often confused:

* ``SQLiteCache`` -- exact-match response cache. Re-running the same prompt in a
  retry loop or a flaky-CI rerun costs zero tokens.
* Prompt-prefix caching -- provider-side. It only fires if the *prefix* is byte
  identical, so all static content (persona, repo conventions, tool docs) must
  come first and nothing volatile (timestamps, run IDs, random ordering) may be
  interleaved. ``stable_system_prompt`` enforces that discipline.
"""

from __future__ import annotations

from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import SystemMessage

from .prompts import STATIC_PREFIX

__all__ = ["enable_response_cache", "stable_system_prompt", "STATIC_PREFIX"]


def enable_response_cache(path: str | None = ".langchain.db") -> None:
    """Turn on exact-match response caching.

    Persists to SQLite when ``langchain_community`` is present so cache hits
    survive across CI jobs; falls back to an in-process cache otherwise. Pass
    ``path=None`` to force the in-memory cache (useful in tests).
    """
    if path is None:
        set_llm_cache(InMemoryCache())
        return
    try:
        from langchain_community.cache import SQLiteCache
    except ImportError:
        set_llm_cache(InMemoryCache())
        return
    set_llm_cache(SQLiteCache(database_path=path))


def stable_system_prompt(repo_conventions: str = "") -> SystemMessage:
    """Static-first system message so the provider can cache the prefix.

    Anything volatile belongs in the HumanMessage, never here.
    """
    content = STATIC_PREFIX
    if repo_conventions:
        content += "\n" + repo_conventions.strip() + "\n"
    return SystemMessage(content=content)
