"""
LangSmith integration utilities.

Provides conditional tracing support via LangSmith. When LANGCHAIN_TRACING_V2=true
and the langsmith package is installed, LLM calls and pipeline functions are
automatically traced. Otherwise, all decorators and wrappers are no-ops.

Required environment variables (set in .env):
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=<your-langsmith-api-key>
    LANGCHAIN_PROJECT=constitution-llm    # optional, defaults to "default"

Usage::

    from utils.langsmith_utils import traceable, wrap_openai_client, wrap_anthropic_client

    @traceable(name="predict_indicator")
    def predict(polity, ...):
        ...

    # Wrap SDK clients for automatic call tracing
    client = wrap_openai_client(OpenAI(api_key=...))
    client = wrap_anthropic_client(Anthropic(api_key=...))
"""

import os
from functools import wraps
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Detect whether LangSmith tracing is active
# ---------------------------------------------------------------------------

_TRACING_ENABLED: bool = False

try:
    if os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true":
        import langsmith  # noqa: F401
        _TRACING_ENABLED = True
except ImportError:
    pass


def is_tracing_enabled() -> bool:
    """Return True if LangSmith tracing is active."""
    return _TRACING_ENABLED


# ---------------------------------------------------------------------------
# Conditional @traceable decorator
# ---------------------------------------------------------------------------

def traceable(
    name: Optional[str] = None,
    run_type: str = "chain",
    **kwargs,
) -> Callable:
    """
    Conditional LangSmith ``@traceable`` decorator.

    When tracing is enabled, wraps the function with ``langsmith.traceable``.
    Otherwise returns the function unchanged (zero overhead).

    Args:
        name:     Display name in LangSmith (defaults to function name).
        run_type: "chain", "llm", "tool", "retriever", etc.
        **kwargs: Additional arguments forwarded to ``langsmith.traceable``.
    """
    if _TRACING_ENABLED:
        from langsmith import traceable as _ls_traceable
        return _ls_traceable(name=name, run_type=run_type, **kwargs)

    # No-op: return the function unmodified
    def _noop_decorator(fn: Callable) -> Callable:
        return fn
    return _noop_decorator


# ---------------------------------------------------------------------------
# SDK client wrappers
# ---------------------------------------------------------------------------

def wrap_openai_client(client: Any) -> Any:
    """
    Wrap an OpenAI client for automatic LangSmith tracing.

    Returns the original client unchanged if tracing is disabled.
    """
    if not _TRACING_ENABLED:
        return client
    try:
        from langsmith.wrappers import wrap_openai
        return wrap_openai(client)
    except (ImportError, Exception):
        return client


def wrap_anthropic_client(client: Any) -> Any:
    """
    Wrap an Anthropic client for automatic LangSmith tracing.

    Returns the original client unchanged if tracing is disabled.
    """
    if not _TRACING_ENABLED:
        return client
    try:
        from langsmith.wrappers import wrap_anthropic
        return wrap_anthropic(client)
    except (ImportError, Exception):
        return client
