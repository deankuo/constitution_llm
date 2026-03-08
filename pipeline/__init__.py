"""
Prediction pipeline components.

This package contains:
- predictor: Core prediction orchestrator
- batch_runner: Batch processing with checkpoints
- search_predictor: Search-augmented predictions (agentic search)
- pre_search: Deterministic pre-search (Wikipedia/DuckDuckGo/Serper)
- batch_gemini: Gemini Batch API runner (50% cost savings)
- classify_assembly: Assembly extended classifier (downstream)
"""

__version__ = '1.0.0'
