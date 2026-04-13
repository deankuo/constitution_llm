"""
Prediction pipeline components.

This package contains:
- predictor: Core prediction orchestrator
- batch_runner: Batch processing with checkpoints
- search_predictor: Search-augmented predictions (agentic search)
- pre_search: Deterministic pre-search (Wikipedia/DuckDuckGo/Serper)
- batch_gemini: Gemini Batch API runner (50% cost savings)
- post_processing: Downstream classifiers (assembly_extended + elections)
"""

__version__ = '1.0.0'
