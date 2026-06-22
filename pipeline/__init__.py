"""
Prediction pipeline components.

This package contains:
- predictor: Core prediction orchestrator
- batch_runner: Batch processing with checkpoints
- search_predictor: Search-augmented predictions (agentic search)
- pre_search: Deterministic pre-search (Wikipedia/DuckDuckGo/Serper)
- jsonl_batch_runner: Gemini Batch API runner via inline requests (50% cost savings, SC support)
- post_processing: Downstream classifiers (assembly_extended + elections)
"""

__version__ = '1.0.0'
