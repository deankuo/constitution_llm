"""
API cost tracking utilities for LLM providers.

This module provides cost tracking and estimation for different LLM providers.
Prices are based on publicly available pricing as of 2024-2025.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import json


# Pricing per 1M tokens (input/output/cached) - update as needed
# Cached pricing applies to Gemini models when using context caching
PRICING = {
    # OpenAI models (per 1M tokens)
    'gpt-4o': {'input': 2.50, 'output': 10.00, 'cached': 1.25},
    'gpt-4o-mini': {'input': 0.15, 'output': 0.60, 'cached': 0.075},
    'gpt-4-turbo': {'input': 10.00, 'output': 30.00, 'cached': 5.00},
    'gpt-4': {'input': 30.00, 'output': 60.00, 'cached': 15.00},
    'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50, 'cached': 0.25},
    'o1': {'input': 15.00, 'output': 60.00, 'cached': 7.50},
    'o1-mini': {'input': 3.00, 'output': 12.00, 'cached': 1.50},
    'o3-mini': {'input': 1.10, 'output': 4.40, 'cached': 0.55},

    # Anthropic models (per 1M tokens)
    # Anthropic has prompt caching: cached reads are 90% cheaper
    'claude-3-5-sonnet-20241022': {'input': 3.00, 'output': 15.00, 'cached': 0.30},
    'claude-3-5-haiku-20241022': {'input': 0.80, 'output': 4.00, 'cached': 0.08},
    'claude-3-opus-20240229': {'input': 15.00, 'output': 75.00, 'cached': 1.50},
    'claude-3-sonnet-20240229': {'input': 3.00, 'output': 15.00, 'cached': 0.30},
    'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25, 'cached': 0.025},

    # Google Gemini models (per 1M tokens)
    # Gemini context caching: cached tokens are ~10% of input price
    'gemini-2.5-pro': {'input': 1.25, 'output': 10.00, 'cached': 0.125},
    'gemini-2.5-flash': {'input': 0.15, 'output': 0.60, 'cached': 0.015},
    'gemini-2.0-flash': {'input': 0.10, 'output': 0.40, 'cached': 0.01},
    'gemini-1.5-pro': {'input': 1.25, 'output': 5.00, 'cached': 0.125},
    'gemini-1.5-flash': {'input': 0.075, 'output': 0.30, 'cached': 0.0075},

    # AWS Bedrock Claude models (per 1M tokens, cross-region pricing)
    # Bedrock also supports prompt caching
    'anthropic.claude-3-5-sonnet-20241022-v2:0': {'input': 3.00, 'output': 15.00, 'cached': 0.30},
    'anthropic.claude-3-5-haiku-20241022-v1:0': {'input': 1.00, 'output': 5.00, 'cached': 0.10},
    'anthropic.claude-3-opus-20240229-v1:0': {'input': 15.00, 'output': 75.00, 'cached': 1.50},
    'anthropic.claude-3-sonnet-20240229-v1:0': {'input': 3.00, 'output': 15.00, 'cached': 0.30},
    'anthropic.claude-3-haiku-20240307-v1:0': {'input': 0.25, 'output': 1.25, 'cached': 0.025},
    'anthropic.claude-sonnet-4-5-20250929-v1:0': {'input': 3.00, 'output': 15.00, 'cached': 0.30},
    'anthropic.claude-opus-4-5-20250514-v1:0': {'input': 15.00, 'output': 75.00, 'cached': 1.50},

    # Bedrock ARN extractions will also match (e.g., global.anthropic.claude-*)
    'global.anthropic.claude-sonnet-4-5-20250929-v1:0': {'input': 3.00, 'output': 15.00, 'cached': 0.30},
    'global.anthropic.claude-opus-4-5-20250514-v1:0': {'input': 15.00, 'output': 75.00, 'cached': 1.50},
    'us.anthropic.claude-sonnet-4-5-20250929-v1:0': {'input': 3.00, 'output': 15.00, 'cached': 0.30},
    'us.anthropic.claude-opus-4-5-20250514-v1:0': {'input': 15.00, 'output': 75.00, 'cached': 1.50},
}


@dataclass
class UsageRecord:
    """Record of a single API call's usage."""
    model: str
    input_tokens: int
    output_tokens: int  # Includes thinking tokens for thinking models
    cost_usd: float
    timestamp: datetime = field(default_factory=datetime.now)
    polity: Optional[str] = None
    indicator: Optional[str] = None
    cached_tokens: int = 0
    thinking_tokens: int = 0  # Thinking tokens (included in output_tokens for billing)


@dataclass
class ModelUsage:
    """Aggregated usage for a single model."""
    model: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0  # Includes thinking tokens for thinking models
    total_cached_tokens: int = 0
    total_thinking_tokens: int = 0  # Thinking tokens (included in output_tokens for billing)
    total_cost_usd: float = 0.0
    call_count: int = 0


class CostTracker:
    """
    Track API costs across multiple models and calls.

    Example usage:
        tracker = CostTracker()

        # After each API call
        tracker.add_usage(
            model='gpt-4o',
            input_tokens=1000,
            output_tokens=500,
            polity='Roman Republic',
            indicator='constitution'
        )

        # Get summary
        print(tracker.get_summary())

        # Save report
        tracker.save_report('costs.json')
    """

    def __init__(self):
        """Initialize cost tracker."""
        self.records: List[UsageRecord] = []
        self.model_usage: Dict[str, ModelUsage] = {}
        self.start_time = datetime.now()

    def get_price(self, model: str) -> Dict[str, float]:
        """
        Get pricing for a model.

        Args:
            model: Model name, identifier, or ARN

        Returns:
            Dict with 'input', 'output', and 'cached' prices per 1M tokens
        """
        # Extract model ID from Bedrock ARN if present
        # Format: arn:aws:bedrock:region:account:inference-profile/model-id
        if model.startswith('arn:aws:bedrock'):
            # Extract the model-id part after the last /
            model = model.split('/')[-1]

        # Direct match
        if model in PRICING:
            return PRICING[model]

        # Try to match by prefix (for versioned models)
        model_lower = model.lower()
        for known_model, price in PRICING.items():
            if model_lower.startswith(known_model.lower().split('-')[0]):
                return price

        # Default fallback pricing (conservative estimate)
        return {'input': 5.00, 'output': 15.00, 'cached': 0.50}

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0
    ) -> float:
        """
        Calculate cost for a single API call.

        Args:
            model: Model name
            input_tokens: Number of input tokens (non-cached)
            output_tokens: Number of output tokens
            cached_tokens: Number of cached input tokens (priced separately)

        Returns:
            Cost in USD
        """
        prices = self.get_price(model)

        input_cost = (input_tokens / 1_000_000) * prices['input']
        output_cost = (output_tokens / 1_000_000) * prices['output']
        cached_cost = (cached_tokens / 1_000_000) * prices.get('cached', 0)

        return input_cost + output_cost + cached_cost

    def add_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        polity: Optional[str] = None,
        indicator: Optional[str] = None,
        cached_tokens: int = 0,
        thinking_tokens: int = 0
    ) -> float:
        """
        Record usage from an API call.

        Args:
            model: Model name
            input_tokens: Number of input tokens (non-cached)
            output_tokens: Number of output tokens (includes thinking tokens for thinking models)
            polity: Optional polity name for tracking
            indicator: Optional indicator name for tracking
            cached_tokens: Number of cached input tokens
            thinking_tokens: Number of thinking tokens (subset of output_tokens, for tracking)

        Returns:
            Cost of this call in USD
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens, cached_tokens)

        # Create record
        record = UsageRecord(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            polity=polity,
            indicator=indicator,
            cached_tokens=cached_tokens,
            thinking_tokens=thinking_tokens
        )
        self.records.append(record)

        # Update model aggregates
        if model not in self.model_usage:
            self.model_usage[model] = ModelUsage(model=model)

        usage = self.model_usage[model]
        usage.total_input_tokens += input_tokens
        usage.total_output_tokens += output_tokens
        usage.total_cached_tokens += cached_tokens
        usage.total_thinking_tokens += thinking_tokens
        usage.total_cost_usd += cost
        usage.call_count += 1

        return cost

    def get_total_cost(self) -> float:
        """Get total cost across all models."""
        return sum(u.total_cost_usd for u in self.model_usage.values())

    def get_model_cost(self, model: str) -> float:
        """Get total cost for a specific model."""
        if model in self.model_usage:
            return self.model_usage[model].total_cost_usd
        return 0.0

    def get_summary(self) -> Dict:
        """
        Get a summary of all usage.

        Returns:
            Dictionary with usage summary
        """
        summary = {
            'total_cost_usd': self.get_total_cost(),
            'total_calls': len(self.records),
            'total_input_tokens': sum(u.total_input_tokens for u in self.model_usage.values()),
            'total_output_tokens': sum(u.total_output_tokens for u in self.model_usage.values()),
            'total_cached_tokens': sum(u.total_cached_tokens for u in self.model_usage.values()),
            'total_thinking_tokens': sum(u.total_thinking_tokens for u in self.model_usage.values()),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'models': {}
        }

        for model, usage in self.model_usage.items():
            summary['models'][model] = {
                'calls': usage.call_count,
                'input_tokens': usage.total_input_tokens,
                'output_tokens': usage.total_output_tokens,
                'cached_tokens': usage.total_cached_tokens,
                'thinking_tokens': usage.total_thinking_tokens,
                'cost_usd': usage.total_cost_usd
            }

        return summary

    def get_indicator_costs(self) -> Dict[str, float]:
        """Get cost breakdown by indicator."""
        indicator_costs = {}
        for record in self.records:
            if record.indicator:
                if record.indicator not in indicator_costs:
                    indicator_costs[record.indicator] = 0.0
                indicator_costs[record.indicator] += record.cost_usd
        return indicator_costs

    def save_report(self, filepath: str) -> None:
        """
        Save detailed cost report to JSON file.

        Args:
            filepath: Path to output file
        """
        report = {
            'summary': self.get_summary(),
            'indicator_costs': self.get_indicator_costs(),
            'records': [
                {
                    'model': r.model,
                    'input_tokens': r.input_tokens,
                    'output_tokens': r.output_tokens,
                    'thinking_tokens': r.thinking_tokens,
                    'cached_tokens': r.cached_tokens,
                    'cost_usd': r.cost_usd,
                    'timestamp': r.timestamp.isoformat(),
                    'polity': r.polity,
                    'indicator': r.indicator
                }
                for r in self.records
            ]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

    def print_summary(self) -> None:
        """Print a formatted summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("COST SUMMARY")
        print("=" * 60)
        print(f"Total Cost: ${summary['total_cost_usd']:.4f}")
        print(f"Total API Calls: {summary['total_calls']}")
        print(f"Total Input Tokens: {summary['total_input_tokens']:,}")
        print(f"Total Output Tokens: {summary['total_output_tokens']:,}")
        if summary['total_thinking_tokens'] > 0:
            non_thinking = summary['total_output_tokens'] - summary['total_thinking_tokens']
            print(f"  - Regular Output: {non_thinking:,}")
            print(f"  - Thinking Tokens: {summary['total_thinking_tokens']:,}")
        print(f"Total Cached Tokens: {summary['total_cached_tokens']:,}")
        print(f"Duration: {summary['duration_seconds']:.1f} seconds")

        print("\nCost by Model:")
        for model, data in summary['models'].items():
            print(f"  {model}:")
            print(f"    Calls: {data['calls']}")
            print(f"    Input: {data['input_tokens']:,} tokens")
            print(f"    Output: {data['output_tokens']:,} tokens")
            if data['thinking_tokens'] > 0:
                non_thinking = data['output_tokens'] - data['thinking_tokens']
                print(f"      - Regular Output: {non_thinking:,} tokens")
                print(f"      - Thinking: {data['thinking_tokens']:,} tokens")
            print(f"    Cached: {data['cached_tokens']:,} tokens")
            print(f"    Cost: ${data['cost_usd']:.4f}")

        indicator_costs = self.get_indicator_costs()
        if indicator_costs:
            print("\nCost by Indicator:")
            for indicator, cost in sorted(indicator_costs.items()):
                print(f"  {indicator}: ${cost:.4f}")

        print("=" * 60 + "\n")


def estimate_batch_cost(
    model: str,
    num_polities: int,
    avg_input_tokens: int = 2000,
    avg_output_tokens: int = 500,
    num_indicators: int = 1,
    avg_cached_tokens: int = 0
) -> float:
    """
    Estimate cost for a batch run.

    Args:
        model: Model name
        num_polities: Number of polities to process
        avg_input_tokens: Average input tokens per call (non-cached)
        avg_output_tokens: Average output tokens per call
        num_indicators: Number of indicators per polity
        avg_cached_tokens: Average cached tokens per call (default 0)

    Returns:
        Estimated cost in USD
    """
    tracker = CostTracker()
    total_calls = num_polities * num_indicators

    return tracker.calculate_cost(
        model=model,
        input_tokens=avg_input_tokens * total_calls,
        output_tokens=avg_output_tokens * total_calls,
        cached_tokens=avg_cached_tokens * total_calls
    )
