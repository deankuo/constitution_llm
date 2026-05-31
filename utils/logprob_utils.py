"""
Token-level log probability utilities for uncertainty quantification.

Gemini returns token-level log probabilities via response_logprobs=True.
This module extracts per-indicator log probabilities from the raw token stream
by finding each indicator's value token in the generated JSON.

Log probability interpretation:
  log_prob = 0.0   → probability 1.0 → model is completely certain
  log_prob = -0.1  → probability ~0.90 → high confidence
  log_prob = -0.7  → probability ~0.50 → coin-flip uncertainty
  log_prob = -2.3  → probability ~0.10 → very uncertain

Usage:
    from utils.logprob_utils import extract_indicator_logprobs

    logprobs = extract_indicator_logprobs(
        logprobs_result=response.candidates[0].logprobs_result,
        json_text=response.text,
        indicator_valid_labels={'sovereign': ['0', '1'], 'assembly': ['0', '1', '2', '3']}
    )
    # {'sovereign': -0.023, 'assembly': -1.12}
"""

import re
from typing import Any, Dict, List, Optional


def extract_indicator_logprobs(
    logprobs_result: Any,
    json_text: str,
    indicator_valid_labels: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Extract per-indicator log probabilities from a Gemini logprobs_result.

    Reconstructs the full text from chosen token candidates, builds a
    character-offset → log_probability mapping, then locates each indicator's
    value in the JSON text to retrieve the log probability of the token at
    that position.

    Works for both single_builder (all indicators in one JSON) and
    multiple_builder (one indicator per JSON).

    Args:
        logprobs_result: response.candidates[0].logprobs_result (Gemini SDK object).
                         Must have a `chosen_candidates` attribute.
        json_text: The full JSON string returned by the model (response.text).
        indicator_valid_labels: Mapping of indicator name → list of valid label strings.
                                 Example: {'sovereign': ['0', '1']}

    Returns:
        Dict mapping indicator_name → log_probability of the chosen value token.
        Returns empty dict if logprobs are unavailable or parsing fails.
        Indicators not found in the JSON are omitted from the result.
    """
    if logprobs_result is None:
        return {}

    chosen_candidates = getattr(logprobs_result, 'chosen_candidates', None)
    if not chosen_candidates:
        return {}

    # Build a character-offset → logprob mapping from the token stream.
    # Every character within a token gets the same logprob as that token.
    char_logprob: Dict[int, float] = {}
    offset = 0
    for cand in chosen_candidates:
        token: str = cand.token
        logprob: float = cand.log_probability
        for i in range(len(token)):
            char_logprob[offset + i] = logprob
        offset += len(token)

    result: Dict[str, float] = {}

    for indicator, valid_labels in indicator_valid_labels.items():
        logprob = _find_value_logprob(
            json_text, indicator, valid_labels, char_logprob
        )
        if logprob is not None:
            result[indicator] = logprob

    return result


def _find_value_logprob(
    json_text: str,
    indicator: str,
    valid_labels: List[str],
    char_logprob: Dict[int, float],
) -> Optional[float]:
    """
    Find the log probability of an indicator's value token in the JSON text.

    Tries three patterns in order:
      1. Quoted string value:   "indicator": "label"
      2. Unquoted numeric:      "indicator": label  (for numeric labels)
      3. Fallback character scan near ': ' after the key

    Returns the log probability of the character that starts the label value,
    or None if the value cannot be located.
    """
    key_pattern_base = r'"' + re.escape(indicator) + r'"\s*:\s*'

    for label in valid_labels:
        # --- Pattern 1: quoted string value ---
        pattern_quoted = re.compile(
            key_pattern_base + r'"(' + re.escape(label) + r')"'
        )
        match = pattern_quoted.search(json_text)
        if match:
            # group(1) is the label; match.start(1) is its character position
            logprob = _logprob_at(char_logprob, match.start(1))
            if logprob is not None:
                return logprob
            # Fallback: try one character back (opening quote might be the token)
            logprob = _logprob_at(char_logprob, match.start(1) - 1)
            if logprob is not None:
                return logprob

        # --- Pattern 2: unquoted numeric value ---
        pattern_num = re.compile(
            key_pattern_base + r'(' + re.escape(label) + r')(?!\d)'
        )
        match = pattern_num.search(json_text)
        if match:
            logprob = _logprob_at(char_logprob, match.start(1))
            if logprob is not None:
                return logprob

    return None


def _logprob_at(char_logprob: Dict[int, float], pos: int) -> Optional[float]:
    """Return the logprob at character position pos, or None if not mapped."""
    return char_logprob.get(pos)


def logprob_to_prob(logprob: Optional[float]) -> Optional[float]:
    """
    Convert a log probability to a probability (0–1).

    Args:
        logprob: Log probability (≤ 0), or None.

    Returns:
        Probability in [0, 1], or None.
    """
    if logprob is None:
        return None
    import math
    return math.exp(logprob)


def get_top_alternatives(
    logprobs_result: Any,
    json_text: str,
    indicator: str,
    valid_labels: List[str],
) -> Optional[List[Dict]]:
    """
    Return the top alternative token log probabilities at the indicator's value position.

    Useful for understanding what other labels the model considered.

    Args:
        logprobs_result: response.candidates[0].logprobs_result
        json_text: Full JSON string from model
        indicator: Indicator name
        valid_labels: Valid label strings for this indicator

    Returns:
        List of {'token': str, 'log_probability': float} dicts for top alternatives,
        or None if not available.
    """
    if logprobs_result is None:
        return None

    chosen_candidates = getattr(logprobs_result, 'chosen_candidates', None)
    top_candidates_list = getattr(logprobs_result, 'top_candidates', None)

    if not chosen_candidates or not top_candidates_list:
        return None

    # Build char → token index mapping
    char_token_idx: Dict[int, int] = {}
    offset = 0
    for idx, cand in enumerate(chosen_candidates):
        for i in range(len(cand.token)):
            char_token_idx[offset + i] = idx
        offset += len(cand.token)

    # Find the value's character position
    char_logprob = {pos: chosen_candidates[char_token_idx[pos]].log_probability
                    for pos in char_token_idx}
    value_logprob = _find_value_logprob(json_text, indicator, valid_labels, char_logprob)

    if value_logprob is None:
        return None

    # Find the token index for the value
    # Re-derive: find which token has the matching logprob near the value
    key_pattern = r'"' + re.escape(indicator) + r'"\s*:\s*'
    match = re.search(key_pattern, json_text)
    if not match:
        return None

    # Find approximate character position of the value after the key
    value_search_start = match.end()
    # Skip opening quote or whitespace
    for pos in range(value_search_start, min(value_search_start + 10, len(json_text))):
        if json_text[pos] in ('"', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'):
            token_idx = char_token_idx.get(pos)
            if token_idx is not None and token_idx < len(top_candidates_list):
                top = top_candidates_list[token_idx]
                return [
                    {'token': c.token, 'log_probability': c.log_probability}
                    for c in top.candidates
                ]

    return None
