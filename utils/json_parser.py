"""
Robust JSON parsing utilities for LLM responses.

This module provides functions to extract and parse JSON from LLM outputs
that may contain additional text or formatting artifacts.
"""

import json
import re
import time
from typing import Any, Dict, Optional


def extract_json_from_response(response: str) -> str:
    """
    Extract JSON object from LLM response that may contain additional text.

    Handles common formatting issues:
    - Markdown code fences (```json ... ```)
    - Text before/after the JSON
    - Nested braces

    Args:
        response: Raw LLM response

    Returns:
        Extracted JSON string or original response if no JSON found
    """
    if not response:
        return response

    # Try to remove markdown code fences first
    cleaned = response.strip()

    # Remove ```json or ``` at start
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]
    elif cleaned.startswith('```'):
        cleaned = cleaned[3:]

    # Remove ``` at end
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]

    cleaned = cleaned.strip()

    # Find the outermost JSON object
    try:
        start_index = cleaned.find('{')
        if start_index == -1:
            return response

        # Find the matching closing brace
        brace_count = 0
        end_index = -1

        for i, char in enumerate(cleaned[start_index:], start=start_index):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_index = i
                    break

        if end_index != -1:
            return cleaned[start_index:end_index + 1]

        # Fallback to simple rfind
        end_index = cleaned.rfind('}')
        if start_index != -1 and end_index != -1:
            return cleaned[start_index:end_index + 1]

    except Exception:
        pass

    return response


def parse_json_response(
    response: str,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Parse the LLM's JSON response with retry logic.

    Args:
        response: Raw LLM response, expected to contain a JSON object
        max_retries: Maximum number of retries for parsing
        retry_delay: Delay in seconds between retries
        verbose: Whether to print debug information

    Returns:
        Dictionary containing parsed results or error structure if parsing fails
    """
    clean_response = extract_json_from_response(response)

    if verbose:
        print("-" * 60)
        print(clean_response)
        print("-" * 60)

    for attempt in range(1, max_retries + 1):
        try:
            return json.loads(clean_response)
        except json.JSONDecodeError as e:
            if verbose:
                print(f"Error decoding JSON: {e}. Attempt {attempt} of {max_retries}.")

            if attempt < max_retries:
                if verbose:
                    print(f"Retrying in {retry_delay} seconds")
                time.sleep(retry_delay)

    # All retries failed - return error structure
    if verbose:
        print(f"All {max_retries} retries failed. Returning an error structure.")
        print(f"Final raw response that failed: {response}")

    return {
        'error': True,
        'error_message': 'JSON parsing failed after multiple retries',
        'raw_response': response
    }


def validate_indicator_response(
    parsed: Dict[str, Any],
    indicator: str,
    valid_labels: list
) -> Dict[str, Any]:
    """
    Validate that the parsed response contains expected fields for an indicator.

    Args:
        parsed: Parsed JSON response
        indicator: Name of the indicator (e.g., 'sovereign', 'assembly')
        valid_labels: List of valid label values (e.g., ['0', '1'] or ['0', '1', '2'])

    Returns:
        Validated response with defaults filled in if needed
    """
    result = parsed.copy()

    # Ensure indicator value exists and is valid
    if indicator not in result:
        result[indicator] = None
    elif str(result[indicator]) not in [str(v) for v in valid_labels]:
        # Try to convert numeric values
        try:
            val = str(int(float(result[indicator])))
            if val in valid_labels:
                result[indicator] = val
            else:
                result[indicator] = None
        except (ValueError, TypeError):
            result[indicator] = None
    else:
        result[indicator] = str(result[indicator])

    # Ensure reasoning exists (check both 'reasoning' and '{indicator}_reasoning')
    reasoning_key = f'{indicator}_reasoning'
    if 'reasoning' in result:
        result['reasoning'] = result['reasoning']
    elif reasoning_key in result:
        result['reasoning'] = result[reasoning_key]
    else:
        result['reasoning'] = ''

    # Ensure confidence_score exists (check both 'confidence_score' and '{indicator}_confidence_score')
    confidence_key = f'{indicator}_confidence_score'
    if 'confidence_score' in result:
        score_value = result['confidence_score']
    elif confidence_key in result:
        score_value = result[confidence_key]
    else:
        score_value = None

    if score_value is not None:
        try:
            score = int(score_value)
            if score < 1:
                score = 1
            elif score > 100:
                score = 100
            result['confidence_score'] = score
        except (ValueError, TypeError):
            result['confidence_score'] = None
    else:
        result['confidence_score'] = None

    return result


def validate_constitution_response(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that the parsed response contains expected fields for constitution.

    Args:
        parsed: Parsed JSON response

    Returns:
        Validated response with defaults filled in if needed
    """
    result = parsed.copy()

    # Handle constitution status - normalize "constitution" or "constitution_status"
    constitution_value = result.get('constitution') or result.get('constitution_status')
    if constitution_value:
        constitution_value = str(constitution_value).strip().lower()
        if constitution_value in ['yes', '1', 'true']:
            result['constitution'] = '1'
        elif constitution_value in ['no', '0', 'false']:
            result['constitution'] = '0'
        else:
            result['constitution'] = None
    else:
        result['constitution'] = None

    # Ensure document_name exists
    if 'document_name' not in result:
        result['document_name'] = 'N/A'

    # Ensure constitution_year exists
    # Keep as string to preserve "N/A" or semicolon-separated years like "1789; 1791"
    if 'constitution_year' not in result:
        result['constitution_year'] = None
    elif result['constitution_year'] is None:
        result['constitution_year'] = None
    else:
        # Convert to string and clean up
        year_str = str(result['constitution_year']).strip()
        # Handle "null" string or empty string
        if year_str.lower() in ['null', 'none', '']:
            result['constitution_year'] = None
        else:
            result['constitution_year'] = year_str

    # Ensure reasoning/explanation exists (check multiple field names)
    if 'reasoning' in result:
        result['reasoning'] = result['reasoning']
    elif 'constitution_reasoning' in result:
        result['reasoning'] = result['constitution_reasoning']
    elif 'explanation' in result:
        result['reasoning'] = result['explanation']
    else:
        result['reasoning'] = ''

    # Ensure confidence_score exists and is in valid range (1-100 for constitution)
    # Check both 'confidence_score' and 'constitution_confidence_score'
    if 'confidence_score' in result:
        score_value = result['confidence_score']
    elif 'constitution_confidence_score' in result:
        score_value = result['constitution_confidence_score']
    else:
        score_value = None

    if score_value is not None:
        try:
            score = int(score_value)
            if score < 1:
                score = 1
            elif score > 100:
                score = 100
            result['confidence_score'] = score
        except (ValueError, TypeError):
            result['confidence_score'] = None
    else:
        result['confidence_score'] = None

    return result
