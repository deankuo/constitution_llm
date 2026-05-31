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
        Validated response with defaults filled in if needed.
        Predictions are returned as float (0.0, 1.0, 2.0) to match ground truth format
        and properly handle NaN values.
    """
    result = parsed.copy()

    # ------------------------------------------------------------------
    # Resolve the indicator value, trying several keys and formats.
    # ------------------------------------------------------------------
    # 1. Primary key is the bare indicator name (e.g. "appointment").
    # 2. Fallback: "{indicator}_prediction" (LLM sometimes uses the column name).
    raw_value = result.get(indicator)
    if raw_value is None:
        raw_value = result.get(f'{indicator}_prediction')

    str_labels = [str(v) for v in valid_labels]

    def _parse_label(val) -> 'float | None':
        """Convert val to a float label, or return None if unresolvable."""
        if val is None:
            return None
        # Direct string match ("0", "1", "2")
        if str(val) in str_labels:
            try:
                return float(val)
            except (ValueError, TypeError):
                return None
        # Numeric coercion (handles integers 0/1/2 and floats 0.0/1.0/2.0)
        try:
            coerced = str(int(float(val)))
            if coerced in str_labels:
                return float(coerced)
        except (ValueError, TypeError):
            pass
        # Leading-digit extraction for descriptive strings like "2 (by election)"
        m = re.match(r'^\s*([0-9]+)', str(val))
        if m and m.group(1) in str_labels:
            return float(m.group(1))
        return None

    parsed_label = _parse_label(raw_value)
    result[indicator] = parsed_label

    # Ensure reasoning exists (check both 'reasoning' and '{indicator}_reasoning')
    reasoning_key = f'{indicator}_reasoning'
    if 'reasoning' in result:
        result['reasoning'] = result['reasoning']
    elif reasoning_key in result:
        result['reasoning'] = result[reasoning_key]
    else:
        result['reasoning'] = ''

    # Resolve confidence_score from any of the key variants the LLM might emit:
    #   'confidence_score'            (multiple_builder / generic)
    #   '{indicator}_confidence'      (single_builder — e.g. sovereign_confidence)
    #   '{indicator}_confidence_score' (legacy variant)
    for _conf_key in (
        'confidence_score',
        f'{indicator}_confidence',
        f'{indicator}_confidence_score',
    ):
        if _conf_key in result:
            score_value = result[_conf_key]
            break
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


def _clean_constitution_year(year_str: str) -> str:
    """
    Clean constitution year string by stripping approximation prefixes.

    Handles patterns like "c. 1240", "circa 1240", "approximately 1240",
    "ca. 1240", "~1240", and semicolon-separated multiples.

    Args:
        year_str: Raw year string from LLM

    Returns:
        Cleaned year string with only integer years (semicolon-separated if multiple)
    """
    # Split by semicolons for multiple years
    parts = [p.strip() for p in year_str.split(';')]
    cleaned_parts = []

    for part in parts:
        if not part:
            continue
        # Strip common approximation prefixes
        cleaned = part.strip()
        cleaned = re.sub(r'^(?:c\.?\s*|circa\s+|approx(?:imately)?\.?\s*|ca\.?\s*|~\s*)', '', cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()

        # Try to extract a valid integer year
        year_match = re.search(r'-?\d+', cleaned)
        if year_match:
            cleaned_parts.append(year_match.group())
        else:
            # If no number found, keep original (may be "N/A" or other valid text)
            cleaned_parts.append(part.strip())

    return '; '.join(cleaned_parts) if cleaned_parts else year_str


def validate_constitution_response(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that the parsed response contains expected fields for constitution.

    Three-type scheme:
      0 = no written code of law or constitution
      1 = code of law for subjects/citizens only
      2 = full constitution (code of law + governance rules + limitations)

    Constitution prediction is returned as float (0.0, 1.0, or 2.0) to match
    the ground-truth format and properly handle NaN values.

    Also parses 'document_types' — a semicolon-separated list of per-document
    type integers parallel to 'document_name'. Cross-checks that
    max(document_types) == constitution prediction.
    """
    result = parsed.copy()

    # ------------------------------------------------------------------
    # 1. Parse 'constitution' as primary integer field (0, 1, or 2)
    # ------------------------------------------------------------------
    # Use explicit None check — 0 is a valid value and must not be dropped by `or`
    _raw = result.get('constitution')
    constitution_value = _raw if _raw is not None else result.get('constitution_status')
    parsed_label: Optional[float] = None

    if constitution_value is not None:
        val_str = str(constitution_value).strip().lower()
        # New format: integer 0/1/2
        if val_str in ('0', '1', '2'):
            parsed_label = float(val_str)
        # Old binary fallback: "yes"/"no" (backwards compat)
        elif val_str in ('yes', 'true'):
            parsed_label = 2.0
        elif val_str in ('no', 'false'):
            parsed_label = 0.0
        else:
            # Try numeric coercion
            try:
                coerced = int(float(val_str))
                if coerced in (0, 1, 2):
                    parsed_label = float(coerced)
            except (ValueError, TypeError):
                pass

    result['constitution'] = parsed_label

    # ------------------------------------------------------------------
    # 2. Parse 'document_name'
    # ------------------------------------------------------------------
    if 'document_name' not in result or result.get('document_name') is None:
        result['document_name'] = 'N/A'
    else:
        result['document_name'] = str(result['document_name']).strip() or 'N/A'

    # ------------------------------------------------------------------
    # 3. Parse 'document_types' (new field: semicolon-separated integers)
    # ------------------------------------------------------------------
    raw_doc_types = result.get('document_types')
    document_types_parsed: Optional[list] = None

    if raw_doc_types is not None:
        raw_str = str(raw_doc_types).strip()
        if raw_str.lower() not in ('n/a', 'none', '', 'null'):
            parts = [p.strip() for p in raw_str.split(';') if p.strip()]
            types = []
            all_valid = True
            for part in parts:
                try:
                    t = int(float(part))
                    if t in (0, 1, 2):
                        types.append(t)
                    else:
                        all_valid = False
                        break
                except (ValueError, TypeError):
                    all_valid = False
                    break
            if all_valid and types:
                document_types_parsed = types
                result['document_types'] = '; '.join(str(t) for t in types)
            else:
                result['document_types'] = None
        else:
            result['document_types'] = None
    else:
        result['document_types'] = None

    # ------------------------------------------------------------------
    # 4. Cross-check: max(document_types) should equal constitution
    #    If constitution field is absent but document_types is valid, derive it.
    # ------------------------------------------------------------------
    if document_types_parsed:
        derived_max = float(max(document_types_parsed))
        if result['constitution'] is None:
            result['constitution'] = derived_max
        elif result['constitution'] != derived_max:
            # Trust 'constitution' field; log discrepancy silently
            pass

    # ------------------------------------------------------------------
    # 5. Parse 'constitution_year'
    # ------------------------------------------------------------------
    if 'constitution_year' not in result or result['constitution_year'] is None:
        result['constitution_year'] = None
    else:
        year_str = str(result['constitution_year']).strip()
        if year_str.lower() in ('null', 'none', '', 'n/a'):
            result['constitution_year'] = None
        else:
            result['constitution_year'] = _clean_constitution_year(year_str)

    # ------------------------------------------------------------------
    # 6. Reasoning
    # ------------------------------------------------------------------
    if 'reasoning' in result:
        pass
    elif 'constitution_reasoning' in result:
        result['reasoning'] = result['constitution_reasoning']
    elif 'explanation' in result:
        result['reasoning'] = result['explanation']
    else:
        result['reasoning'] = ''

    # ------------------------------------------------------------------
    # 7. Confidence score
    # ------------------------------------------------------------------
    score_value = result.get('confidence_score') or result.get('constitution_confidence_score')
    if score_value is not None:
        try:
            score = int(score_value)
            result['confidence_score'] = max(1, min(100, score))
        except (ValueError, TypeError):
            result['confidence_score'] = None
    else:
        result['confidence_score'] = None

    return result
