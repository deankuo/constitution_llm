"""
Gemini Batch API Runner

Submits prediction requests to the Gemini Batch API for 50% cost savings.
Reuses existing prompt builders and parsing logic; only the LLM call is
replaced by batch submission + polling.

Requires the ``google-genai`` SDK::

    pip install google-genai

Usage from CLI::

    python main.py --pipeline leader \\
        --indicators sovereign assembly \\
        --model gemini-2.5-pro \\
        --use-batch

Usage from Python::

    from pipeline.batch_gemini import GeminiBatchRunner
    runner = GeminiBatchRunner(predictor, batch_config, output_path)
    results_df = runner.run(df)
"""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from config import (
    COL_TERRITORY_NAME, COL_LEADER_NAME,
    COL_START_YEAR, COL_END_YEAR,
    INDICATOR_LABELS, REQUIRED_COLUMNS,
    VerificationType,
)
from pipeline.batch_runner import BatchConfig
from pipeline.predictor import Predictor, IndicatorPrediction, PolityPrediction
from utils.json_parser import (
    parse_json_response,
    validate_indicator_response,
    validate_constitution_response,
)

# Batch API discount (50 %)
BATCH_DISCOUNT = 0.5


def _import_genai():
    """Lazily import google.genai and return the module + types."""
    try:
        from google import genai
        from google.genai import types as genai_types
        return genai, genai_types
    except ImportError:
        raise ImportError(
            "Gemini Batch API requires the google-genai SDK.\n"
            "Install it with:  pip install google-genai"
        )


# =============================================================================
# Data structures
# =============================================================================

@dataclass
class _BatchRequest:
    """Internal representation of a single batch request."""
    custom_id: str          # "{row_idx}:{indicator1,indicator2,...}"
    system_prompt: str
    user_prompt: str
    row_idx: int
    indicators: List[str]
    metadata: Dict = field(default_factory=dict)  # search_queries, urls_used, etc.


# =============================================================================
# GeminiBatchRunner
# =============================================================================

class GeminiBatchRunner:
    """
    Runs predictions via Gemini Batch API (50 % cost discount).

    Reuses the existing ``Predictor`` for prompt building and result parsing.
    Verification (if configured) runs synchronously after the batch completes.

    Args:
        predictor: Configured Predictor instance (prompt builder + config)
        config: BatchConfig (checkpoint_interval, max_workers, etc.)
        output_path: Path prefix for output files
    """

    def __init__(
        self,
        predictor: Predictor,
        config: Optional[BatchConfig] = None,
        output_path: str = "data/results/batch_results.csv",
    ):
        self.predictor = predictor
        self.config = config or BatchConfig()
        self.output_path = output_path

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run batch predictions on all rows in *df*.

        Returns a DataFrame with prediction columns appended.
        """
        self._validate_input(df)

        print(f"[Batch] Building prompts for {len(df)} rows ...")
        all_requests = self._build_all_requests(df)
        print(f"[Batch] Total requests: {len(all_requests)}")

        # Save requests JSONL for debugging
        self._save_requests_jsonl(all_requests)

        # Split into sub-batches based on checkpoint_interval (in rows)
        # Each row may produce multiple requests (e.g., multiple mode)
        prompts_per_row = self._prompts_per_row()
        chunk_size = self.config.checkpoint_interval * prompts_per_row
        sub_batches = [
            all_requests[i:i + chunk_size]
            for i in range(0, len(all_requests), chunk_size)
        ]
        rows_per_batch = self.config.checkpoint_interval
        print(f"[Batch] Sub-batches: {len(sub_batches)} "
              f"(~{rows_per_batch} rows × {prompts_per_row} prompts = "
              f"{chunk_size} requests per batch)")

        # Submit and poll all sub-batches
        raw_results: Dict[str, str] = {}  # custom_id -> response_text
        for batch_idx, sub_batch in enumerate(sub_batches):
            print(f"\n[Batch] Submitting sub-batch {batch_idx + 1}/{len(sub_batches)} "
                  f"({len(sub_batch)} requests) ...")
            batch_results = self._submit_and_poll(sub_batch)
            raw_results.update(batch_results)

            # Checkpoint
            self._save_checkpoint(df, raw_results, batch_idx + 1)

        # Save responses JSONL for debugging
        self._save_responses_jsonl(raw_results)

        # Parse results into DataFrame
        results_df = self._parse_all_results(df, raw_results, all_requests)

        # Optional: synchronous verification pass
        if self.predictor.config.verify != VerificationType.NONE:
            results_df = self._apply_verification(df, results_df)

        # Save final outputs
        self._save_final(results_df)

        # Cost summary
        self.predictor.cost_tracker.print_summary()

        return results_df

    # ------------------------------------------------------------------
    # Phase 1 — Build requests
    # ------------------------------------------------------------------

    def _build_all_requests(self, df: pd.DataFrame) -> List[_BatchRequest]:
        """Build a batch request for every (row, prompt) pair."""
        requests_out: List[_BatchRequest] = []

        for row_idx in tqdm(range(len(df)), desc="[Batch] Building prompts"):
            row = df.iloc[row_idx]
            polity = str(row.get(COL_TERRITORY_NAME, "Unknown"))
            name = str(row.get(COL_LEADER_NAME, "Unknown"))
            start_year = int(row[COL_START_YEAR])
            end_year = None if pd.isna(row[COL_END_YEAR]) else int(row[COL_END_YEAR])

            prompts = self.predictor.prompt_builder.build(polity, name, start_year, end_year)

            for prompt in prompts:
                custom_id = f"{row_idx}:{','.join(prompt.indicators)}"
                requests_out.append(_BatchRequest(
                    custom_id=custom_id,
                    system_prompt=prompt.system_prompt,
                    user_prompt=prompt.user_prompt,
                    row_idx=row_idx,
                    indicators=prompt.indicators,
                    metadata=prompt.metadata,
                ))

        return requests_out

    # ------------------------------------------------------------------
    # Phase 2 — Submit & poll
    # ------------------------------------------------------------------

    def _submit_and_poll(
        self, batch_requests: List[_BatchRequest]
    ) -> Dict[str, str]:
        """
        Submit a list of requests to the Gemini Batch API and poll until done.

        Uses inline request dicts per the google-genai SDK format.
        Responses are ordered to match the input requests.

        Returns a mapping of custom_id -> response text.
        """
        genai, genai_types = _import_genai()

        api_key = self.predictor.api_keys.get("gemini", "")
        if not api_key:
            raise ValueError("Gemini API key is required for batch mode")

        client = genai.Client(api_key=api_key)

        # Build inline request dicts (google-genai SDK format)
        inline_requests = []
        for req in batch_requests:
            request_dict = {
                'contents': [
                    {'parts': [{'text': req.user_prompt}], 'role': 'user'}
                ],
                'config': {
                    'system_instruction': {
                        'parts': [{'text': req.system_prompt}]
                    },
                    'temperature': self.predictor.config.temperature,
                    'max_output_tokens': self.predictor.config.max_tokens,
                    'top_p': self.predictor.config.top_p,
                },
            }
            inline_requests.append(request_dict)

        # Submit batch job
        model_name = self.predictor.config.model

        batch_job = client.batches.create(
            model=model_name,
            src=inline_requests,
            config={
                'display_name': f"constitution-llm-{int(time.time())}",
            },
        )
        print(f"[Batch] Job submitted: {batch_job.name}")
        state_str = self._get_state_str(batch_job)
        print(f"[Batch] State: {state_str}")

        # Poll for completion
        # The SDK may return JOB_STATE_* or BATCH_STATE_* depending on version
        poll_interval = 30  # seconds

        while True:
            batch_job = client.batches.get(name=batch_job.name)
            state_str = self._get_state_str(batch_job)

            if self._is_terminal_state(state_str):
                break

            print(f"[Batch] Polling ... state={state_str}")
            time.sleep(poll_interval)

        state_str = self._get_state_str(batch_job)
        if not self._is_success_state(state_str):
            print(f"[Batch] WARNING: Job ended with state {state_str}")
            return {}

        # Collect results — inline responses are ordered to match input
        results: Dict[str, str] = {}
        inlined_responses = getattr(
            getattr(batch_job, "dest", None), "inlined_responses", None
        )
        if inlined_responses is None:
            print("[Batch] WARNING: No inlined_responses found in batch job result")
            return {}

        for i, inline_response in enumerate(inlined_responses):
            if i >= len(batch_requests):
                break
            cid = batch_requests[i].custom_id

            # Check for per-request error
            if getattr(inline_response, "error", None):
                print(f"[Batch] Error for {cid}: {inline_response.error}")
                results[cid] = ""
                continue

            try:
                response = inline_response.response
                # Try .text shortcut first, then dig into candidates
                text = ""
                if hasattr(response, "text"):
                    try:
                        text = response.text
                    except (ValueError, AttributeError):
                        pass
                if not text and hasattr(response, "candidates"):
                    try:
                        text = response.candidates[0].content.parts[0].text
                    except (IndexError, AttributeError):
                        pass
                if not text:
                    text = str(response)

                results[cid] = text

                # Track cost (batch discount applied)
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    self.predictor.cost_tracker.add_usage(
                        model=self.predictor.config.model,
                        input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                        output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                        cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
                        thinking_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
                        batch_discount=BATCH_DISCOUNT,
                    )
            except Exception as e:
                print(f"[Batch] Error extracting result for {cid}: {e}")
                results[cid] = ""

        print(f"[Batch] Collected {len(results)}/{len(batch_requests)} results")
        return results

    # ------------------------------------------------------------------
    # Phase 3 — Parse results
    # ------------------------------------------------------------------

    def _parse_all_results(
        self,
        original_df: pd.DataFrame,
        raw_results: Dict[str, str],
        all_requests: List[_BatchRequest],
    ) -> pd.DataFrame:
        """Parse raw batch results into a DataFrame with prediction columns."""
        df = original_df.copy()

        # Group requests by row_idx
        row_requests: Dict[int, List[_BatchRequest]] = {}
        for req in all_requests:
            row_requests.setdefault(req.row_idx, []).append(req)

        for row_idx, requests_for_row in tqdm(
            row_requests.items(), desc="[Batch] Parsing results"
        ):
            # Collect search metadata across all prompts for this row
            row_search_queries: List[str] = []
            row_urls_used: List[str] = []

            for req in requests_for_row:
                # Accumulate search metadata from prompt metadata
                row_search_queries.extend(req.metadata.get('search_queries', []))
                row_urls_used.extend(req.metadata.get('urls_used', []))

                response_text = raw_results.get(req.custom_id, "")
                if not response_text:
                    # Set error values
                    for indicator in req.indicators:
                        df.at[row_idx, f"{indicator}_prediction"] = None
                        if indicator == "constitution" or self.predictor.config.reasoning:
                            df.at[row_idx, f"{indicator}_reasoning"] = "Error: No batch response"
                        df.at[row_idx, f"{indicator}_confidence"] = None
                    continue

                parsed = parse_json_response(response_text, verbose=False)

                for indicator in req.indicators:
                    if indicator == "constitution":
                        validated = validate_constitution_response(parsed)
                        pred = validated.get("constitution")
                        reasoning = validated.get("reasoning", "")
                        confidence = validated.get("confidence_score")
                        df.at[row_idx, "constitution_document_name"] = validated.get("document_name")
                        df.at[row_idx, "constitution_year"] = validated.get("constitution_year")
                    else:
                        valid_labels = INDICATOR_LABELS.get(indicator, ["0", "1"])
                        validated = validate_indicator_response(parsed, indicator, valid_labels)
                        pred = validated.get(indicator)
                        reasoning = validated.get("reasoning", "")
                        confidence = validated.get("confidence_score")

                    df.at[row_idx, f"{indicator}_prediction"] = pred
                    if indicator == "constitution" or self.predictor.config.reasoning:
                        df.at[row_idx, f"{indicator}_reasoning"] = reasoning
                    df.at[row_idx, f"{indicator}_confidence"] = confidence

            # Write search metadata columns for this row
            if row_search_queries:
                df.at[row_idx, "search_queries"] = " | ".join(row_search_queries)
            if row_urls_used:
                df.at[row_idx, "urls_used"] = " | ".join(row_urls_used)

        return df

    # ------------------------------------------------------------------
    # Phase 4 — Optional synchronous verification
    # ------------------------------------------------------------------

    def _apply_verification(
        self, original_df: pd.DataFrame, results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Apply verification synchronously for configured indicators."""
        verify_indicators = self.predictor.config.verify_indicators
        if not verify_indicators:
            return results_df

        print(f"\n[Batch] Running synchronous verification for: {verify_indicators}")

        for row_idx in tqdm(range(len(results_df)), desc="Verifying"):
            row = original_df.iloc[row_idx]
            polity = str(row.get(COL_TERRITORY_NAME, "Unknown"))
            name = str(row.get(COL_LEADER_NAME, "Unknown"))
            start_year = int(row[COL_START_YEAR])
            end_year = None if pd.isna(row[COL_END_YEAR]) else int(row[COL_END_YEAR])

            prompts = self.predictor.prompt_builder.build(polity, name, start_year, end_year)

            for indicator in verify_indicators:
                if indicator not in self.predictor.verifiers:
                    continue

                pred_col = f"{indicator}_prediction"
                if pred_col not in results_df.columns:
                    continue

                prediction = results_df.at[row_idx, pred_col]
                reasoning = results_df.at[row_idx, f"{indicator}_reasoning"] if f"{indicator}_reasoning" in results_df.columns else ""

                # Find the prompt that covers this indicator
                prompt = next((p for p in prompts if indicator in p.indicators), None)
                if prompt is None:
                    continue

                if indicator == "constitution":
                    valid_labels = [1, 0]
                else:
                    valid_labels = INDICATOR_LABELS.get(indicator, ["0", "1"])

                try:
                    verify_result = self.predictor.verifiers[indicator].verify(
                        system_prompt=prompt.system_prompt,
                        user_prompt=prompt.user_prompt,
                        indicator=indicator,
                        valid_labels=valid_labels,
                        initial_prediction=prediction,
                        initial_reasoning=reasoning or "",
                        polity=polity,
                        name=name,
                        start_year=start_year,
                        end_year=end_year,
                    )
                    results_df.at[row_idx, f"{indicator}_verified"] = verify_result.verified_prediction
                    results_df.at[row_idx, f"{indicator}_verification"] = str(verify_result.verification_details)
                except Exception as e:
                    results_df.at[row_idx, f"{indicator}_verification"] = f"Error: {e}"

            time.sleep(self.config.delay_between_calls)

        return results_df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _prompts_per_row(self) -> int:
        """Estimate number of prompts generated per row."""
        from config import PromptMode
        if self.predictor.config.mode in (PromptMode.SINGLE, PromptMode.SEQUENTIAL):
            return 1
        return len(self.predictor.config.indicators)

    @staticmethod
    def _get_state_str(batch_job) -> str:
        """Extract state as a string, handling both enum and raw string formats."""
        state = batch_job.state
        if hasattr(state, "name"):
            return state.name
        return str(state)

    @staticmethod
    def _is_terminal_state(state_str: str) -> bool:
        """Check if a state string indicates the job is done (any format)."""
        s = state_str.upper()
        return any(keyword in s for keyword in ("SUCCEEDED", "FAILED", "CANCELLED", "EXPIRED"))

    @staticmethod
    def _is_success_state(state_str: str) -> bool:
        """Check if a state string indicates success (any format)."""
        return "SUCCEEDED" in state_str.upper()

    def _save_requests_jsonl(self, all_requests: List[_BatchRequest]) -> None:
        """Save all batch requests to JSONL for debugging."""
        stem = Path(self.output_path).stem
        out_dir = Path(self.output_path).parent
        path = out_dir / f"{stem}_batch_requests.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for req in all_requests:
                record = {
                    "custom_id": req.custom_id,
                    "row_idx": req.row_idx,
                    "indicators": req.indicators,
                    "system_prompt": req.system_prompt,
                    "user_prompt": req.user_prompt,
                    "metadata": req.metadata,
                }
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        print(f"[Batch] Saved requests JSONL: {path}")

    def _save_responses_jsonl(self, raw_results: Dict[str, str]) -> None:
        """Save all batch responses to JSONL for debugging."""
        stem = Path(self.output_path).stem
        out_dir = Path(self.output_path).parent
        path = out_dir / f"{stem}_batch_responses.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for custom_id, response_text in raw_results.items():
                record = {
                    "custom_id": custom_id,
                    "response": response_text,
                }
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        print(f"[Batch] Saved responses JSONL: {path}")

    def _save_checkpoint(
        self, df: pd.DataFrame, raw_results: Dict[str, str], batch_num: int
    ) -> None:
        checkpoint_path = f"{self.output_path}.batch_checkpoint_{batch_num}.json"
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, ensure_ascii=False, indent=2, default=str)
        print(f"[Batch] Checkpoint saved: {checkpoint_path}")

    def _save_final(self, results_df: pd.DataFrame) -> None:
        # CSV
        results_df.to_csv(self.output_path, index=False)
        print(f"[Batch] Saved CSV: {self.output_path}")

        # JSON
        json_path = self.output_path.replace(".csv", ".json")
        records = results_df.to_dict(orient="records")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2, default=str)
        print(f"[Batch] Saved JSON: {json_path}")

        # Cost report
        logs_dir = Path("data/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        cost_path = logs_dir / f"{Path(self.output_path).stem}_costs.json"
        self.predictor.cost_tracker.save_report(str(cost_path))
        print(f"[Batch] Saved cost report: {cost_path}")

        # Clean up checkpoints
        for ckpt in Path(self.output_path).parent.glob(
            f"{Path(self.output_path).stem}.batch_checkpoint_*.json"
        ):
            ckpt.unlink()
            print(f"[Batch] Removed checkpoint: {ckpt}")
