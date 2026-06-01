"""
Downstream Elections Classifier
=================================

Post-processing script that runs AFTER the main pipeline. Depends on the
``assembly_prediction`` column (0/1/2) produced by the main pipeline.

Dependency
----------
Assembly uses a 4-level schema (0/1/2/3):
  0 = None
  1 = Council (small advisory)
  2 = Legislature (large representative body) → LLM call for elections
  3 = Popular assembly (all citizens or chosen by lot)

Only assembly = 2 (Legislature) can have legislative elections.
  - assembly = 0, 1, or 3 → pass-through with elections_prediction = "0" (no API call)
  - assembly = 2 → LLM call to determine 0/1/2

Classifier
----------

**ElectionsClassifier** (``--task elections``)
  For polities where a Legislature exists (assembly = 2), codes whether
  assembly members are elected and whether elections are contested by organized
  factions or parties.

  Label meanings:
    0  No legislature / members not elected (also: assembly = 0, 1, or 3)
    1  Members elected, no organized factions/parties
    2  Competitive elections (organized factions/parties)

  Output columns added:
    elections_prediction      : "0", "1", or "2"
    elections_confidence      : integer 1-100 (null for pass-through rows)
    elections_reasoning       : reasoning text  (null for pass-through rows)
    elections_search_queries  : pipe-delimited search queries (search mode only)
    elections_urls_used       : pipe-delimited URLs with source tags (search mode only)

Usage
-----
  python pipeline/post_processing.py \\
      --input  data/results/predictions.csv \\
      --output data/results/predictions_extended.csv \\
      --model  gemini-2.5-pro \\
      --parallel-rows 4

The original columns are never modified.
"""

import argparse
import os
import sys
import time
from collections import Counter

# Ensure project root is on sys.path when run as `python pipeline/post_processing.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from models.llm_clients import create_llm
from utils.json_parser import parse_json_response
from config import SearchMode

load_dotenv()


# =============================================================================
# ELECTIONS PROMPT
# =============================================================================

ELECTIONS_SYSTEM_PROMPT = """You are a professional political scientist and historian specializing in \
electoral systems and representative institutions across different historical periods.

A LARGE ASSEMBLY (Type 2) is KNOWN TO EXIST in the polity below. Your task is to determine whether \
members of that assembly were elected to their positions, and—if so—whether those \
elections were contested by organized factions or parties.

## Definition of an Election

An election is a selection procedure in which:
- Members are chosen by an electorate through defined rules (e.g., majority, proportionality)
  that translate votes into seats.
- The electorate must be considerably larger than the body itself (though it may be far short
  of universal suffrage).

## Election Categories

**No Elections (0):**
- Assembly members are NOT elected.
- Members hold seats through appointment, hereditary right, ex-officio status, or
  selection by a small ruling group rather than a broader electorate.
- Examples: appointed councils, hereditary nobility in a legislature.

**Elections Exist (1):**
- Most members of the assembly are elected.
- Elections follow defined rules translating votes into seats.
- The electorate is larger than the body itself.
- Elections are NOT (or not consistently) contested by organized factions or parties.

**Competitive Elections (2):**
- Elections exist AND are contested by organized factions or parties.
- Distinct blocs, factions, or parties compete for seats.
- Examples: Roman Optimates vs Populares competing for magistracies, English Whigs vs
  Tories competing for parliamentary seats, modern multi-party elections.

## Important Notes

- The extent of suffrage (who can vote) is NOT relevant for coding.
- Focus on whether MOST members are elected, not necessarily all.
- The key distinction for code 2 is organized factions/parties, not just informal competition.
- Code based on de facto (actual) practice, not de jure (formal) rules.

## Output Requirements

Provide a JSON object with exactly these fields:
- "elections": Must be exactly "0", "1", or "2" (string)
- "reasoning": Your step-by-step reasoning (string)
- "confidence_score": Integer from 1 to 100

**CRITICAL OUTPUT FORMAT:**
- Respond with ONLY a JSON object
- Do NOT include markdown code fences (```json)
- Do NOT include any text before or after the JSON
- Your response must start with { and end with }
"""

ELECTIONS_USER_PROMPT_TEMPLATE = """A LARGE ASSEMBLY (Type 2) is KNOWN TO EXIST in this polity. Determine how \
assembly members obtained their positions.

**Polity:** {polity}
**Leader:** {name}
**Reign Period:** {start_year}-{end_year}

Determine the elections category:
- 0: Members are NOT elected (appointed, hereditary, or selected by small ruling group)
- 1: Members ARE elected through defined rules, but elections are NOT organized by factions/parties
- 2: Members ARE elected AND elections are contested by organized factions or parties

**IMPORTANT:** Focus on the SELECTION METHOD for assembly members during THIS LEADER'S REIGN.

Respond with a single JSON object:
{{"elections": "0, 1, or 2", "reasoning": "your analysis", "confidence_score": 1-100}}
"""


# =============================================================================
# ELECTIONS CLASSIFIER
# =============================================================================

class ElectionsClassifier:
    """
    Classifies whether large-assembly members are elected, and if so, whether
    elections are contested by organized factions or parties.

    Rows where assembly_prediction != "2" are passed through without
    an API call (elections_prediction = "0").
    assembly = 0: no assembly at all
    assembly = 1: small advisory council only — elections not applicable
    assembly = 2: Legislature (large representative body) → LLM call to determine 0/1/2
    assembly = 3: popular assembly — not a legislature, so elections = 0 (pass-through)
    """

    PRED_COL = "elections_prediction"
    CONF_COL = "elections_confidence"
    REASON_COL = "elections_reasoning"
    LOGPROB_COL = "elections_logprob"
    SEARCH_QUERIES_COL = "elections_search_queries"
    URLS_USED_COL = "elections_urls_used"

    def __init__(
        self,
        model: str,
        api_keys: Dict[str, str],
        assembly_col: str = "assembly_prediction",
        max_workers: int = 1,
        delay: float = 1.0,
        search_mode: SearchMode = SearchMode.NONE,
        serper_api_key: str = "",
        use_logprobs: bool = False,
        n_samples: int = 0,
        sc_temperatures: Optional[List[float]] = None,
    ):
        self.model = model
        self.use_logprobs = use_logprobs
        self.llm = create_llm(model, api_keys, use_logprobs=use_logprobs)
        self.n_samples = n_samples
        # Default SC temperatures: n_samples draws at 0.7
        self.sc_temperatures = sc_temperatures or ([0.7] * n_samples if n_samples > 0 else [])
        self.api_keys = api_keys
        self.assembly_col = assembly_col
        self.max_workers = max_workers
        self.delay = delay
        self.search_mode = search_mode
        self.serper_api_key = serper_api_key
        self.pre_searcher = None
        if search_mode == SearchMode.FORCED:
            from pipeline.pre_search import PreSearcher
            self.pre_searcher = PreSearcher(serper_api_key=serper_api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run classification and return df with new columns added.

        Rows with assembly != 2 receive elections_prediction = "0" (pass-through).
        Rows with assembly == 2 receive an LLM call.
        """
        df = df.copy()

        df[self.PRED_COL] = None
        df[self.CONF_COL] = None
        df[self.REASON_COL] = None
        if self.use_logprobs:
            df[self.LOGPROB_COL] = None
        if self.search_mode != SearchMode.NONE:
            df[self.SEARCH_QUERIES_COL] = None
            df[self.URLS_USED_COL] = None

        # Pass-through rows where assembly != 2 (no large assembly)
        mask_large_assembly = pd.to_numeric(df[self.assembly_col], errors="coerce") == 2
        df.loc[~mask_large_assembly, self.PRED_COL] = "0"

        rows_to_classify = df[mask_large_assembly].copy()
        total = len(rows_to_classify)

        if total == 0:
            print("No rows with assembly=2 found. Nothing to classify.")
            return df

        print(f"Classifying elections for {total} rows with assembly=2 (workers={self.max_workers})...")

        if self.max_workers > 1:
            results = self._classify_parallel(rows_to_classify)
        else:
            results = self._classify_sequential(rows_to_classify)

        for orig_idx, pred, conf, reason, logprob, queries_str, urls_str in results:
            df.at[orig_idx, self.PRED_COL] = pred
            df.at[orig_idx, self.CONF_COL] = conf
            df.at[orig_idx, self.REASON_COL] = reason
            if self.use_logprobs:
                df.at[orig_idx, self.LOGPROB_COL] = logprob
            if self.search_mode != SearchMode.NONE:
                df.at[orig_idx, self.SEARCH_QUERIES_COL] = queries_str or None
                df.at[orig_idx, self.URLS_USED_COL] = urls_str or None

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, row: pd.Series) -> Tuple[str, Optional[int], str, Optional[float], str, str]:
        """Call LLM with optional self-consistency majority voting.

        Returns:
            (prediction, confidence_score, reasoning, logprob, search_queries_str, urls_used_str)

        When n_samples == 0 (default), makes a single call (no SC).
        When n_samples > 0, makes 1 main call + n_samples additional SC calls, then
        returns the majority prediction with agreement*100 as the confidence score.
        Total API calls = n_samples + 1.
        """
        pred, conf, reason, logprob, queries_str, urls_str = self._call_llm_once(row)

        if self.n_samples == 0:
            return pred, conf, reason, logprob, queries_str, urls_str

        # Collect majority-vote predictions across n_samples additional calls
        predictions = [pred] if pred in ('0', '1', '2') else []
        for i, temp in enumerate(self.sc_temperatures[:self.n_samples]):
            try:
                sc_pred, _, _, _, _, _ = self._call_llm_once(row, temperature=temp)
                if sc_pred in ('0', '1', '2'):
                    predictions.append(sc_pred)
            except Exception as e:
                from tqdm import tqdm as _tqdm
                _tqdm.write(f"WARN: elections SC sample {i + 1} failed: {e}")

        if not predictions:
            return pred, conf, reason, logprob, queries_str, urls_str

        counter = Counter(predictions)
        majority_pred, majority_count = counter.most_common(1)[0]
        agreement = majority_count / len(predictions)
        return majority_pred, round(agreement * 100), reason, logprob, queries_str, urls_str

    def _call_llm_once(self, row: pd.Series, temperature: float = 0.0) -> Tuple[str, Optional[int], str, Optional[float], str, str]:
        """Single LLM call (no SC). Used directly by _call_llm().

        Returns:
            (prediction, confidence_score, reasoning, logprob, search_queries_str, urls_used_str)
        """
        polity = str(
            row.get("territorynamehistorical")
            or row.get("polity_name")
            or "Unknown Polity"
        )
        name = str(
            row.get("name")
            or row.get("leader_name")
            or "Unknown Leader"
        )
        start_year = (
            row.get("start_year")
            or row.get("entrydateyear")
            or row.get("leader_first_year")
            or "?"
        )
        end_year = (
            row.get("end_year")
            or row.get("exitdateyear")
            or row.get("leader_last_year")
            or "?"
        )

        user_prompt = ELECTIONS_USER_PROMPT_TEMPLATE.format(
            polity=polity,
            name=name,
            start_year=start_year,
            end_year=end_year,
        )

        queries_str = ""
        urls_str = ""

        if self.pre_searcher is not None:
            try:
                sy = int(start_year) if start_year != "?" else 0
                ey = int(end_year) if end_year != "?" else None
            except (ValueError, TypeError):
                sy, ey = 0, None
            search_result = self.pre_searcher.search(polity, name, sy, ey)
            user_prompt = self.pre_searcher.enrich_prompt(user_prompt, search_result)
            queries_str = " | ".join(search_result.search_queries)
            urls_str = " | ".join(search_result.urls_used)

        response = None  # will be set only for non-agentic calls (needed for logprobs)

        if self.search_mode == SearchMode.AGENTIC:
            from models.llm_clients import detect_provider
            from models.search_agents import (
                run_openai_search_agent, run_gemini_search_agent,
                run_bedrock_search_agent, run_anthropic_search_agent,
            )
            provider = detect_provider(self.model)
            agent_kwargs = dict(
                system_prompt=ELECTIONS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                serper_api_key=self.serper_api_key,
                force_search=False,
            )
            if provider == 'openai':
                content = run_openai_search_agent(model=self.model, api_key=self.api_keys.get('openai', ''), **agent_kwargs)
            elif provider == 'gemini':
                content = run_gemini_search_agent(model=self.model, api_key=self.api_keys.get('gemini', ''), **agent_kwargs)
            elif provider == 'bedrock':
                content = run_bedrock_search_agent(model=self.model, api_keys=self.api_keys, **agent_kwargs)
            elif provider == 'anthropic':
                content = run_anthropic_search_agent(model=self.model, api_key=self.api_keys.get('anthropic', ''), **agent_kwargs)
            else:
                content = run_openai_search_agent(model=self.model, api_key=self.api_keys.get('openai', ''), **agent_kwargs)
            parsed = parse_json_response(content or '', verbose=False)
        else:
            response = self.llm.call(
                system_prompt=ELECTIONS_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=temperature,
            )
            parsed = parse_json_response(response.content, verbose=False)

        pred = str(parsed.get("elections", "0"))
        if pred not in ("0", "1", "2"):
            pred = "0"
        conf = parsed.get("confidence_score")
        reason = parsed.get("reasoning", "")

        # Extract logprob of the 'elections' value token (Gemini only)
        logprob = None
        if hasattr(response, 'logprobs_result') and response.logprobs_result is not None:
            try:
                from utils.logprob_utils import extract_indicator_logprobs
                lp_map = extract_indicator_logprobs(
                    logprobs_result=response.logprobs_result,
                    json_text=response.content,
                    indicator_valid_labels={'elections': ['0', '1', '2']},
                )
                logprob = lp_map.get('elections')
            except Exception:
                pass

        return pred, conf, reason, logprob, queries_str, urls_str

    def _classify_sequential(
        self, rows: pd.DataFrame
    ) -> List[Tuple[int, str, Optional[int], str, Optional[float], str, str]]:
        results = []
        for orig_idx, row in tqdm(rows.iterrows(), total=len(rows), desc="Classifying elections"):
            try:
                pred, conf, reason, logprob, queries_str, urls_str = self._call_llm(row)
            except Exception as e:
                print(f"\nError on row {orig_idx}: {e}")
                pred, conf, reason, logprob, queries_str, urls_str = "0", None, f"Error: {e}", None, "", ""
            results.append((orig_idx, pred, conf, reason, logprob, queries_str, urls_str))
            time.sleep(self.delay)
        return results

    def _classify_parallel(
        self, rows: pd.DataFrame
    ) -> List[Tuple[int, str, Optional[int], str, Optional[float], str, str]]:
        all_results: List[Tuple[int, str, Optional[int], str, Optional[float], str, str]] = []
        indices = list(rows.index)
        windows = [indices[i:i + self.max_workers] for i in range(0, len(indices), self.max_workers)]
        lock = Lock()

        with tqdm(total=len(indices), desc="Classifying elections") as pbar:
            for window in windows:
                window_results: List[Tuple[int, str, Optional[int], str, Optional[float], str, str]] = []

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_idx = {
                        executor.submit(self._call_llm, rows.loc[orig_idx]): orig_idx
                        for orig_idx in window
                    }
                    for future in as_completed(future_to_idx):
                        orig_idx = future_to_idx[future]
                        try:
                            pred, conf, reason, logprob, queries_str, urls_str = future.result()
                        except Exception as e:
                            print(f"\nError on row {orig_idx}: {e}")
                            pred, conf, reason, logprob, queries_str, urls_str = "0", None, f"Error: {e}", None, "", ""
                        window_results.append((orig_idx, pred, conf, reason, logprob, queries_str, urls_str))

                window_results.sort(key=lambda x: x[0])
                all_results.extend(window_results)
                pbar.update(len(window))

                if window != windows[-1]:
                    time.sleep(self.delay)

        return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Downstream Elections Classifier\n"
            "\n"
            "Post-processing script that runs AFTER the main pipeline.\n"
            "Depends on the assembly_prediction column (0/1/2).\n"
            "\n"
            "elections:\n"
            "  assembly=0,1,3 → elections=0 (pass-through, no LLM call)\n"
            "  assembly=2 (Legislature) → LLM call\n"
            "    0  Members not elected   1  Elected, no factions  2  Competitive elections\n"
            "\n"
            "Must be run AFTER the main pipeline has produced a predictions file\n"
            "that contains an assembly_prediction column."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline/post_processing.py \\
      --input  data/results/predictions.csv \\
      --output data/results/predictions_extended.csv

  # Use a different model, 4 rows in parallel
  python pipeline/post_processing.py \\
      --input  data/results/predictions.csv \\
      --output data/results/predictions_extended.csv \\
      --model  gpt-4o --parallel-rows 4
        """
    )

    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to predictions file (CSV or JSONL) from the main pipeline"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to write the extended predictions CSV"
    )
    parser.add_argument(
        "--model", "-m",
        default="gemini-3.1-pro-preview",
        help="LLM model identifier (default: gemini-3.1-pro-preview)"
    )
    parser.add_argument(
        "--assembly-col",
        default="assembly_prediction",
        help="Column name holding assembly predictions (default: assembly_prediction)"
    )
    parser.add_argument(
        "--parallel-rows",
        type=int,
        default=1,
        metavar="N",
        help="Number of rows to process in parallel (default: 1 = sequential)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between API calls / between parallel windows (default: 1.0)"
    )
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N rows (for testing)"
    )
    parser.add_argument(
        "--search-mode",
        choices=["none", "agentic", "forced"],
        default="none",
        help=(
            "Search mode (default: none).\n"
            "  none    — Pure LLM output.\n"
            "  agentic — LLM decides whether to search.\n"
            "  forced  — Always search before LLM answers (Wikipedia/DuckDuckGo/Serper)."
        )
    )
    parser.add_argument(
        "--logprobs",
        action="store_true",
        default=False,
        help=(
            "Request token-level log probabilities from Gemini for uncertainty quantification.\n"
            "Adds elections_logprob column to output.\n"
            "Supported models: gemini-2.5-flash, gemini-2.5-pro (default: off)."
        )
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Number of additional self-consistency samples for elections (default: 0 = single call).\n"
            "With N>0: 1 main call + N SC calls = N+1 total votes; majority prediction wins.\n"
            "Example: --n-samples 2 → 3 total votes."
        )
    )

    args = parser.parse_args()

    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_session_token": os.getenv("AWS_SESSION_TOKEN"),
    }

    print(f"Loading predictions from: {args.input}")
    from utils.data_loader import load_dataframe
    df = load_dataframe(args.input)

    if args.assembly_col not in df.columns:
        raise ValueError(
            f"Column '{args.assembly_col}' not found in input file. "
            f"Available columns: {list(df.columns)}"
        )

    if args.test:
        df = df.head(args.test)
        print(f"Test mode: processing only first {args.test} rows")

    assembly_counts = df[args.assembly_col].astype(str).value_counts().sort_index()
    print(f"Loaded {len(df)} rows. Assembly distribution:")
    for label, count in assembly_counts.items():
        print(f"  assembly={label}: {count}")

    search_mode = SearchMode(args.search_mode)

    if search_mode == SearchMode.AGENTIC and not os.getenv("SERPER_API_KEY"):
        raise ValueError("Agentic search requires SERPER_API_KEY environment variable.")

    classifier = ElectionsClassifier(
        model=args.model,
        api_keys=api_keys,
        assembly_col=args.assembly_col,
        max_workers=args.parallel_rows,
        delay=args.delay,
        search_mode=search_mode,
        serper_api_key=os.getenv("SERPER_API_KEY", ""),
        use_logprobs=args.logprobs,
        n_samples=args.n_samples,
    )

    result_df = classifier.classify(df)

    pred_counts = result_df[ElectionsClassifier.PRED_COL].value_counts().sort_index()
    print("\nelections_prediction distribution:")
    for label, count in pred_counts.items():
        print(f"  {label}: {count}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"\nSaved predictions to: {args.output}")


if __name__ == "__main__":
    main()
