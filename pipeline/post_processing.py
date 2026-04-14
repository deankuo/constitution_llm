"""
Downstream Assembly Classifiers
=================================

Post-processing scripts that run AFTER the main pipeline.  Both classifiers
depend on the binary ``assembly_prediction`` column produced by the main
pipeline and must be run in order if you need both.

Classifiers
-----------

**AssemblyExtendedClassifier** (``--task assembly_extended``)
  Extends binary assembly predictions (0/1) to a three-label scheme (0/1/2).

  Label meanings:
    0  No assembly exists              (pass-through, no API call)
    1  Assembly exists, no competitive factions or parties
    2  Assembly exists WITH competitive factions or parties

  Output columns added:
    assembly_extended_prediction  : "0", "1", or "2"
    assembly_extended_confidence  : integer 1-100 (null for pass-through rows)
    assembly_extended_reasoning   : reasoning text  (null for pass-through rows)

**ElectionsClassifier** (``--task elections``)
  For polities where an assembly exists, codes whether assembly members are
  elected and whether elections are contested by organized factions/parties.

  Label meanings:
    0  No assembly / members not elected   (pass-through when assembly=0;
                                            LLM call when assembly=1 → 0)
    1  Members elected, no organized factions/parties
    2  Members elected via competitive elections (organized factions/parties)

  Output columns added:
    elections_prediction  : "0", "1", or "2"
    elections_confidence  : integer 1-100 (null for pass-through rows)
    elections_reasoning   : reasoning text  (null for pass-through rows)

Usage
-----
  # Run assembly_extended only (default)
  python pipeline/post_processing.py \\
      --input  data/results/predictions.csv \\
      --output data/results/predictions_extended.csv \\
      --task   assembly_extended

  # Run elections only
  python pipeline/post_processing.py \\
      --input  data/results/predictions.csv \\
      --output data/results/predictions_extended.csv \\
      --task   elections

  # Run both classifiers in sequence (assembly_extended first, then elections)
  python pipeline/post_processing.py \\
      --input  data/results/predictions.csv \\
      --output data/results/predictions_extended.csv \\
      --task   all \\
      --model  gemini-2.5-pro \\
      --parallel-rows 4          # optional: process N rows concurrently

The original columns are never modified.
"""

import argparse
import os
import sys
import time

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
# PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a professional political scientist and historian specializing in \
legislative institutions across different historical periods.

Your task is to further classify an assembly that is KNOWN TO EXIST in a given polity. \
You must determine whether this assembly featured competitive factions or organized political parties.

## Definition of Competitive Factions or Parties (Label 2)

An assembly is coded as having **competitive factions or parties (2)** if:
- Organized groups or factions competed for influence within the assembly
- Distinct political blocs, parties, or factions held seats and competed for policy outcomes
- Examples: Roman political factions (Optimates vs Populares), English Whigs and Tories, \
modern parliamentary parties, organized voting blocs in a legislature

An assembly is coded as **existing without competitive factions (1)** if:
- The assembly exists and functions (role in selection/taxation/policy, independent, regular)
- But no organized competing factions or parties are present
- Members vote as individuals rather than as part of organized blocs

## Output Requirements

Provide a JSON object with exactly these fields:
- "assembly_extended": Must be exactly "1" or "2" (string)
- "reasoning": Your step-by-step reasoning (string)
- "confidence_score": Integer from 1 to 100

**CRITICAL OUTPUT FORMAT:**
- Respond with ONLY a JSON object
- Do NOT include markdown code fences (```json)
- Do NOT include any text before or after the JSON
- Your response must start with { and end with }
"""

USER_PROMPT_TEMPLATE = """An assembly is KNOWN TO EXIST in this polity. Determine whether it \
featured competitive factions or organized political parties.

**Polity:** {polity}
**Leader:** {name}
**Reign Period:** {start_year}-{end_year}

Determine whether this assembly had competitive factions or parties:
- 2: Assembly exists AND has competitive factions or organized parties
- 1: Assembly exists but NO competitive factions or organized parties

**IMPORTANT:** The assembly is confirmed to exist. Only classify whether competitive \
factions/parties are present (→ 2) or absent (→ 1). Focus on THIS LEADER'S REIGN.

Respond with a single JSON object:
{{"assembly_extended": "1 or 2", "reasoning": "your analysis", "confidence_score": 1-100}}
"""


# =============================================================================
# CLASSIFIER
# =============================================================================

class AssemblyExtendedClassifier:
    """
    Extends binary assembly predictions to a three-label scheme.

    Rows where assembly_prediction == "0" are passed through without
    an API call (assembly_extended_prediction = "0").

    Rows where assembly_prediction == "1" receive a second LLM call
    to determine label 1 (no competitive factions) or 2 (with factions).
    """

    # Column names for the prediction output
    PRED_COL = "assembly_extended_prediction"
    CONF_COL = "assembly_extended_confidence"
    REASON_COL = "assembly_extended_reasoning"

    def __init__(
        self,
        model: str,
        api_keys: Dict[str, str],
        assembly_col: str = "assembly_prediction",
        max_workers: int = 1,
        delay: float = 1.0,
        search_mode: SearchMode = SearchMode.NONE,
        serper_api_key: str = "",
    ):
        """
        Args:
            model:         LLM model identifier (e.g. "gemini-2.5-pro")
            api_keys:      API key dictionary
            assembly_col:  Name of the column holding binary assembly predictions
            max_workers:   Number of rows to process concurrently (1 = sequential)
            delay:         Seconds between sequential calls / between parallel windows
            search_mode:   Search mode (none, agentic, forced)
            serper_api_key: Serper API key (for agentic/forced search)
        """
        self.model = model
        self.llm = create_llm(model, api_keys)
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
        Run classification and return df with three new columns added.

        Args:
            df: DataFrame with at least `assembly_col`, territorynamehistorical,
                name, start_year, end_year columns.

        Returns:
            Copy of df with assembly_extended_prediction/confidence/reasoning added.
        """
        df = df.copy()

        # Initialise output columns
        df[self.PRED_COL] = None
        df[self.CONF_COL] = None
        df[self.REASON_COL] = None

        # Pass-through rows where assembly == 0 (handles "0", "0.0", 0, 0.0)
        mask_no_assembly = pd.to_numeric(df[self.assembly_col], errors="coerce") == 0
        df.loc[mask_no_assembly, self.PRED_COL] = "0"

        # Rows that need an LLM call
        rows_to_classify = df[~mask_no_assembly].copy()
        total = len(rows_to_classify)

        if total == 0:
            print("No rows with assembly=1 found. Nothing to classify.")
            return df

        print(f"Classifying {total} rows with assembly=1 (workers={self.max_workers})...")

        if self.max_workers > 1:
            results = self._classify_parallel(rows_to_classify)
        else:
            results = self._classify_sequential(rows_to_classify)

        for orig_idx, pred, conf, reason in results:
            df.at[orig_idx, self.PRED_COL] = pred
            df.at[orig_idx, self.CONF_COL] = conf
            df.at[orig_idx, self.REASON_COL] = reason

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, row: pd.Series) -> Tuple[str, Optional[int], str]:
        """
        Make a single LLM call for one row.

        Returns:
            (prediction, confidence_score, reasoning)
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

        user_prompt = USER_PROMPT_TEMPLATE.format(
            polity=polity,
            name=name,
            start_year=start_year,
            end_year=end_year,
        )

        # Enrich prompt with pre-search context if forced search mode
        if self.pre_searcher is not None:
            try:
                sy = int(start_year) if start_year != "?" else 0
                ey = int(end_year) if end_year != "?" else None
            except (ValueError, TypeError):
                sy, ey = 0, None
            search_result = self.pre_searcher.search(polity, name, sy, ey)
            user_prompt = self.pre_searcher.enrich_prompt(user_prompt, search_result)

        # Agentic search: route through search agent
        if self.search_mode == SearchMode.AGENTIC:
            from models.llm_clients import detect_provider
            from models.search_agents import (
                run_openai_search_agent, run_gemini_search_agent,
                run_bedrock_search_agent, run_anthropic_search_agent,
            )
            provider = detect_provider(self.model)
            agent_kwargs = dict(
                system_prompt=SYSTEM_PROMPT,
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
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.0,
            )
            parsed = parse_json_response(response.content, verbose=False)

        pred = str(parsed.get("assembly_extended", "1"))
        # Clamp to valid labels
        if pred not in ("1", "2"):
            pred = "1"
        conf = parsed.get("confidence_score")
        reason = parsed.get("reasoning", "")
        return pred, conf, reason

    def _classify_sequential(
        self, rows: pd.DataFrame
    ) -> List[Tuple[int, str, Optional[int], str]]:
        """Classify rows one at a time."""
        results = []
        for orig_idx, row in tqdm(rows.iterrows(), total=len(rows), desc="Classifying"):
            try:
                pred, conf, reason = self._call_llm(row)
            except Exception as e:
                print(f"\nError on row {orig_idx}: {e}")
                pred, conf, reason = "1", None, f"Error: {e}"
            results.append((orig_idx, pred, conf, reason))
            time.sleep(self.delay)
        return results

    def _classify_parallel(
        self, rows: pd.DataFrame
    ) -> List[Tuple[int, str, Optional[int], str]]:
        """Classify rows in parallel windows of max_workers size."""
        all_results: List[Tuple[int, str, Optional[int], str]] = []
        indices = list(rows.index)
        windows = [indices[i:i + self.max_workers] for i in range(0, len(indices), self.max_workers)]
        lock = Lock()

        with tqdm(total=len(indices), desc="Classifying") as pbar:
            for window in windows:
                window_results: List[Tuple[int, str, Optional[int], str]] = []

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_idx = {
                        executor.submit(self._call_llm, rows.loc[orig_idx]): orig_idx
                        for orig_idx in window
                    }
                    for future in as_completed(future_to_idx):
                        orig_idx = future_to_idx[future]
                        try:
                            pred, conf, reason = future.result()
                        except Exception as e:
                            print(f"\nError on row {orig_idx}: {e}")
                            pred, conf, reason = "1", None, f"Error: {e}"
                        window_results.append((orig_idx, pred, conf, reason))

                # Sort by original index to preserve order
                window_results.sort(key=lambda x: x[0])
                all_results.extend(window_results)
                pbar.update(len(window))

                if window != windows[-1]:
                    time.sleep(self.delay)

        return all_results


# =============================================================================
# ELECTIONS PROMPT
# =============================================================================

ELECTIONS_SYSTEM_PROMPT = """You are a professional political scientist and historian specializing in \
electoral systems and representative institutions across different historical periods.

An assembly is KNOWN TO EXIST in the polity below. Your task is to determine whether \
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

ELECTIONS_USER_PROMPT_TEMPLATE = """An assembly is KNOWN TO EXIST in this polity. Determine how \
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
    Classifies whether assembly members are elected, and if so, whether
    elections are contested by organized factions or parties.

    Rows where assembly_prediction == "0" are passed through without
    an API call (elections_prediction = "0").

    Rows where assembly_prediction == "1" receive an LLM call to
    determine label:
      0 — members not elected (appointed, hereditary, etc.)
      1 — members elected, no organized factions/parties
      2 — members elected via competitive elections (factions/parties present)
    """

    PRED_COL = "elections_prediction"
    CONF_COL = "elections_confidence"
    REASON_COL = "elections_reasoning"

    def __init__(
        self,
        model: str,
        api_keys: Dict[str, str],
        assembly_col: str = "assembly_prediction",
        max_workers: int = 1,
        delay: float = 1.0,
        search_mode: SearchMode = SearchMode.NONE,
        serper_api_key: str = "",
    ):
        self.model = model
        self.llm = create_llm(model, api_keys)
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
        Run classification and return df with three new columns added.

        Args:
            df: DataFrame with at least `assembly_col`, territorynamehistorical,
                name, start_year, end_year columns.

        Returns:
            Copy of df with elections_prediction/confidence/reasoning added.
        """
        df = df.copy()

        df[self.PRED_COL] = None
        df[self.CONF_COL] = None
        df[self.REASON_COL] = None

        # Rows without an assembly → elections = 0 (pass-through, no LLM call)
        # Handles "0", "0.0", 0, 0.0 from CSV float serialisation
        mask_no_assembly = pd.to_numeric(df[self.assembly_col], errors="coerce") == 0
        df.loc[mask_no_assembly, self.PRED_COL] = "0"

        rows_to_classify = df[~mask_no_assembly].copy()
        total = len(rows_to_classify)

        if total == 0:
            print("No rows with assembly=1 found. Nothing to classify.")
            return df

        print(f"Classifying elections for {total} rows with assembly=1 (workers={self.max_workers})...")

        if self.max_workers > 1:
            results = self._classify_parallel(rows_to_classify)
        else:
            results = self._classify_sequential(rows_to_classify)

        for orig_idx, pred, conf, reason in results:
            df.at[orig_idx, self.PRED_COL] = pred
            df.at[orig_idx, self.CONF_COL] = conf
            df.at[orig_idx, self.REASON_COL] = reason

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call_llm(self, row: pd.Series) -> Tuple[str, Optional[int], str]:
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

        if self.pre_searcher is not None:
            try:
                sy = int(start_year) if start_year != "?" else 0
                ey = int(end_year) if end_year != "?" else None
            except (ValueError, TypeError):
                sy, ey = 0, None
            search_result = self.pre_searcher.search(polity, name, sy, ey)
            user_prompt = self.pre_searcher.enrich_prompt(user_prompt, search_result)

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
                temperature=0.0,
            )
            parsed = parse_json_response(response.content, verbose=False)

        pred = str(parsed.get("elections", "0"))
        if pred not in ("0", "1", "2"):
            pred = "0"
        conf = parsed.get("confidence_score")
        reason = parsed.get("reasoning", "")
        return pred, conf, reason

    def _classify_sequential(
        self, rows: pd.DataFrame
    ) -> List[Tuple[int, str, Optional[int], str]]:
        results = []
        for orig_idx, row in tqdm(rows.iterrows(), total=len(rows), desc="Classifying elections"):
            try:
                pred, conf, reason = self._call_llm(row)
            except Exception as e:
                print(f"\nError on row {orig_idx}: {e}")
                pred, conf, reason = "0", None, f"Error: {e}"
            results.append((orig_idx, pred, conf, reason))
            time.sleep(self.delay)
        return results

    def _classify_parallel(
        self, rows: pd.DataFrame
    ) -> List[Tuple[int, str, Optional[int], str]]:
        all_results: List[Tuple[int, str, Optional[int], str]] = []
        indices = list(rows.index)
        windows = [indices[i:i + self.max_workers] for i in range(0, len(indices), self.max_workers)]
        lock = Lock()

        with tqdm(total=len(indices), desc="Classifying elections") as pbar:
            for window in windows:
                window_results: List[Tuple[int, str, Optional[int], str]] = []

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    future_to_idx = {
                        executor.submit(self._call_llm, rows.loc[orig_idx]): orig_idx
                        for orig_idx in window
                    }
                    for future in as_completed(future_to_idx):
                        orig_idx = future_to_idx[future]
                        try:
                            pred, conf, reason = future.result()
                        except Exception as e:
                            print(f"\nError on row {orig_idx}: {e}")
                            pred, conf, reason = "0", None, f"Error: {e}"
                        window_results.append((orig_idx, pred, conf, reason))

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
            "Downstream Assembly Classifiers\n"
            "\n"
            "Post-processing scripts that run AFTER the main pipeline.\n"
            "Both classifiers depend on the assembly_prediction column.\n"
            "\n"
            "assembly_extended (default):\n"
            "  0  No assembly (pass-through)  1  No factions  2  With factions\n"
            "\n"
            "elections:\n"
            "  0  No assembly / not elected   1  Elected, no factions  2  Competitive elections\n"
            "\n"
            "all: runs assembly_extended then elections in sequence.\n"
            "\n"
            "Must be run AFTER the main pipeline has produced a predictions file\n"
            "that contains an assembly_prediction column."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Assembly extended (default)
  python pipeline/post_processing.py \\
      --input  data/results/predictions.csv \\
      --output data/results/predictions_extended.csv

  # Elections only
  python pipeline/post_processing.py \\
      --input  data/results/predictions.csv \\
      --output data/results/predictions_extended.csv \\
      --task   elections

  # Both classifiers in sequence
  python pipeline/post_processing.py \\
      --input  data/results/predictions.csv \\
      --output data/results/predictions_extended.csv \\
      --task   all

  # Use a different model, 4 rows in parallel
  python pipeline/post_processing.py \\
      --input  data/results/predictions.csv \\
      --output data/results/predictions_extended.csv \\
      --task   all --model gpt-4o --parallel-rows 4
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
        "--task",
        choices=["assembly_extended", "elections", "all"],
        default="all",
        help=(
            "Which downstream classifier(s) to run (default: assembly_extended).\n"
            "  assembly_extended — codes whether the assembly has competitive factions (0/1/2)\n"
            "  elections         — codes how assembly members are selected (0/1/2)\n"
            "  all               — runs assembly_extended then elections in sequence"
        )
    )
    parser.add_argument(
        "--model", "-m",
        default="gemini-2.5-pro",
        help="LLM model identifier (default: gemini-2.5-pro)"
    )
    parser.add_argument(
        "--assembly-col",
        default="assembly_prediction",
        help="Column name holding binary assembly predictions (default: assembly_prediction)"
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

    args = parser.parse_args()

    # --- Load API keys ---
    api_keys = {
        "openai": os.getenv("OPENAI_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "aws_session_token": os.getenv("AWS_SESSION_TOKEN"),
    }

    # --- Load input data ---
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

    print(f"Loaded {len(df)} rows. "
          f"assembly=0: {(df[args.assembly_col].astype(str) == '0').sum()}, "
          f"assembly=1: {(df[args.assembly_col].astype(str) == '1').sum()}")

    # --- Resolve search mode ---
    search_mode = SearchMode(args.search_mode)

    if search_mode == SearchMode.AGENTIC and not os.getenv("SERPER_API_KEY"):
        raise ValueError(
            "Agentic search requires SERPER_API_KEY environment variable."
        )

    classifier_kwargs = dict(
        model=args.model,
        api_keys=api_keys,
        assembly_col=args.assembly_col,
        max_workers=args.parallel_rows,
        delay=args.delay,
        search_mode=search_mode,
        serper_api_key=os.getenv("SERPER_API_KEY", ""),
    )

    result_df = df.copy()

    # --- Run assembly_extended classifier ---
    if args.task in ("assembly_extended", "all"):
        print("\n=== Running: AssemblyExtendedClassifier ===")
        ae_classifier = AssemblyExtendedClassifier(**classifier_kwargs)
        result_df = ae_classifier.classify(result_df)
        pred_counts = result_df[AssemblyExtendedClassifier.PRED_COL].value_counts().sort_index()
        print("\nassembly_extended_prediction distribution:")
        for label, count in pred_counts.items():
            print(f"  {label}: {count}")

    # --- Run elections classifier ---
    if args.task in ("elections", "all"):
        print("\n=== Running: ElectionsClassifier ===")
        el_classifier = ElectionsClassifier(**classifier_kwargs)
        result_df = el_classifier.classify(result_df)
        pred_counts = result_df[ElectionsClassifier.PRED_COL].value_counts().sort_index()
        print("\nelections_prediction distribution:")
        for label, count in pred_counts.items():
            print(f"  {label}: {count}")

    # --- Save output ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"\nSaved predictions to: {args.output}")


if __name__ == "__main__":
    main()
