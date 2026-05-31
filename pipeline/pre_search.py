"""
Deterministic Pre-Search Pipeline

This module provides search functions that run BEFORE the LLM call,
injecting retrieved context into the prompt. Unlike agentic search where
the LLM decides what to search, pre-search always runs and uses a fixed
query strategy per indicator.

Search order (tiered):
  1. Wikipedia (MediaWiki API) -- free, best for pre-modern polities
  2. DuckDuckGo                -- free, broader web coverage
  3. Serper (Google)           -- optional, requires API key

Usage::

    searcher = PreSearcher(serper_api_key="...")  # serper key is optional
    context = searcher.search("Roman Republic", "Julius Caesar", -49, -44)
    # context is a string of retrieved reference text to inject into prompts
"""

import json
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import requests

from utils.langsmith_utils import traceable


# =============================================================================
# Wikipedia Search (MediaWiki API)
# =============================================================================

def search_wikipedia(
    query: str,
    max_results: int = 3,
    extract_chars: int = 4000,
    url_tracker: Optional[List[str]] = None,
    query_tracker: Optional[List[str]] = None,
) -> str:
    """
    Search Wikipedia using the MediaWiki API and return article extracts.

    Args:
        query: Search query
        max_results: Maximum number of articles to retrieve
        extract_chars: Maximum characters to extract per article
        url_tracker: Optional list to append article URLs to
        query_tracker: Optional list to append query strings to

    Returns:
        Formatted string of Wikipedia article extracts, or empty string
    """
    if query_tracker is not None:
        query_tracker.append(f"[Wikipedia] {query}")

    import time as _time
    from tqdm import tqdm as _tqdm

    search_url = "https://en.wikipedia.org/w/api.php"
    # Contact info required by Wikipedia's bot policy for the 200 req/10s tier
    headers = {
        "User-Agent": "ConstitutionLLM/1.0 (academic research bot; deankuopt@gmail.com)",
    }

    def _wiki_get(params: dict, max_retries: int = 3) -> dict:
        """GET with exponential backoff on 429 Too Many Requests."""
        delay = 2.0
        for attempt in range(max_retries):
            resp = requests.get(search_url, params=params, headers=headers, timeout=15)
            if resp.status_code == 429:
                _tqdm.write(f"WARN: Wikipedia rate-limited (429), retrying in {delay:.0f}s...")
                _time.sleep(delay)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Wikipedia returned 429 after {max_retries} retries")

    try:
        # Step 1: Search for matching articles
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": max_results,
            "format": "json",
        }
        search_data = _wiki_get(search_params)

        results = search_data.get("query", {}).get("search", [])
        if not results:
            return ""

        # Step 2: Get extracts for each matching article
        page_ids = [str(r["pageid"]) for r in results]
        extract_params = {
            "action": "query",
            "pageids": "|".join(page_ids),
            "prop": "extracts|info",
            "exintro": False,
            "explaintext": True,
            "exchars": extract_chars,
            "inprop": "url",
            "format": "json",
        }
        _time.sleep(0.5)  # small gap between the two Wikipedia calls
        extract_data = _wiki_get(extract_params)

        pages = extract_data.get("query", {}).get("pages", {})
        output_parts = []

        for page_id, page in pages.items():
            title = page.get("title", "")
            extract = page.get("extract", "")
            full_url = page.get("fullurl", f"https://en.wikipedia.org/?curid={page_id}")

            tagged_url = f"[Wikipedia] {full_url}"
            if url_tracker is not None and tagged_url not in url_tracker:
                url_tracker.append(tagged_url)

            if extract:
                output_parts.append(
                    f"[Wikipedia] {title}\n"
                    f"URL: {full_url}\n"
                    f"{extract}\n"
                )

        return "\n".join(output_parts)

    except Exception as e:
        _tqdm.write(f"WARNING: Wikipedia search failed for '{query}': {e}")
        return ""


# =============================================================================
# DuckDuckGo Search
# =============================================================================

def search_duckduckgo(
    query: str,
    max_results: int = 5,
    url_tracker: Optional[List[str]] = None,
    query_tracker: Optional[List[str]] = None,
) -> str:
    """
    Search DuckDuckGo using the duckduckgo_search library.

    Args:
        query: Search query
        max_results: Maximum number of results
        url_tracker: Optional list to append result URLs to
        query_tracker: Optional list to append query strings to

    Returns:
        Formatted search results string, or empty string
    """
    if query_tracker is not None:
        query_tracker.append(f"[DuckDuckGo] {query}")

    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            from tqdm import tqdm as _tqdm
            _tqdm.write("WARNING: ddgs not installed. Run: pip install ddgs")
            return ""

    try:
        ddgs = DDGS()
        results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return ""

        output_parts = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "")
            link = result.get("href", "")
            body = result.get("body", "")

            if url_tracker is not None and link:
                url_tracker.append(f"[DuckDuckGo] {link}")

            output_parts.append(
                f"{i}. {title}\n"
                f"   URL: {link}\n"
                f"   {body}\n"
            )

        return "[DuckDuckGo] Results:\n" + "\n".join(output_parts)

    except Exception as e:
        from tqdm import tqdm as _tqdm
        _tqdm.write(f"WARNING: DuckDuckGo search failed for '{query}': {e}")
        return ""


# =============================================================================
# Serper (Google) Search
# =============================================================================

def search_serper(
    query: str,
    serper_api_key: str,
    max_results: int = 5,
    url_tracker: Optional[List[str]] = None,
    query_tracker: Optional[List[str]] = None,
) -> str:
    """
    Search Google via Serper API (requires API key).

    Args:
        query: Search query
        serper_api_key: Serper API key
        max_results: Maximum number of results
        url_tracker: Optional list to append URLs to
        query_tracker: Optional list to append query strings to

    Returns:
        Formatted search results string, or empty string
    """
    if not serper_api_key:
        return ""

    if query_tracker is not None:
        query_tracker.append(f"[Serper] {query}")

    try:
        payload = json.dumps({"q": query})
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }

        response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            data=payload,
            timeout=30
        )
        response.raise_for_status()
        search_results = response.json()

        output = ""
        if "answerBox" in search_results:
            ab = search_results["answerBox"]
            snippet = ab.get("snippet", ab.get("answer", ""))
            output += f"Answer Box: {snippet}\n\n"

        if "organic" in search_results:
            output += "[Serper] Results:\n"
            for i, result in enumerate(search_results["organic"][:max_results], 1):
                title = result.get("title", "")
                link = result.get("link", "")
                snippet = result.get("snippet", "")
                output += f"{i}. {title}\n   URL: {link}\n   {snippet}\n\n"
                if url_tracker is not None and link:
                    url_tracker.append(f"[Serper] {link}")

        return output

    except Exception as e:
        from tqdm import tqdm as _tqdm
        _tqdm.write(f"WARNING: Serper search failed for '{query}': {e}")
        return ""


# =============================================================================
# Pre-Searcher (Tiered)
# =============================================================================

@dataclass
class PreSearchResult:
    """Result from pre-search for a single leader row."""
    context: str
    search_queries: List[str] = field(default_factory=list)
    urls_used: List[str] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)


class PreSearcher:
    """
    Deterministic pre-search that retrieves reference context before LLM calls.

    Searches in order: Wikipedia -> DuckDuckGo -> Serper (if key provided).
    Each tier only runs if previous tiers returned insufficient context.

    Example::

        searcher = PreSearcher(serper_api_key="...")
        result = searcher.search("Roman Republic", "Julius Caesar", -49, -44)
        # result.context contains reference text to inject into prompts
    """

    # Minimum characters of context before skipping lower tiers
    MIN_CONTEXT_CHARS = 200

    def __init__(self, serper_api_key: str = ''):
        self.serper_api_key = serper_api_key

    @traceable(name="PreSearcher.search", run_type="retriever")
    def search(
        self,
        polity: str,
        name: str,
        start_year: int,
        end_year: Optional[int],
    ) -> PreSearchResult:
        """
        Run tiered search for a leader row.

        Args:
            polity: Name of the polity
            name: Leader name
            start_year: Start year
            end_year: End year (None if unknown)

        Returns:
            PreSearchResult with context text and metadata
        """
        url_tracker: List[str] = []
        query_tracker: List[str] = []
        sources: List[str] = []
        context_parts: List[str] = []

        # Build query components — skip empty name (polity pipeline)
        name_part = name.strip() if name else ""

        # Build a clear, informative query string
        if name_part:
            if start_year is not None and end_year is not None:
                base_query = f"{name_part} of {polity} during {start_year}-{end_year}"
            elif start_year is not None:
                base_query = f"{name_part} of {polity} reign started in {start_year}"
            elif end_year is not None:
                base_query = f"{name_part} of {polity} reign ended in {end_year}"
            else:
                base_query = f"{name_part} of {polity}"
        else:
            if start_year is not None and end_year is not None:
                base_query = f"{polity} during {start_year}-{end_year}"
            elif start_year is not None:
                base_query = f"{polity} from {start_year}"
            elif end_year is not None:
                base_query = f"{polity} until {end_year}"
            else:
                base_query = polity

        # --- Tier 1: Wikipedia ---
        wiki_result = search_wikipedia(
            base_query,
            max_results=3,
            extract_chars=4000,
            url_tracker=url_tracker,
            query_tracker=query_tracker,
        )
        if wiki_result:
            context_parts.append(wiki_result)
            sources.append("wikipedia")

        total_context = "\n".join(context_parts)

        # --- Tier 2: DuckDuckGo (if Wikipedia insufficient) ---
        if len(total_context) < self.MIN_CONTEXT_CHARS:
            ddg_result = search_duckduckgo(
                f"{base_query} political history governance",
                max_results=5,
                url_tracker=url_tracker,
                query_tracker=query_tracker,
            )
            if ddg_result:
                context_parts.append(ddg_result)
                sources.append("duckduckgo")
                total_context = "\n".join(context_parts)

        # --- Tier 3: Serper (if still insufficient and key available) ---
        if len(total_context) < self.MIN_CONTEXT_CHARS and self.serper_api_key:
            serper_result = search_serper(
                f"{base_query} governance political system",
                self.serper_api_key,
                max_results=5,
                url_tracker=url_tracker,
                query_tracker=query_tracker,
            )
            if serper_result:
                context_parts.append(serper_result)
                sources.append("serper")

        final_context = "\n".join(context_parts).strip()

        return PreSearchResult(
            context=final_context,
            search_queries=query_tracker,
            urls_used=url_tracker,
            sources_used=sources,
        )

    def enrich_prompt(
        self,
        user_prompt: str,
        search_result: PreSearchResult,
    ) -> str:
        """
        Inject search context into a user prompt.

        Args:
            user_prompt: Original user prompt
            search_result: PreSearchResult from self.search()

        Returns:
            Enriched user prompt with reference material prepended
        """
        if not search_result.context:
            return user_prompt

        return (
            "The following reference material was retrieved from external sources. "
            "Use it to inform your analysis, but rely on your own knowledge when "
            "the sources are incomplete or contradictory.\n\n"
            "--- REFERENCE MATERIAL ---\n"
            f"{search_result.context}\n"
            "--- END REFERENCE MATERIAL ---\n\n"
            f"{user_prompt}"
        )
