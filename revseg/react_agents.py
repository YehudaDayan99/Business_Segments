from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup

from revseg.llm_client import OpenAIChatClient
from revseg.table_candidates import TableCandidate, extract_table_grid_normalized


_WS_RE = re.compile(r"\s+")
_MONEY_CLEAN_RE = re.compile(r"[^0-9.\-]")
_ITEM8_RE = re.compile(
    r"\bitem\s*8\b|\bfinancial statements\b|\bnotes to (?:the )?financial statements\b",
    re.IGNORECASE,
)
_ITEM7_RE = re.compile(r"\bitem\s*7\b|\bmanagement['']s discussion\b|\bmd&a\b", re.IGNORECASE)
_ITEM1_RE = re.compile(r"\bitem\s*1[.\s]+business\b|\bitem\s*1\b(?![\d])", re.IGNORECASE)
_SEGMENT_NOTE_RE = re.compile(r"\bsegment(s)?\b|\breportable segment(s)?\b", re.IGNORECASE)

# Revenue line label expansions for better matching
_LABEL_EXPANSIONS: Dict[str, List[str]] = {
    "advertising": ["advertising", "ad revenue", "ads", "advertising services"],
    "subscription": ["subscription", "subscriptions", "subscriber"],
    "services": ["services", "service revenue", "service offerings"],
    "cloud": ["cloud", "cloud services", "cloud computing"],
    "gaming": ["gaming", "game", "games"],
    "compute": ["compute", "data center", "computing"],
    "networking": ["networking", "network"],
    "professional visualization": ["professional visualization", "visualization", "professional"],
    "automotive": ["automotive", "auto", "vehicle"],
    "online stores": ["online stores", "online store", "e-commerce"],
    "physical stores": ["physical stores", "physical store", "retail stores"],
    "third-party seller": ["third-party seller", "third party seller", "marketplace", "3p seller"],
    "aws": ["aws", "amazon web services", "cloud services"],
}


def _clean(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())


def _extract_section(text: str, section_pattern: re.Pattern, max_chars: int = 50000) -> str:
    """
    Extract a specific section from the filing text.
    
    Finds the section start using the pattern and extracts up to max_chars,
    stopping at the next major section boundary (Item X).
    """
    if not text:
        return ""
    
    match = section_pattern.search(text)
    if not match:
        return ""
    
    start_idx = match.start()
    
    # Find the next section boundary (Item followed by number)
    next_item_pattern = re.compile(r"\bitem\s+\d+", re.IGNORECASE)
    search_start = start_idx + len(match.group())
    
    end_idx = start_idx + max_chars
    for next_match in next_item_pattern.finditer(text, search_start):
        if next_match.start() > start_idx + 500:  # Must be at least 500 chars after start
            end_idx = min(end_idx, next_match.start())
            break
    
    return text[start_idx:end_idx]


def _expand_search_terms(label: str) -> List[str]:
    """
    Expand a revenue line label into multiple search terms for better matching.
    Returns the original label plus any known variations.
    """
    terms = [label.lower()]
    label_low = label.lower().strip()
    
    # Check for known expansions
    for key, expansions in _LABEL_EXPANSIONS.items():
        if key in label_low or label_low in key:
            for exp in expansions:
                if exp.lower() not in terms:
                    terms.append(exp.lower())
    
    # Also add individual significant words (3+ chars)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', label)
    for word in words:
        word_low = word.lower()
        if word_low not in terms and word_low not in ('the', 'and', 'for', 'from', 'other'):
            terms.append(word_low)
    
    return terms


# Pattern to extract footnote markers from labels like "Online stores (1)"
_FOOTNOTE_MARKER_RE = re.compile(r'\((\d+)\)\s*$')


def _extract_footnotes_from_text(text: str, prioritize_includes: bool = True) -> Dict[str, str]:
    """
    Extract footnote definitions from text.
    
    Looks for patterns like:
    - "(1) Includes product sales..."
    - "(2) Includes product sales where..."
    
    Args:
        text: Text to search for footnotes
        prioritize_includes: If True, strongly prefer footnotes starting with "Includes"
        
    Returns dict mapping footnote number to definition text.
    """
    if not text:
        return {}
    
    footnotes: Dict[str, str] = {}
    
    # Pattern: (N) followed by "Includes" - this is the most reliable pattern for revenue footnotes
    for i in range(1, 10):
        # First priority: (N) Includes... (standard revenue footnote format)
        pattern_includes = rf'\({i}\)\s*(Includes\s+[^(]*?)(?=\(\d+\)|_____|$)'
        matches = re.findall(pattern_includes, text, re.IGNORECASE | re.DOTALL)
        if matches:
            for match in matches:
                cleaned = _clean(match)
                if len(cleaned) >= 20:
                    footnotes[str(i)] = cleaned[:600]
                    break
        
        # Second priority: Other substantive verbs
        if str(i) not in footnotes:
            pattern_verbs = rf'\({i}\)\s*((?:Represents|Consists|Comprises|Contains)\s+[^(]*?)(?=\(\d+\)|_____|$)'
            matches2 = re.findall(pattern_verbs, text, re.IGNORECASE | re.DOTALL)
            if matches2:
                for match in matches2:
                    cleaned = _clean(match)
                    if len(cleaned) >= 20:
                        footnotes[str(i)] = cleaned[:600]
                        break
        
        # Third priority (only if not prioritizing includes): Capital letter start
        if str(i) not in footnotes and not prioritize_includes:
            pattern_capital = rf'\({i}\)\s*([A-Z][^(]*?)(?=\(\d+\)|_____|$)'
            matches3 = re.findall(pattern_capital, text, re.DOTALL)
            if matches3:
                for match in matches3:
                    cleaned = _clean(match)
                    # Filter out non-definition matches
                    if len(cleaned) >= 30 and not cleaned[0].isdigit():
                        if cleaned.count('$') <= 1:
                            footnotes[str(i)] = cleaned[:600]
                            break
    
    return footnotes


def _extract_footnote_for_label(label: str, html_text: str, table_context_text: str) -> Optional[str]:
    """
    Extract footnote definition for a revenue line label that contains a footnote marker.
    
    Args:
        label: Revenue line label like "Online stores (1)"
        html_text: Full HTML text to search
        table_context_text: Table's nearby text context
        
    Returns:
        Footnote definition text if found, None otherwise.
    """
    # Check if label has a footnote marker
    match = _FOOTNOTE_MARKER_RE.search(label)
    if not match:
        return None
    
    footnote_num = match.group(1)
    label_clean = _FOOTNOTE_MARKER_RE.sub('', label).strip().lower()
    
    # Strategy: Look for table separator (___) followed by footnotes
    # This is the most reliable way to find footnotes for a specific table
    
    # Step 1: Find the label in context of revenue table, then look for separator + footnotes
    low = html_text.lower() if html_text else ""
    
    # Find occurrences of the label
    label_positions = []
    search_pos = 0
    while True:
        idx = low.find(label_clean, search_pos)
        if idx == -1:
            break
        label_positions.append(idx)
        search_pos = idx + len(label_clean)
        if len(label_positions) >= 10:  # Limit search
            break
    
    # For each label position, look for "_____" separator followed by footnotes
    for label_idx in label_positions:
        # Look for separator after the label (within 3000 chars)
        separator_idx = html_text.find("_____", label_idx, label_idx + 3000)
        if separator_idx != -1:
            # Found separator - extract footnotes from right after it
            footnote_window = html_text[separator_idx:separator_idx + 10000]
            
            # Look specifically for (N) Includes pattern in this window
            pattern = rf'\({footnote_num}\)\s*(Includes\s+[^(]+?)(?=\(\d+\)|_____|$)'
            matches = re.findall(pattern, footnote_window, re.IGNORECASE | re.DOTALL)
            if matches:
                cleaned = _clean(matches[0])
                if len(cleaned) >= 20:
                    return cleaned[:600]
    
    # Step 2: Fallback - search for all (N) Includes patterns near the label
    for label_idx in label_positions:
        window = html_text[label_idx:label_idx + 15000]
        
        # Look for (N) Includes pattern
        pattern = rf'\({footnote_num}\)\s*(Includes\s+[^(]+?)(?=\(\d+\)|_____|$)'
        matches = re.findall(pattern, window, re.IGNORECASE | re.DOTALL)
        if matches:
            cleaned = _clean(matches[0])
            if len(cleaned) >= 20:
                return cleaned[:600]
    
    # Step 3: Try table context
    if table_context_text:
        footnotes = _extract_footnotes_from_text(table_context_text)
        if footnote_num in footnotes:
            return footnotes[footnote_num]
    
    # Step 4: Fallback - search Item 8 section
    item8_match = _ITEM8_RE.search(html_text)
    if item8_match:
        item8_start = item8_match.start()
        item8_section = html_text[item8_start:item8_start + 200000]
        
        # Look for separator + footnotes pattern
        separator_pattern = r'_____+\s*\(\d+\)'
        sep_matches = list(re.finditer(separator_pattern, item8_section))
        for sep_match in sep_matches:
            footnote_window = item8_section[sep_match.start():sep_match.start() + 10000]
            pattern = rf'\({footnote_num}\)\s*(Includes\s+[^(]+?)(?=\(\d+\)|_____|$)'
            matches = re.findall(pattern, footnote_window, re.IGNORECASE | re.DOTALL)
            if matches:
                cleaned = _clean(matches[0])
                if len(cleaned) >= 20:
                    return cleaned[:600]
    
    return None


def _parse_number(s: str) -> Optional[float]:
    if s is None:
        return None
    t = _clean(s)
    if t in {"", "-", "—", "–"}:
        return None
    # Handle parentheses negatives
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg = True
        t = t[1:-1]
    t = t.replace("$", "").replace(",", "").strip()
    try:
        v = float(t)
        return -v if neg else v
    except Exception:
        return None


def _parse_money_to_int(s: str) -> Optional[int]:
    v = _parse_number(s)
    if v is None:
        return None
    return int(round(v))


def rank_candidates_for_financial_tables(candidates: List[TableCandidate]) -> List[TableCandidate]:
    return sorted(
        candidates,
        key=lambda c: (
            float(guess_item8_score(c)),
            bool(getattr(c, "has_year_header", False)),
            bool(getattr(c, "has_units_marker", False)),
            float(getattr(c, "money_cell_ratio", 0.0)),
            float(getattr(c, "numeric_cell_ratio", 0.0)),
            len(getattr(c, "keyword_hits", []) or []),
            int(getattr(c, "n_rows", 0)) * int(getattr(c, "n_cols", 0)),
        ),
        reverse=True,
    )


def guess_item8_score(c: TableCandidate) -> float:
    """Soft signal: does the local context look like Item 8 / Notes / Segment Note?"""
    blob = " ".join(
        [
            str(getattr(c, "heading_context", "") or ""),
            str(getattr(c, "caption_text", "") or ""),
            str(getattr(c, "nearby_text_context", "") or ""),
        ]
    )
    blob = _clean(blob)
    score = 0.0
    if _ITEM8_RE.search(blob):
        score += 3.0
    if _SEGMENT_NOTE_RE.search(blob):
        score += 1.0
    # If it looks like Item 7/MD&A, slightly downweight (soft preference, not exclusion)
    if _ITEM7_RE.search(blob):
        score -= 1.0
    return score


def extract_keyword_windows(
    html_path: Path,
    *,
    keywords: List[str],
    window_chars: int = 2500,
    max_windows: int = 12,
) -> List[str]:
    """Deterministically extract short text windows around keywords for LLM context."""
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    text = _clean(text)
    low = text.lower()

    windows: List[str] = []
    for kw in keywords:
        k = kw.lower()
        start = 0
        while True:
            i = low.find(k, start)
            if i == -1:
                break
            a = max(0, i - window_chars // 3)
            b = min(len(text), i + window_chars)
            snippet = _clean(text[a:b])
            if snippet and snippet not in windows:
                windows.append(snippet)
            start = i + max(1, len(k))
            if len(windows) >= max_windows:
                return windows
    return windows


def document_scout(html_path: Path, *, max_headings: int = 80) -> Dict[str, Any]:
    """Lightweight scan of headings to help the LLM orient itself."""
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    headings: List[str] = []
    for tag in soup.find_all(["h1", "h2", "h3", "b", "strong"]):
        txt = _clean(tag.get_text(" ", strip=True))
        if 5 <= len(txt) <= 180 and txt not in headings:
            headings.append(txt)
        if len(headings) >= max_headings:
            break
    return {"headings": headings}


def _candidate_summary(c: TableCandidate) -> Dict[str, Any]:
    return {
        "table_id": c.table_id,
        "n_rows": c.n_rows,
        "n_cols": c.n_cols,
        "detected_years": c.detected_years,
        "keyword_hits": c.keyword_hits,
        "item8_score": guess_item8_score(c),
        "has_year_header": getattr(c, "has_year_header", False),
        "has_units_marker": getattr(c, "has_units_marker", False),
        "units_hint": getattr(c, "units_hint", ""),
        "money_cell_ratio": getattr(c, "money_cell_ratio", 0.0),
        "numeric_cell_ratio": getattr(c, "numeric_cell_ratio", 0.0),
        "row_label_preview": getattr(c, "row_label_preview", [])[:12],
        "caption_text": getattr(c, "caption_text", "")[:200],
        "heading_context": getattr(c, "heading_context", "")[:200],
        "nearby_text_context": getattr(c, "nearby_text_context", "")[:280],
    }


def select_segment_revenue_table(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    candidates: List[TableCandidate],
    scout: Dict[str, Any],
    snippets: List[str],
    max_candidates: int = 80,
) -> Dict[str, Any]:
    ranked = rank_candidates_for_financial_tables(candidates)[:max_candidates]
    payload = [_candidate_summary(c) for c in ranked]

    system = (
        "You are a financial filings analyst. You select the single best HTML table candidate "
        "that represents REVENUE BY REPORTABLE SEGMENT (or equivalent business segments) for the latest fiscal year. "
        "Prefer tables from Item 8 / Notes to Financial Statements when possible, but you may select other sections if they clearly match and are consistent. "
        "Output must be STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "objective": "Find the reportable segment revenue table (e.g., segments with revenue totals).",
            "headings": scout.get("headings", [])[:40],
            "retrieved_snippets": snippets[:10],
            "table_candidates": payload,
            "output_schema": {
                "table_id": "string like t0071",
                "confidence": "number 0..1",
                "kind": "string, use 'segment_revenue' or 'not_found'",
                "rationale": "short string",
            },
        },
        ensure_ascii=False,
    )
    out = llm.json_call(system=system, user=user, max_output_tokens=700)
    return out


TABLE_KINDS = [
    "segment_revenue",
    "product_service_revenue",
    "segment_results_of_operations",
    "other",
]


def discover_primary_business_lines(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    snippets: List[str],
) -> Dict[str, Any]:
    """Text-first agent: infer primary business lines for CSV1 (Option 1).

    Output contracts:
      - dimension: product_category | reportable_segments
      - segments: list[str] (primary business lines)
      - include_segments_optional: list[str] (e.g., Corporate adjustments) if needed for reconciliation
    """
    system = (
        "You are a financial filings analyst. Determine the primary business-line dimension for CSV1.\n"
        "Rules:\n"
        "- For AAPL, treat business lines as product categories (iPhone, Mac, iPad, Wearables/Home/Accessories, Services).\n"
        "- For MSFT and GOOGL, treat business lines as reportable segments (e.g., Intelligent Cloud).\n"
        "- If the filing includes corporate adjustments (e.g., hedging gains/losses) that are included in Total Revenues, "
        "put that under include_segments_optional=['Corporate'].\n"
        "Output STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "snippets": snippets[:10],
            "few_shot_examples": [
                {
                    "ticker": "AAPL",
                    "dimension": "product_category",
                    "segments": ["iPhone", "Mac", "iPad", "Wearables, Home and Accessories", "Services"],
                    "include_segments_optional": [],
                },
                {
                    "ticker": "MSFT",
                    "dimension": "reportable_segments",
                    "segments": [
                        "Productivity and Business Processes",
                        "Intelligent Cloud",
                        "More Personal Computing",
                    ],
                    "include_segments_optional": [],
                },
                {
                    "ticker": "GOOGL",
                    "dimension": "reportable_segments",
                    "segments": ["Google Services", "Google Cloud", "Other Bets"],
                    "include_segments_optional": ["Corporate"],
                },
            ],
            "output_schema": {
                "dimension": "product_category | reportable_segments",
                "segments": "list[string]",
                "include_segments_optional": "list[string]",
                "notes": "short string",
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=700)


def _classify_table_dimension(c: TableCandidate) -> str:
    """
    Pre-classify a table candidate's disclosure dimension based on metadata.
    
    Returns: 'product_service', 'segment', 'geography', or 'unknown'
    """
    text = " ".join([
        str(getattr(c, "caption_text", "") or ""),
        str(getattr(c, "heading_context", "") or ""),
        " ".join(getattr(c, "row_label_preview", []) or []),
    ]).lower()
    
    # Product/service patterns (most specific first)
    product_service_patterns = [
        r"groups?\s+of\s+similar\s+products?\s+(and|&)\s+services?",
        r"disaggregat(ed|ion)\s+(of\s+)?revenue",
        r"revenue\s+(from\s+external\s+customers\s+)?by\s+(product|service|category)",
        r"net\s+sales\s+by\s+(product|category|type)",
        r"by\s+(product|service)\s+(line|category|type)",
    ]
    for pattern in product_service_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "product_service"
    
    # Geography patterns
    geography_patterns = [
        r"by\s+geograph",
        r"geographic\s+(area|region)",
        r"revenue\s+by\s+region",
    ]
    for pattern in geography_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "geography"
    
    # Segment patterns
    segment_patterns = [
        r"reportable\s+segment",
        r"operating\s+segment",
        r"segment\s+(revenue|result)",
        r"revenue\s+by\s+segment",
    ]
    for pattern in segment_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "segment"
    
    return "unknown"


def select_revenue_disaggregation_table(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    candidates: List[TableCandidate],
    scout: Dict[str, Any],
    snippets: List[str],
    segments: List[str],
    keyword_hints: Optional[List[str]] = None,
    max_candidates: int = 80,
    prefer_granular: bool = True,
) -> Dict[str, Any]:
    """Select the most granular revenue disaggregation table that includes a Total row.
    
    When prefer_granular=True, prioritize tables with product/service line items
    (e.g., 'Revenue by Products and Services') over segment-level totals.
    """
    ranked = rank_candidates_for_financial_tables(candidates)[:max_candidates]
    
    # Pre-classify each candidate's dimension
    payload = []
    for c in ranked:
        summary = _candidate_summary(c)
        summary["inferred_dimension"] = _classify_table_dimension(c)
        payload.append(summary)
    
    # Sort to prioritize product_service dimension tables
    dimension_priority = {"product_service": 0, "segment": 1, "unknown": 2, "geography": 3}
    payload.sort(key=lambda x: dimension_priority.get(x.get("inferred_dimension", "unknown"), 2))
    
    granular_guidance = ""
    if prefer_granular:
        granular_guidance = (
            "- **CRITICAL**: When multiple tables exist, PREFER tables with inferred_dimension='product_service' "
            "over tables with inferred_dimension='segment'. Product/service tables provide the most granular "
            "revenue breakdown (e.g., 'Online stores', 'Third-party seller services', 'Subscription services').\n"
            "- Tables titled 'Net sales by groups of similar products and services' or 'Disaggregation of Revenue' "
            "are BETTER than 'Revenue by Segment' or 'Segment Information' tables.\n"
            "- For Amazon (AMZN), the product/service table has: Online stores, Physical stores, Third-party seller "
            "services, Subscription services, Advertising services, AWS, Other.\n"
        )
    
    system = (
        "You are a financial filings analyst. Select the single best table that DISAGGREGATES revenue "
        "by business lines (segments or product categories) and includes a Total Revenue/Net Sales row.\n"
        "Constraints:\n"
        f"{granular_guidance}"
        "- Ignore geography-only tables (inferred_dimension='geography').\n"
        "- Prefer Item 8 / Notes (Note 17 or Note 18 often has the most granular breakdown).\n"
        "- Prefer tables whose year columns are recent fiscal years (>= 2018).\n"
        "- Each candidate includes 'inferred_dimension' field indicating detected dimension type.\n"
        "Output STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "business_lines": segments,
            "keyword_hints": keyword_hints or [],
            "headings": scout.get("headings", [])[:40],
            "retrieved_snippets": snippets[:10],
            "table_candidates": payload,
            "output_schema": {
                "table_id": "tXXXX",
                "confidence": "0..1",
                "selected_dimension": "product_service|segment|other",
                "rationale": "short string",
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=700)


def infer_disaggregation_layout(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    table_id: str,
    candidate: TableCandidate,
    grid: List[List[str]],
    business_lines: List[str],
    max_rows_for_llm: int = 40,
) -> Dict[str, Any]:
    """Infer layout for tables like:
    - AAPL: Category | Product/Service | FY2025 | FY2024 | ...
    - MSFT/GOOGL: Segment | Product/Service | FY... | ...
    """
    preview = grid[:max_rows_for_llm]
    system = (
        "You analyze a revenue disaggregation table from a 10-K. "
        "Identify which columns correspond to Segment (optional), Item/Product (required), and years, "
        "and how to identify the Total row.\n"
        "Important: year columns should be recent fiscal years (>= 2018) and usually appear as FY2025/FY2024 or 2025/2024.\n"
        "Output STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "table_id": table_id,
            "business_lines": business_lines,
            "candidate_summary": _candidate_summary(candidate),
            "table_grid_preview": preview,
            "output_schema": {
                "segment_col": "int|null (e.g., 0 for Segment; null if no segment column)",
                "item_col": "int (e.g., Product / Service column)",
                "year_cols": {"YYYY": "int column index"},
                "header_rows": "list[int]",
                "total_row_regex": "string regex matching the Total row label (e.g., Total Revenues|Total Net Sales)",
                "exclude_row_regex": "string regex for rows to exclude (e.g., Hedging gains)",
                "units_multiplier": "int (1, 1000, 1000000, 1000000000)",
                "notes": "short string",
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=900)


def extract_disaggregation_rows_from_grid(
    grid: List[List[str]],
    *,
    layout: Dict[str, Any],
    target_year: Optional[int] = None,
    business_lines: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Deterministically extract (segment,item,value) rows and a table total."""
    # Pad rows so column indices inferred from a wide header row work across short rows.
    max_len = max((len(r) for r in grid), default=0)
    if max_len > 0:
        grid = [list(r) + [""] * (max_len - len(r)) for r in grid]

    seg_col = layout.get("segment_col")
    seg_col = int(seg_col) if seg_col is not None else None
    item_col = int(layout["item_col"])
    year_cols_raw = layout.get("year_cols") or {}
    year_cols: Dict[int, int] = {int(y): int(ci) for y, ci in year_cols_raw.items()}
    if not year_cols:
        raise ValueError("No year_cols detected")
    year = target_year or max(year_cols.keys())
    if year not in year_cols:
        year = max(year_cols.keys())
    val_col = year_cols[year]

    header_rows = set(int(i) for i in (layout.get("header_rows") or []))
    total_re = re.compile(layout.get("total_row_regex") or r"total", re.IGNORECASE)
    exclude_re = re.compile(layout.get("exclude_row_regex") or r"$^", re.IGNORECASE)
    mult = int(layout.get("units_multiplier") or 1)
    if mult <= 0:
        mult = 1

    bl_norm = {b.lower(): b for b in (business_lines or [])}
    def _is_business_line(s: str) -> bool:
        if not bl_norm:
            return True
        return s.lower() in bl_norm

    rows: List[Dict[str, Any]] = []
    total_val: Optional[int] = None
    last_seg: str = ""

    for r_i, row in enumerate(grid):
        if r_i in header_rows:
            continue
        if item_col >= len(row) or val_col >= len(row):
            continue
        seg = _clean(row[seg_col]) if seg_col is not None and seg_col < len(row) else ""
        if seg_col is not None:
            if seg:
                last_seg = seg
            else:
                # iXBRL often blanks repeated segment labels; fill down.
                seg = last_seg
        item = _clean(row[item_col])
        if not item:
            continue
        if exclude_re.search(item) or exclude_re.search(seg):
            continue

        # Some tables put a currency symbol column before the number (e.g., '$', '209,586').
        raw_val = _parse_money_to_int(row[val_col])
        if raw_val is None and (val_col + 1) < len(row):
            raw_val = _parse_money_to_int(row[val_col + 1])
        if raw_val is None and (val_col + 2) < len(row):
            raw_val = _parse_money_to_int(row[val_col + 2])
        if raw_val is None:
            continue
        val = int(raw_val) * mult

        # Total row detection: match across the row, not just item/segment cell.
        if total_re.search(item) or total_re.search(seg) or any(total_re.search(_clean(c)) for c in row if c):
            total_val = val
            continue

        if seg and not _is_business_line(seg) and seg.lower() != "corporate":
            # keep corporate as optional; otherwise require match if business lines provided
            continue

        rows.append({"segment": seg, "item": item, "value": val, "year": year})

    return {"year": year, "rows": rows, "total_value": total_val}


def extract_segment_revenue_from_segment_results_grid(
    grid: List[List[str]],
    *,
    segments: List[str],
    target_year: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract segment revenues from a 'segment results of operations' style table.

    Shape example (MSFT t0071):
      - segment header rows: 'Productivity and Business Processes'
      - metric rows under each segment: 'Revenue', 'Cost of revenue', ...
      - final 'Total' section with 'Revenue'
    """
    import re

    # Pad rows to a common width
    max_len = max((len(r) for r in grid), default=0)
    if max_len > 0:
        grid = [list(r) + [""] * (max_len - len(r)) for r in grid]

    year_re = re.compile(r"\b(20\d{2})\b")
    year_cols: dict[int, int] = {}
    for r in grid[:15]:
        for ci, cell in enumerate(r):
            m = year_re.search(str(cell or ""))
            if not m:
                continue
            y = int(m.group(1))
            if 2015 <= y <= 2100:
                year_cols.setdefault(y, ci)
    if not year_cols:
        raise ValueError("No year columns detected in segment results grid")

    year = target_year or max(year_cols.keys())
    if year not in year_cols:
        year = max(year_cols.keys())
    val_col = year_cols[year]

    seg_norm = {s.lower(): s for s in segments}
    current_seg = ""
    out: dict[str, int] = {}
    total_value: Optional[int] = None

    for row in grid:
        if not row:
            continue
        first = _clean(row[0] or "")
        if not first:
            continue

        # Segment header row
        if first.lower() in seg_norm or first.lower() == "total":
            current_seg = seg_norm.get(first.lower(), "Total")
            continue

        # Metric row under current segment
        if first.lower() == "revenue" and current_seg:
            raw = _parse_money_to_int(row[val_col])
            if raw is None and (val_col + 1) < len(row):
                raw = _parse_money_to_int(row[val_col + 1])
            if raw is None and (val_col + 2) < len(row):
                raw = _parse_money_to_int(row[val_col + 2])
            if raw is None:
                continue
            if current_seg == "Total":
                total_value = int(raw)
            else:
                out[current_seg] = int(raw)

    if not out:
        raise ValueError("No segment revenues extracted from segment results grid")
    return {"year": year, "segment_totals": out, "total_value": total_value}


def classify_table_candidates(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    candidates: List[TableCandidate],
    scout: Dict[str, Any],
    snippets: List[str],
    max_candidates: int = 60,
) -> Dict[str, Any]:
    """Classify top candidates into a strict table_kind enum for routing."""
    ranked = rank_candidates_for_financial_tables(candidates)[:max_candidates]
    payload = [_candidate_summary(c) for c in ranked]

    system = (
        "You are a financial filings analyst. Classify each table candidate into a strict table_kind enum.\n"
        "Definitions:\n"
        "- segment_revenue: revenue by reportable segment/business segment\n"
        "- product_service_revenue: revenue by product/service offerings or disaggregation\n"
        "- segment_results_of_operations: segment operating income/costs/expenses (NOT revenue)\n"
        "- other: anything else\n"
        "Output STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "retrieved_snippets": snippets[:8],
            "headings": scout.get("headings", [])[:30],
            "table_candidates": payload,
            "table_kind_enum": TABLE_KINDS,
            "output_schema": {
                "tables": [
                    {
                        "table_id": "tXXXX",
                        "table_kind": "one of table_kind_enum",
                        "confidence": "0..1",
                        "rationale": "short string",
                    }
                ]
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=1200)


def select_other_revenue_tables(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    candidates: List[TableCandidate],
    scout: Dict[str, Any],
    snippets: List[str],
    exclude_table_ids: Iterable[str],
    max_tables: int = 3,
    max_candidates: int = 120,
) -> Dict[str, Any]:
    ranked = rank_candidates_for_financial_tables(candidates)[:max_candidates]
    payload = [_candidate_summary(c) for c in ranked if c.table_id not in set(exclude_table_ids)]

    system = (
        "You are a financial filings analyst. Identify up to N additional REVENUE tables (not the main segments table), "
        "such as revenue by product/service offering, geography, customer type, or disaggregation. "
        "Prefer Item 8 / Notes sources when available; otherwise select the best matching revenue disclosures. "
        "Output must be STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "objective": "Find other revenue tables (product/service offerings etc.)",
            "N": max_tables,
            "headings": scout.get("headings", [])[:40],
            "retrieved_snippets": snippets[:10],
            "table_candidates": payload,
            "output_schema": {
                "tables": [
                    {
                        "table_id": "tXXXX",
                        "kind": "revenue_by_product_service | revenue_by_geography | other_revenue",
                        "confidence": "0..1",
                        "rationale": "short string",
                    }
                ]
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=900)


def extract_table_grid_normalized_with_fallback(
    html_path: Path, table_id: str, *, max_rows: int = 250
) -> List[List[str]]:
    # Wrapper in case we want to add fallbacks later (e.g., pandas.read_html)
    return extract_table_grid_normalized(html_path, table_id, max_rows=max_rows)


def infer_table_layout(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    table_id: str,
    candidate: TableCandidate,
    grid: List[List[str]],
    max_rows_for_llm: int = 30,
) -> Dict[str, Any]:
    """Ask the LLM to identify label/year columns and which rows are data."""
    preview = grid[:max_rows_for_llm]
    system = (
        "You analyze HTML tables from SEC 10-K filings. "
        "Your job: identify which column contains row labels and which columns correspond to fiscal years. "
        "Output STRICT JSON ONLY."
    )
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "table_id": table_id,
            "candidate_summary": _candidate_summary(candidate),
            "table_grid_preview": preview,
            "output_schema": {
                "label_col": "int",
                "year_cols": {"YYYY": "int column index"},
                "header_rows": "list[int] (rows to ignore as header, from the preview)",
                "skip_row_regex": "string regex for rows to skip (e.g., totals, separators) or empty",
                "units_multiplier": "int (1, 1000, 1000000, 1000000000) inferred from units_hint if possible",
                "notes": "short string",
            },
        },
        ensure_ascii=False,
    )
    return llm.json_call(system=system, user=user, max_output_tokens=900)


def extract_revenue_rows_from_grid(
    grid: List[List[str]],
    *,
    layout: Dict[str, Any],
    target_year: Optional[int] = None,
) -> Tuple[int, Dict[str, int]]:
    """Return (year, {label -> revenue_usd_scaled}). Values are scaled by units_multiplier."""
    label_col = int(layout["label_col"])
    year_cols_raw = layout.get("year_cols") or {}
    year_cols: Dict[int, int] = {int(y): int(ci) for y, ci in year_cols_raw.items()}
    if not year_cols:
        raise ValueError("No year_cols detected")

    year = target_year or max(year_cols.keys())
    if year not in year_cols:
        year = max(year_cols.keys())
    value_col = year_cols[year]

    header_rows = set(int(i) for i in (layout.get("header_rows") or []))
    skip_row_re = layout.get("skip_row_regex") or ""
    skip_pat = re.compile(skip_row_re, re.IGNORECASE) if skip_row_re else None
    mult = int(layout.get("units_multiplier") or 1)
    if mult <= 0:
        mult = 1

    out: Dict[str, int] = {}
    for r_i, row in enumerate(grid):
        if r_i in header_rows:
            continue
        if label_col >= len(row) or value_col >= len(row):
            continue
        label = _clean(row[label_col])
        if not label:
            continue
        if skip_pat and skip_pat.search(label):
            continue
        if label.lower() in {"total", "total revenue", "revenues", "net sales"}:
            continue

        val = _parse_money_to_int(row[value_col])
        if val is None:
            continue
        out[label] = int(val) * mult

    return year, out


def summarize_segment_descriptions(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    sec_doc_url: str,
    html_text: str,
    segment_names: List[str],
    revenue_items: Optional[List[str]] = None,
    max_chars_per_segment: int = 6000,
    dimension: str = "segment",
    table_context: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Produce CSV2-style rows via LLM from extracted filing text snippets.
    
    Enhanced to:
    1. Find segment descriptions in Notes sections with better boundary detection
    2. Use revenue_items (from CSV1) as "must include" keywords for grounding
    3. Extract bounded segments (from one segment heading to the next)
    4. For product_service dimension: use table_context (caption/footnotes) instead of global search
    
    Args:
        dimension: The disclosure dimension ('segment', 'product_service', 'end_market').
                   "Other" is only skipped for dimension='segment'.
        table_context: Dict with 'caption', 'heading', 'nearby_text' from accepted table.
                       Used for product_service dimension instead of global text search.
    """
    snippets: Dict[str, str] = {}
    
    # For product_service dimension: use table context instead of global text search
    # This prevents cross-category contamination (e.g., AWS getting advertising terms)
    if dimension == "product_service" and table_context:
        # Build combined table context for product/service category definitions
        table_context_text = " ".join([
            table_context.get("caption", ""),
            table_context.get("heading", ""),
            table_context.get("nearby_text", ""),
        ])
        
        # For product_service categories, use the same snippet for all
        # (table context contains the category definitions/footnotes)
        for seg in segment_names:
            if seg.lower() in ("other", "other revenue", "corporate"):
                continue
            snippets[seg] = _clean(table_context_text) if table_context_text.strip() else ""
        
        # Skip the global search path
        filtered_segments = [s for s in segment_names if s.lower() not in ("other", "other revenue", "corporate")]
        
    else:
        # For segment/end_market dimensions: use global search (existing logic)
        t = html_text
        low = t.lower()
        
        # Find all segment boundary positions for bounded extraction
        segment_positions: Dict[str, List[int]] = {}
        for seg in segment_names:
            seg_low = seg.lower()
            positions = []
            idx = 0
            while True:
                found = low.find(seg_low, idx)
                if found == -1:
                    break
                positions.append(found)
                idx = found + 1
            segment_positions[seg] = positions
        
        # Key section markers
        segment_info_patterns = [
            "segment information",
            "reportable segments",
            "note 18",
            "note 17",
            "segment results",
        ]
        segment_info_idx = -1
        for pattern in segment_info_patterns:
            idx = low.find(pattern)
            if idx >= 0:
                segment_info_idx = idx
                break
        
        # Notes section
        notes_idx = low.find("notes to consolidated financial statements")
        if notes_idx == -1:
            notes_idx = low.find("notes to financial statements")
        
        # Item 1 Business section
        item1_idx = low.find("item 1")
        item1_business_idx = low.find("item 1.", item1_idx) if item1_idx >= 0 else -1
        
        for seg in segment_names:
            # Only skip "Other" for segment dimensions (residual catch-all)
            # Include "Other" for product_service/end_market (explicit revenue line)
            if dimension == "segment" and seg.lower() in ("other", "other revenue", "corporate"):
                continue
                
            key = seg
            seg_low = seg.lower()
            positions = segment_positions.get(seg, [])
            
            if not positions:
                snippets[key] = ""
                continue
            
            # Find the best occurrence using priority:
            # 1. In segment info section (Note 17/18)
            # 2. In Notes section after segment_info_idx
            # 3. In Item 1 Business section
            # 4. First occurrence
            
            best_idx = -1
            
            # Priority 1: In segment info section
            if segment_info_idx >= 0:
                for pos in positions:
                    if pos >= segment_info_idx and pos < segment_info_idx + 100000:
                        best_idx = pos
                        break
            
            # Priority 2: In Notes section
            if best_idx == -1 and notes_idx >= 0:
                for pos in positions:
                    if pos >= notes_idx:
                        best_idx = pos
                        break
            
            # Priority 3: In Item 1
            if best_idx == -1 and item1_business_idx >= 0:
                for pos in positions:
                    if pos >= item1_business_idx:
                        best_idx = pos
                        break
            
            # Priority 4: First occurrence
            if best_idx == -1 and positions:
                best_idx = positions[0]
            
            if best_idx == -1:
                snippets[key] = ""
                continue
            
            # Find the end boundary (next segment heading or max chars)
            end_idx = best_idx + max_chars_per_segment
            other_seg_names = [s for s in segment_names if s.lower() != seg_low and s.lower() not in ("other", "corporate")]
            for other_seg in other_seg_names:
                other_pos = low.find(other_seg.lower(), best_idx + len(seg))
                if other_pos > best_idx and other_pos < end_idx:
                    # Found next segment - use as boundary but include some padding
                    end_idx = min(end_idx, other_pos + 200)
            
            # Extract bounded snippet
            start = max(0, best_idx - 200)
            end = min(len(t), end_idx)
            snippets[key] = _clean(t[start:end])
        
        # Filter out "Other" segments for the else path
        filtered_segments = [s for s in segment_names if s.lower() not in ("other", "other revenue", "corporate")]
    
    system = (
        "You summarize company business segments from SEC 10-K text. "
        "CRITICAL RULES:\n"
        "1. For each segment, write a description GROUNDED in the provided text snippet.\n"
        "2. List SPECIFIC product/brand names that appear in the text "
        "(e.g., 'Azure', 'Microsoft 365 Commercial', 'LinkedIn', 'YouTube ads').\n"
        "3. The 'revenue_items_from_filing' field contains ACTUAL revenue line items from the 10-K. "
        "Map these to the correct segment and include them in key_products_services.\n"
        "4. Do NOT invent products not mentioned in the text or revenue items.\n"
        "Output STRICT JSON ONLY."
    )
    
    # Build segment data with revenue items mapping hint
    segment_data = []
    for s in filtered_segments:
        item_data = {
            "segment": s,
            "text_snippet": snippets.get(s, ""),
        }
        # Add revenue items hint if provided
        if revenue_items:
            item_data["revenue_items_from_filing"] = revenue_items
        segment_data.append(item_data)
    
    user = json.dumps(
        {
            "ticker": ticker,
            "company_name": company_name,
            "sec_doc_url": sec_doc_url,
            "segments": segment_data,
            "output_schema": {
                "rows": [
                    {
                        "segment": "string",
                        "segment_description": "string (comprehensive, 2-3 sentences, grounded in text)",
                        "key_products_services": "list[string] (specific brand/product names from text and revenue_items)",
                        "primary_source": "string short",
                    }
                ]
            },
        },
        ensure_ascii=False,
    )
    result = llm.json_call(system=system, user=user, max_output_tokens=2000)
    
    # Enrich LLM output rows with original text snippets for downstream validation
    rows = result.get("rows", [])
    for row in rows:
        seg_name = row.get("segment", "")
        row["text_snippet"] = snippets.get(seg_name, "")
    
    return result


def expand_key_items_per_segment(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    sec_doc_url: str,
    segment_rows: List[Dict[str, Any]],
    html_text: str = "",
    dimension: str = "segment",
) -> Dict[str, Any]:
    """Produce CSV3 rows: key items per segment with short + long description.
    
    EVIDENCE-BASED EXTRACTION:
    - Only extract items that appear verbatim in the provided text
    - Each item must include an evidence_span copied from the source
    - Post-validate that evidence_span exists in SEGMENT-SPECIFIC snippet (not full HTML)
    
    Process each segment individually to prevent token truncation and cross-segment leakage.
    
    Args:
        dimension: The disclosure dimension ('segment', 'product_service', 'end_market').
                   "Other" is only skipped for dimension='segment'.
    """
    all_rows: List[Dict[str, Any]] = []
    seen_items: set = set()  # For de-duplication
    html_text_lower = html_text.lower() if html_text else ""
    
    for seg_row in segment_rows:
        segment_name = seg_row.get("segment", "Unknown")
        
        # Only skip "Other" for segment dimensions (residual catch-all)
        # Include "Other" for product_service/end_market (explicit revenue line)
        if dimension == "segment" and segment_name.lower() in ("other", "other revenue", "corporate"):
            continue
        
        segment_description = seg_row.get("segment_description", "")
        key_products = seg_row.get("key_products_services", [])
        
        # Use segment-specific snippet for evidence validation (prevents cross-segment leakage)
        segment_snippet = seg_row.get("text_snippet", "")
        segment_snippet_lower = segment_snippet.lower() if segment_snippet else ""
        
        system = (
            "You are an EXTRACTIVE information retrieval system. You ONLY output items that are "
            "EXPLICITLY NAMED as products, services, or brands in the provided text.\n\n"
            "STRICT RULES - VIOLATIONS WILL BE REJECTED:\n"
            "1. ONLY output items whose EXACT NAME appears VERBATIM in the text.\n"
            "2. 'evidence_span' MUST be a WORD-FOR-WORD quote from the text (15-40 words) that includes the item name.\n"
            "3. Do NOT infer, generalize, or add items based on your knowledge. If it's not in the text, don't output it.\n"
            "4. Do NOT output generic categories (e.g., 'cloud services', 'advertising') unless that EXACT phrase names a distinct product.\n"
            "5. If only 1-3 items are explicitly named, output only those 1-3 items. Empty output is valid.\n"
            "6. Descriptions must summarize ONLY what the text says, not general knowledge.\n\n"
            "Output STRICT JSON ONLY."
        )
        # Use text_snippet as primary source (raw filing text), not segment_description (LLM output)
        user = json.dumps(
            {
                "ticker": ticker,
                "company_name": company_name,
                "segment": segment_name,
                "source_text": segment_snippet if segment_snippet else segment_description,
                "key_products_hint": key_products,
                "instructions": (
                    "EXTRACT items from 'source_text' ONLY. "
                    "For each item, the evidence_span must be copy-pasted VERBATIM from source_text "
                    "and must contain the item name. "
                    "If you cannot find verbatim evidence for an item in source_text, DO NOT include it. "
                    "key_products_hint is for reference only - do not include items not in source_text."
                ),
                "output_schema": {
                    "rows": [
                        {
                            "segment": "string (must match input segment name exactly)",
                            "business_item": "string (EXACT product/brand name as it appears in source_text)",
                            "business_item_short_description": "string (1 sentence from source_text)",
                            "business_item_long_description": "string (2-3 sentences from source_text)",
                            "evidence_span": "string (VERBATIM quote from source_text, 15-40 words, containing the item name)",
                        }
                    ]
                },
            },
            ensure_ascii=False,
        )
        try:
            result = llm.json_call(system=system, user=user, max_output_tokens=1800)
            rows = result.get("rows", [])
            
            for row in rows:
                row["segment"] = segment_name
                item_name = row.get("business_item", "").strip()
                evidence = row.get("evidence_span", "").strip()
                
                # De-duplication check
                item_key = item_name.lower().replace(" ", "").replace("-", "")
                if item_key in seen_items:
                    continue
                
                # SEGMENT-SCOPED Evidence validation (prevents cross-segment leakage)
                # Validate against segment snippet first, fall back to full HTML if no snippet
                validation_text = segment_snippet_lower if segment_snippet_lower else html_text_lower
                
                evidence_found = False
                item_in_text = item_name.lower() in validation_text if item_name else False
                item_in_evidence = item_name.lower() in evidence.lower() if (item_name and evidence) else False
                
                if evidence and validation_text:
                    # Normalize for matching (remove extra spaces, lowercase)
                    evidence_normalized = " ".join(evidence.lower().split())
                    
                    # Check if evidence span exists in segment snippet
                    if evidence_normalized in validation_text:
                        evidence_found = True
                    else:
                        # Stricter fuzzy match: 85% of significant words must appear
                        evidence_words = [w for w in evidence_normalized.split() if len(w) > 3]
                        if len(evidence_words) >= 4:
                            matches = sum(1 for w in evidence_words if w in validation_text)
                            if matches / len(evidence_words) >= 0.85:
                                evidence_found = True
                
                # Strict acceptance criteria:
                # 1. Item name must appear in segment snippet, AND
                # 2. Either evidence is validated OR evidence contains the item name
                accept_item = item_in_text and (evidence_found or item_in_evidence)
                
                if accept_item or not validation_text:
                    row["evidence_validated"] = evidence_found
                    row["item_in_source"] = item_in_text
                    row["item_in_evidence"] = item_in_evidence
                    seen_items.add(item_key)
                    all_rows.append(row)
                else:
                    # Reject items that fail segment-scoped validation
                    reject_reason = []
                    if not item_in_text:
                        reject_reason.append("item_not_in_segment_text")
                    if not evidence_found:
                        reject_reason.append("evidence_not_in_segment")
                    if not item_in_evidence:
                        reject_reason.append("item_not_in_evidence")
                    print(f"[{ticker}] Rejected item '{item_name}' for segment '{segment_name}' - {', '.join(reject_reason)}", flush=True)
                    
        except Exception as e:
            print(f"[{ticker}] Warning: expand_key_items failed for segment '{segment_name}': {e}", flush=True)
            continue
    
    return {"rows": all_rows}


def describe_revenue_lines(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    fiscal_year: int,
    revenue_lines: List[Dict[str, Any]],
    table_context: Dict[str, str],
    html_text: str,
    max_chars_per_line: int = 5000,
) -> Dict[str, Any]:
    """
    Generate company-language descriptions for each revenue line.
    
    Enhanced with section-aware priority search:
    1. Item 1 Business (primary source for product/service descriptions)
    2. MD&A (Item 7) - management discussion
    3. Notes to Financial Statements (Item 8)
    4. Full text fallback
    
    Args:
        revenue_lines: List of dicts with 'item' (revenue line label) and 'value'
        table_context: Dict with 'caption', 'heading', 'nearby_text' from accepted table
        html_text: Full filing text for evidence retrieval
        
    Returns:
        Dict with 'rows' containing line descriptions
    """
    if not revenue_lines:
        return {"rows": []}
    
    # Build combined table context
    table_context_text = " ".join([
        table_context.get("caption", ""),
        table_context.get("heading", ""),
        table_context.get("nearby_text", ""),
    ])
    
    # STEP 1: Try to extract footnote definitions for lines with footnote markers
    # This is the most reliable source for AMZN-style disclosures
    footnote_descriptions: Dict[str, str] = {}
    for line_info in revenue_lines:
        item_label = line_info.get("item", "")
        if not item_label:
            continue
        
        footnote_desc = _extract_footnote_for_label(item_label, html_text, table_context_text)
        if footnote_desc:
            footnote_descriptions[item_label] = footnote_desc
    
    # Lines that still need LLM-based description extraction
    lines_needing_llm = [
        line_info for line_info in revenue_lines
        if line_info.get("item", "") not in footnote_descriptions
    ]
    
    # If we got footnotes for all lines, skip LLM entirely
    if not lines_needing_llm:
        return {
            "rows": [
                {"revenue_line": line_info.get("item", ""), "description": footnote_descriptions.get(line_info.get("item", ""), "")}
                for line_info in revenue_lines
            ]
        }
    
    # STEP 2: Section-aware search for remaining lines
    # Pre-extract major sections for priority search
    item1_section = _extract_section(html_text, _ITEM1_RE, max_chars=80000)
    item7_section = _extract_section(html_text, _ITEM7_RE, max_chars=80000)
    item8_section = _extract_section(html_text, _ITEM8_RE, max_chars=80000)
    
    # Priority order: Item 1 → MD&A → Notes → Full text
    sections_priority = [
        ("item1", item1_section),
        ("item7", item7_section),
        ("item8", item8_section),
        ("full", html_text),
    ]
    
    evidence_by_line: Dict[str, str] = {}
    
    for line_info in lines_needing_llm:
        item_label = line_info.get("item", "")
        if not item_label:
            continue
        
        # Expand search terms for this label
        search_terms = _expand_search_terms(item_label)
        
        snippets = []
        
        # 1. Always include table context first (footnotes often have descriptions)
        if table_context_text:
            snippets.append(f"[TABLE CONTEXT] {table_context_text[:2000]}")
        
        # 2. Search through sections in priority order
        for section_name, section_text in sections_priority:
            if not section_text:
                continue
            
            section_low = section_text.lower()
            
            # Try each search term
            for term in search_terms:
                idx = section_low.find(term)
                if idx != -1:
                    # Found a match - extract larger window
                    start = max(0, idx - 800)
                    end = min(len(section_text), idx + 2500)
                    window = section_text[start:end]
                    snippet = f"[{section_name.upper()}] {_clean(window)}"
                    
                    # Avoid duplicates
                    if snippet not in snippets:
                        snippets.append(snippet)
                    
                    # Found in this section, try to get more context
                    # Look for additional matches in this section
                    next_idx = section_low.find(term, idx + len(term) + 500)
                    if next_idx != -1 and len(snippets) < 5:
                        start2 = max(0, next_idx - 500)
                        end2 = min(len(section_text), next_idx + 2000)
                        window2 = section_text[start2:end2]
                        snippet2 = f"[{section_name.upper()}] {_clean(window2)}"
                        if snippet2 not in snippets:
                            snippets.append(snippet2)
                    
                    break  # Found in this section, move to next section
            
            # Stop if we have enough snippets
            if len(snippets) >= 4:
                break
        
        # Combine snippets for this line
        combined = " [...] ".join(snippets)[:max_chars_per_line]
        evidence_by_line[item_label] = combined
    
    # Build LLM prompt for lines that need LLM extraction
    llm_descriptions: Dict[str, str] = {}
    
    if lines_needing_llm and evidence_by_line:
        system = (
            "You are extracting product/service descriptions from SEC 10-K filings.\n\n"
            "CRITICAL RULES:\n"
            "1. Use ONLY company language from the provided evidence text.\n"
            "2. Each description should be 1-2 sentences explaining what the revenue line includes.\n"
            "3. Quote or closely paraphrase the company's own words.\n"
            "4. Focus on the [ITEM1] and [ITEM7] sections which contain business descriptions.\n"
            "5. If no description is found in the evidence, return an empty string.\n"
            "6. Do NOT invent or infer descriptions not present in the text.\n\n"
            "Output STRICT JSON ONLY."
        )
        
        lines_data = []
        for line_info in lines_needing_llm:
            item = line_info.get("item", "")
            lines_data.append({
                "revenue_line": item,
                "evidence_text": evidence_by_line.get(item, "")[:4000],
            })
        
        user = json.dumps(
            {
                "ticker": ticker,
                "company_name": company_name,
                "fiscal_year": fiscal_year,
                "revenue_lines": lines_data,
                "instructions": (
                    "For each revenue_line, extract a description from evidence_text using company language. "
                    "Priority sources: [ITEM1] for business descriptions, [ITEM7] for MD&A context, [TABLE CONTEXT] for footnotes. "
                    "Focus on what products/services are included in this revenue category. "
                    "If the evidence doesn't describe this line, return empty string."
                ),
                "output_schema": {
                    "rows": [
                        {
                            "revenue_line": "string (exact match from input)",
                            "description": "string (1-2 sentences in company language, or empty)",
                        }
                    ]
                },
            },
            ensure_ascii=False,
        )
        
        try:
            result = llm.json_call(system=system, user=user, max_output_tokens=2500)
            rows = result.get("rows", [])
            
            for row in rows:
                line = row.get("revenue_line", "")
                desc = row.get("description", "")
                if line:
                    llm_descriptions[line] = desc
                
        except Exception as e:
            print(f"[{ticker}] Warning: describe_revenue_lines LLM call failed: {e}", flush=True)
    
    # STEP 3: Merge footnote descriptions with LLM descriptions
    # Footnote descriptions take priority (they are direct quotes from the filing)
    output_rows = []
    for line_info in revenue_lines:
        item = line_info.get("item", "")
        # Priority: footnote > LLM
        description = footnote_descriptions.get(item, "") or llm_descriptions.get(item, "")
        output_rows.append({
            "revenue_line": item,
            "description": description,
        })
    
    return {"rows": output_rows}

