from __future__ import annotations

import csv
import functools
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup

# ============================================================================
# CACHING LAYER - Avoid redundant LLM calls and HTML parsing within a run
# ============================================================================
# These caches are per-process and cleared between runs.
# Key insight: Same filing → same result, so cache by file path.

_CACHE_HTML_TEXT: Dict[str, str] = {}
_CACHE_CANDIDATES: Dict[str, List] = {}
_CACHE_SCOUT: Dict[str, Dict] = {}
_CACHE_SNIPPETS: Dict[str, List] = {}
_CACHE_DISCOVERY: Dict[str, Dict] = {}


def _clear_caches() -> None:
    """Clear all caches (call at start of run_pipeline)."""
    _CACHE_HTML_TEXT.clear()
    _CACHE_CANDIDATES.clear()
    _CACHE_SCOUT.clear()
    _CACHE_SNIPPETS.clear()
    _CACHE_DISCOVERY.clear()

from revseg.llm_client import OpenAIChatClient
from revseg.react_agents import (
    document_scout,
    extract_table_grid_normalized,
    rank_candidates_for_financial_tables,
    extract_keyword_windows,
    discover_primary_business_lines,
    select_revenue_disaggregation_table,
    infer_disaggregation_layout,
    summarize_segment_descriptions,
    expand_key_items_per_segment,
    describe_revenue_lines,
)
from revseg.mappings import get_segment_for_item, is_subtotal_row, is_total_row, is_adjustment_item
from revseg.sec_edgar import SEC_ARCHIVES_BASE, download_latest_10k
from revseg.table_candidates import (
    TableCandidate,
    extract_table_candidates_from_html,
    find_latest_downloaded_filing_dir,
    find_primary_document_html,
    write_candidates_json,
)
from revseg.validate import fetch_companyfacts_total_revenue_usd
from revseg.table_kind import tablekind_gate
from revseg.extraction import (
    extract_revenue_unified,
    extract_with_layout_fallback,
    validate_extraction,
    detect_dimension,
    ExtractionResult,
    ValidationResult,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_company_name_from_submission(filing_dir: Path) -> str:
    sub = filing_dir / "submission.json"
    if not sub.exists():
        return ""
    data = _load_json(sub)
    return str(data.get("name") or "").strip()


def _read_filing_ref(filing_dir: Path) -> Dict[str, Any]:
    ref = filing_dir / "filing_ref.json"
    if not ref.exists():
        raise FileNotFoundError(f"Missing filing_ref.json in {filing_dir}")
    return _load_json(ref)


def _sec_doc_url_from_filing_ref(ref: Dict[str, Any]) -> str:
    cik = int(ref["cik"])
    acc = str(ref["accession_number"])
    primary = str(ref["primary_document"])
    return f"{SEC_ARCHIVES_BASE}/{cik}/{acc.replace('-', '')}/{primary}"


def _ensure_filing_dir(ticker: str, *, base_dir: Path, cache_dir: Optional[Path]) -> Path:
    try:
        return find_latest_downloaded_filing_dir(base_dir, ticker)
    except Exception:
        return download_latest_10k(
            ticker,
            base_dir,
            cache_dir=cache_dir,
            include_amendments=False,
            min_interval_s=0.2,
        )


import re

# Dimension priority for CSV1 (prefer product/service granularity over segment totals)
_DIMENSION_PRIORITY = ["product_service", "revenue_source", "end_market", "segment"]


def _clean_revenue_line(label: str) -> str:
    """Remove footnote markers like (1), (4) from revenue line labels."""
    return re.sub(r'\s*\(\d+\)\s*$', '', label).strip()


def _select_primary_dimension(rows: list) -> str:
    """
    Select the best dimension for CSV1 output to avoid duplication.
    Prefers product/service granularity over segment totals.
    """
    dimensions = set(r.dimension for r in rows if hasattr(r, 'dimension'))
    for dim in _DIMENSION_PRIORITY:
        if dim in dimensions:
            return dim
    return "segment"  # fallback


def _get_revenue_group(ticker: str, item: str, dimension: str, row_segment: str = "") -> str:
    """
    Determine Revenue Group for CSV1.
    Uses mappings.py for segment attribution, else:
    - For segment dimension: use segment name directly
    - For other dimensions: use "Product/Service disclosure"
    """
    # Check for explicit segment mapping
    segment = get_segment_for_item(ticker, item)
    if segment:
        return segment
    
    # If this is a segment-level row, use the segment name as Revenue Group
    if dimension == "segment" and row_segment:
        return row_segment
    
    # No mapping found - use "Product/Service disclosure"
    return "Product/Service disclosure"


def _to_millions(value_usd: int) -> float:
    """Convert base units (USD) to millions."""
    return round(value_usd / 1_000_000, 2)


def _html_text_for_llm(html_path: Path, *, max_chars: int = 400_000) -> str:
    """Extract text from HTML for LLM consumption (cached).
    
    Note: Increased limit to 400k to ensure footnotes (which appear after tables)
    are captured for description extraction.
    """
    cache_key = str(html_path)
    if cache_key in _CACHE_HTML_TEXT:
        return _CACHE_HTML_TEXT[cache_key]
    
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(" ", strip=True)
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[:max_chars]
    
    _CACHE_HTML_TEXT[cache_key] = text
    return text


def _cached_extract_candidates(html_path: Path, preview_rows: int = 15, preview_cols: int = 10) -> List:
    """Extract table candidates from HTML (cached)."""
    cache_key = str(html_path)
    if cache_key in _CACHE_CANDIDATES:
        return _CACHE_CANDIDATES[cache_key]
    
    candidates = extract_table_candidates_from_html(html_path, preview_rows=preview_rows, preview_cols=preview_cols)
    _CACHE_CANDIDATES[cache_key] = candidates
    return candidates


def _cached_document_scout(html_path: Path) -> Dict[str, Any]:
    """Run document scout (cached)."""
    cache_key = str(html_path)
    if cache_key in _CACHE_SCOUT:
        return _CACHE_SCOUT[cache_key]
    
    scout = document_scout(html_path)
    _CACHE_SCOUT[cache_key] = scout
    return scout


def _cached_extract_snippets(html_path: Path, keywords: List[str], window_chars: int, max_windows: int) -> List[Dict]:
    """Extract keyword windows from HTML (cached)."""
    cache_key = str(html_path)
    if cache_key in _CACHE_SNIPPETS:
        return _CACHE_SNIPPETS[cache_key]
    
    snippets = extract_keyword_windows(
        html_path,
        keywords=keywords,
        window_chars=window_chars,
        max_windows=max_windows,
    )
    _CACHE_SNIPPETS[cache_key] = snippets
    return snippets


def _cached_discover_business_lines(
    llm: OpenAIChatClient,
    ticker: str,
    company_name: str,
    snippets: List[Dict],
    html_path: Path,
) -> Dict[str, Any]:
    """Discover primary business lines (cached by filing path)."""
    # Cache key uses html_path to ensure per-filing caching
    cache_key = str(html_path)
    if cache_key in _CACHE_DISCOVERY:
        return _CACHE_DISCOVERY[cache_key]
    
    discovery = discover_primary_business_lines(
        llm, ticker=ticker, company_name=company_name, snippets=snippets
    )
    _CACHE_DISCOVERY[cache_key] = discovery
    return discovery


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _trace_append(t_art: Path, event: Dict[str, Any]) -> None:
    """Append a trace event (JSONL) for audit/debug."""
    trace_path = t_art / "trace.jsonl"
    with trace_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


# Note: _override_layout_with_heuristics removed - now handled in extraction/core.py


def _keyword_hints_for_ticker(ticker: str) -> List[str]:
    t = ticker.upper()
    if t == "AAPL":
        return ["net sales", "total net sales", "category", "product / service"]
    if t == "MSFT":
        return [
            "significant product and service offerings",
            "server products and cloud services",
            "microsoft 365",
            "gaming",
            "linkedin",
            "windows and devices",
            "search and news advertising",
            "dynamics products and cloud services",
        ]
    if t == "GOOGL":
        return [
            "disaggregation of revenue",
            "google search & other",
            "youtube ads",
            "google network",
            "subscriptions, platforms, and devices",
            "hedging gains",
            "total revenues",
        ]
    return []


def _canonicalize_business_lines(ticker: str, segments: List[str]) -> List[str]:
    """Light canonicalization - fuzzy matching in extraction handles most normalization."""
    out: List[str] = []
    for s in segments:
        ss = str(s or "").strip()
        if ss:
            out.append(ss)
    return out


# Note: _looks_like_geography_table, _pick_segment_results_candidate, 
# _business_line_overlap_score, _is_corporate_adjustment_item removed
# - Now handled by unified extraction in extraction/core.py


def _pick_income_statement_candidate(candidates: List[TableCandidate]) -> Optional[TableCandidate]:
    """Heuristic pick: table whose preview mentions Revenue + Net income/Operating income."""
    best: Optional[TableCandidate] = None
    best_score = -1.0
    for c in candidates:
        preview_text = " ".join([" ".join(r) for r in (c.preview or [])]).lower()
        if "revenue" not in preview_text:
            continue
        score = 0.0
        if "net income" in preview_text:
            score += 3.0
        if "operating income" in preview_text:
            score += 2.0
        if "cost of revenue" in preview_text or "cost of sales" in preview_text:
            score += 1.5
        score += float(getattr(c, "money_cell_ratio", 0.0)) * 3.0
        score += float(getattr(c, "numeric_cell_ratio", 0.0)) * 1.0
        if getattr(c, "has_year_header", False):
            score += 1.0
        if score > best_score:
            best_score = score
            best = c
    return best


def _has_negative(values: Dict[str, int]) -> bool:
    return any((v is not None and int(v) < 0) for v in values.values())

def _is_nonempty_revenue_set(values: Dict[str, int], *, min_rows: int = 2) -> bool:
    if not values:
        return False
    if len(values) < min_rows:
        return False
    try:
        s = sum(int(v) for v in values.values())
    except Exception:
        return False
    return s > 0


def _pick_best_by_kind(
    classifications: Dict[str, Any],
    *,
    kind: str,
    fallback_ranked: List[TableCandidate],
    exclude: set[str] | None = None,
) -> List[str]:
    """Return ordered table_ids for a given kind based on LLM classification + fallback rank."""
    exclude = exclude or set()
    tables = (classifications.get("tables") or []) if isinstance(classifications, dict) else []
    scored: List[Tuple[float, str]] = []
    for t in tables:
        try:
            tid = str(t.get("table_id") or "")
            tk = str(t.get("table_kind") or "")
            conf = float(t.get("confidence") or 0.0)
        except Exception:
            continue
        if not tid or tid in exclude:
            continue
        if tk != kind:
            continue
        scored.append((conf, tid))
    scored.sort(reverse=True)
    ordered = [tid for _, tid in scored]

    # IMPORTANT: Do NOT backfill with non-classified tables. If classifier returns none for a kind,
    # the caller should fall back to a dedicated selector, not guess via generic ranking.
    return ordered

def _extract_row_value_for_year(
    grid: List[List[str]],
    *,
    layout: Dict[str, Any],
    row_label_regex: str,
    year: int,
) -> Optional[int]:
    import re

    label_col = int(layout["label_col"])
    year_cols_raw = layout.get("year_cols") or {}
    year_cols: Dict[int, int] = {int(y): int(ci) for y, ci in year_cols_raw.items()}
    if year not in year_cols:
        return None
    value_col = year_cols[year]
    header_rows = set(int(i) for i in (layout.get("header_rows") or []))
    pat = re.compile(row_label_regex, re.IGNORECASE)
    mult = int(layout.get("units_multiplier") or 1)
    if mult <= 0:
        mult = 1

    for r_i, row in enumerate(grid):
        if r_i in header_rows:
            continue
        if label_col >= len(row) or value_col >= len(row):
            continue
        lab = str(row[label_col] or "").strip()
        if not lab:
            continue
        if not pat.search(lab):
            continue
        v = row[value_col]
        # parse simple money-ish values
        try:
            s = str(v).replace("$", "").replace(",", "").strip()
            if s.startswith("(") and s.endswith(")"):
                s = "-" + s[1:-1]
            return int(round(float(s))) * mult
        except Exception:
            continue
    return None


def run_pipeline(
    *,
    tickers: List[str],
    out_dir: Path | str = Path("data/outputs"),
    filings_base_dir: Path | str = Path("data/10k"),
    cache_dir: Path | str = Path(".cache/sec"),
    model_fast: str = "gpt-4.1-mini",
    model_quality: str = "gpt-4.1",
    max_react_iters: int = 3,
    validation_tolerance_pct: float = 0.02,
    csv1_only: bool = False,
) -> Dict[str, Any]:
    """End-to-end run for multiple tickers (latest 10-K per ticker).
    
    Uses tiered model approach:
    - model_fast (gpt-4.1-mini): high-volume tasks (table selection, layout inference)
    - model_quality (gpt-4.1): quality-critical tasks (descriptions, segment discovery)
    
    Args:
        csv1_only: If True, skip CSV2/CSV3 generation to save tokens.
    """
    # Clear caches at start of run to ensure fresh state
    _clear_caches()
    
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    filings_base_dir = Path(filings_base_dir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()

    # Tiered model approach: fast for volume, quality for descriptions
    llm_fast = OpenAIChatClient(model=model_fast, rate_limit_rpm=60.0)
    llm_quality = OpenAIChatClient(model=model_quality, rate_limit_rpm=30.0)
    
    # Default llm for backward compatibility
    llm = llm_fast

    artifacts_dir = out_dir.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    csv1_rows: List[Dict[str, Any]] = []
    csv2_rows: List[Dict[str, Any]] = []
    csv3_rows: List[Dict[str, Any]] = []

    report: Dict[str, Any] = {"tickers": {}, "outputs_dir": str(out_dir)}

    for t in tickers:
        ticker = str(t).upper().strip()
        if not ticker:
            continue

        per = {"ok": False, "errors": [], "artifacts_dir": str(artifacts_dir / ticker)}
        report["tickers"][ticker] = per
        t_art = artifacts_dir / ticker
        t_art.mkdir(parents=True, exist_ok=True)

        try:
            print(f"[{ticker}] start", flush=True)
            filing_dir = _ensure_filing_dir(ticker, base_dir=filings_base_dir, cache_dir=cache_dir)
            html_path = find_primary_document_html(filing_dir)
            company_name = _read_company_name_from_submission(filing_dir) or ticker
            filing_ref = _read_filing_ref(filing_dir)
            sec_doc_url = _sec_doc_url_from_filing_ref(filing_ref)
            cik = int(filing_ref["cik"])

            # Stage: candidates (CACHED)
            candidates = _cached_extract_candidates(html_path, preview_rows=15, preview_cols=10)
            write_candidates_json(candidates, t_art / f"{ticker}_table_candidates.json")
            per["html_path"] = str(html_path)
            per["sec_doc_url"] = sec_doc_url
            income_cand = _pick_income_statement_candidate(candidates)
            per["income_statement_table_id_guess"] = income_cand.table_id if income_cand else None
            _trace_append(t_art, {"stage": "candidates", "n_candidates": len(candidates), "income_guess": per["income_statement_table_id_guess"]})

            # Document scout (CACHED)
            scout = _cached_document_scout(html_path)
            (t_art / "scout.json").write_text(json.dumps(scout, indent=2), encoding="utf-8")
            
            # Snippet retrieval (CACHED)
            snippets = _cached_extract_snippets(
                html_path,
                keywords=[
                    "Item 8",
                    "Financial Statements",
                    "Notes to Financial Statements",
                    "Segment",
                    "Reportable segment",
                    "disaggregation",
                    "revenue by",
                    "net sales",
                ],
                window_chars=2500,
                max_windows=12,
            )
            (t_art / "retrieved_snippets.json").write_text(json.dumps(snippets, indent=2, ensure_ascii=False), encoding="utf-8")
            _trace_append(t_art, {"stage": "scout", "n_headings": len(scout.get("headings", [])), "n_snippets": len(snippets)})

            # Business line discovery (CACHED - single LLM call per filing)
            # Use quality model for better segment identification
            discovery = _cached_discover_business_lines(
                llm_quality, ticker=ticker, company_name=company_name, snippets=snippets, html_path=html_path
            )
            (t_art / "business_lines.json").write_text(json.dumps(discovery, indent=2, ensure_ascii=False), encoding="utf-8")
            _trace_append(t_art, {"stage": "discover", "discovery": discovery})

            segments: List[str] = _canonicalize_business_lines(ticker, list(discovery.get("segments") or []))
            include_optional: List[str] = list(discovery.get("include_segments_optional") or [])
            print(f"[{ticker}] business lines: {segments} (optional={include_optional})", flush=True)

            # Priority-0 deterministic gate: remove common non-target tables (unearned/deferred/RPO/etc.)
            gated: list[TableCandidate] = []
            gated_out: list[dict[str, Any]] = []
            for c in candidates:
                dec = tablekind_gate(c)
                if dec.ok:
                    gated.append(c)
                else:
                    gated_out.append(
                        {
                            "table_id": c.table_id,
                            "reason": dec.reason,
                            "negative_hit": dec.negative_hit,
                            "caption": getattr(c, "caption_text", "")[:160],
                            "heading": getattr(c, "heading_context", "")[:160],
                            "row_labels": (getattr(c, "row_label_preview", []) or [])[:8],
                        }
                    )
            (t_art / "tablekind_gate_rejects.json").write_text(
                json.dumps({"rejected": gated_out[:200]}, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            # If gate removes everything (rare), fall back to all candidates.
            if gated:
                candidates_for_select = gated
            else:
                candidates_for_select = candidates

            # =====================================================================
            # UNIFIED EXTRACTION FLOW
            # Uses extraction module for both AAPL-style and MSFT-style tables
            # =====================================================================
            
            # Select the best revenue table using LLM
            choice = select_revenue_disaggregation_table(
                llm,
                ticker=ticker,
                company_name=company_name,
                candidates=candidates_for_select,
                scout=scout,
                snippets=snippets,
                segments=segments,
                keyword_hints=_keyword_hints_for_ticker(ticker),
            )
            (t_art / "disagg_choice.json").write_text(json.dumps(choice, indent=2, ensure_ascii=False), encoding="utf-8")
            _trace_append(t_art, {"stage": "disagg_select", "choice": choice})
            preferred_table_id = str(choice.get("table_id") or "")
            if not preferred_table_id:
                per["errors"].append("No disaggregation table selected")
                continue
            print(f"[{ticker}] preferred table: {preferred_table_id}", flush=True)

            # Build a ranked fallback list by keyword hits in candidate context
            hints = [h.lower() for h in _keyword_hints_for_ticker(ticker)]
            def _hint_score(c: TableCandidate) -> int:
                blob = " ".join(
                    [
                        " ".join([" ".join(r) for r in (c.preview or [])]),
                        str(getattr(c, "caption_text", "") or ""),
                        str(getattr(c, "heading_context", "") or ""),
                        str(getattr(c, "nearby_text_context", "") or ""),
                    ]
                ).lower()
                return sum(1 for h in hints if h and h in blob)

            ranked_by_hints = sorted(candidates_for_select, key=_hint_score, reverse=True)
            
            # Also find tables that are clearly "revenue by X" tables (high-confidence revenue tables)
            def _is_revenue_breakdown_table(c: TableCandidate) -> tuple:
                """
                Identify tables that show revenue breakdown by product/segment/end market.
                Returns (is_match, is_geography) tuple for prioritization.
                """
                row_labels = getattr(c, "row_label_preview", []) or []
                row_labels_lower = [r.lower() for r in row_labels]
                row_text = " ".join(row_labels_lower)
                
                # Check for geography (lower priority)
                geography_patterns = ["geographic", "geography", "united states", "singapore", "taiwan", "china", "europe"]
                is_geography = any(p in row_text for p in geography_patterns)
                
                # Positive: table heading/title indicates revenue breakdown
                positive_patterns = [
                    "revenue by",
                    "net sales by",
                    "revenue from",
                    "disaggregation of revenue",
                ]
                has_positive = any(p in row_text for p in positive_patterns)
                
                # Also check for "Total revenue" row (indicates revenue aggregation table)
                has_total_revenue = "total revenue" in row_text or "total net sales" in row_text
                
                # Negative: income statement or expense table
                negative_patterns = [
                    "net income", "operating expense", "gross margin", "gross profit",
                    "operating income", "cost of revenue", "research and development",
                    "deferred revenue", "liability", "payable", "accrued"
                ]
                has_negative = any(p in row_text for p in negative_patterns)
                
                is_match = (has_positive or has_total_revenue) and not has_negative
                return (is_match, is_geography)
            
            # Get all revenue tables, prioritize non-geography over geography
            revenue_tables_with_priority = [
                (c.table_id, _is_revenue_breakdown_table(c))
                for c in candidates_for_select
            ]
            # Filter to matches, sort by: non-geography first, then by table_id
            revenue_caption_tables = [
                tid for tid, (is_match, is_geo) in sorted(
                    [(tid, match) for tid, match in revenue_tables_with_priority if match[0]],
                    key=lambda x: (x[1][1], x[0])  # (is_geography, table_id)
                )
                if tid != preferred_table_id
            ]
            
            candidate_table_ids = [preferred_table_id] + revenue_caption_tables + [
                c.table_id for c in ranked_by_hints[:15] 
                if c.table_id != preferred_table_id and c.table_id not in revenue_caption_tables
            ]

            year = None
            seg_totals: Dict[str, int] = {}
            adj_totals: Dict[str, int] = {}
            validation: Optional[ValidationResult] = None
            table_id = ""
            extraction_result: Optional[ExtractionResult] = None
            accepted_table_context: Dict[str, str] = {}

            for attempt_id in candidate_table_ids:
                print(f"[{ticker}] try table {attempt_id} ...", flush=True)
                cand = next((c for c in candidates if c.table_id == attempt_id), None)
                if cand is None:
                    continue
                try:
                    grid = extract_table_grid_normalized(html_path, attempt_id)
                    
                    # Get table metadata for dimension detection
                    table_caption = str(getattr(cand, "caption_text", "") or "")
                    table_heading = str(getattr(cand, "heading_context", "") or "")
                    row_labels = [r[0] if r else "" for r in (cand.preview or [])]
                    
                    # Detect disclosure dimension (product_service, end_market, segment, etc.)
                    dimension = detect_dimension(
                        caption=table_caption,
                        heading=table_heading,
                        row_labels=row_labels,
                        ticker=ticker,
                    )
                    
                    # Get LLM layout hints
                    layout = infer_disaggregation_layout(
                        llm,
                        ticker=ticker,
                        company_name=company_name,
                        table_id=attempt_id,
                        candidate=cand,
                        grid=grid,
                        business_lines=segments,
                    )
                    
                    # Use unified extraction with fallback strategies
                    all_segments = segments + include_optional
                    result = extract_with_layout_fallback(
                        grid,
                        expected_segments=all_segments,
                        llm_layout=layout,
                        ticker=ticker,
                        prefer_granular=True,
                        dimension=dimension,
                        caption=table_caption,
                        heading=table_heading,
                    )
                    
                    if result is None or not result.segment_revenues:
                        _trace_append(t_art, {"stage": "extract_fail", "table_id": attempt_id, "reason": "no_segments"})
                        continue
                    
                    # Validate using self-consistent validation
                    external_total = fetch_companyfacts_total_revenue_usd(cik, result.year)
                    attempt_validation = validate_extraction(
                        segment_revenues=result.segment_revenues,
                        adjustment_revenues=result.adjustment_revenues,
                        table_total=result.table_total,
                        external_total=external_total,
                        tolerance_pct=validation_tolerance_pct,
                    )
                    
                    _trace_append(t_art, {
                        "stage": "extract_attempt",
                        "table_id": attempt_id,
                        "year": result.year,
                        "n_segments": len(result.segment_revenues),
                        "segment_sum": sum(result.segment_revenues.values()),
                        "table_total": result.table_total,
                        "validation_ok": attempt_validation.ok,
                        "validation_notes": attempt_validation.notes,
                    })
                    
                    if not attempt_validation.ok:
                        continue
                    
                    # Accept this table
                    table_id = attempt_id
                    year = result.year
                    seg_totals = result.segment_revenues
                    adj_totals = result.adjustment_revenues
                    validation = attempt_validation
                    extraction_result = result
                    
                    # Save table context for CSV2/CSV3 (needed for product_service dimension)
                    accepted_table_context = {
                        "caption": str(getattr(cand, "caption_text", "") or ""),
                        "heading": str(getattr(cand, "heading_context", "") or ""),
                        "nearby_text": str(getattr(cand, "nearby_text_context", "") or ""),
                    }
                    
                    # Write extraction artifacts
                    (t_art / "disagg_layout.json").write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")
                    extraction_dict = {
                        "year": result.year,
                        "dimension": result.dimension,
                        "rows": [{"segment": r.segment, "item": r.item, "value": r.value, "row_type": r.row_type, "dimension": r.dimension} for r in result.rows],
                        "table_total": result.table_total,
                        "segment_revenues": result.segment_revenues,
                        "adjustment_revenues": result.adjustment_revenues,
                    }
                    (t_art / "disagg_extracted.json").write_text(json.dumps(extraction_dict, indent=2, ensure_ascii=False), encoding="utf-8")
                    print(f"[{ticker}] accepted table {table_id} year={year} dim={result.dimension} segments={len(seg_totals)}", flush=True)
                    break
                    
                except Exception as e:
                    _trace_append(t_art, {"stage": "extract_error", "table_id": attempt_id, "error": str(e)})
                    continue

            if not seg_totals or year is None or validation is None:
                per["errors"].append("No rows extracted from disaggregation table")
                continue

            # Write validation artifact
            validation_dict = {
                "ok": validation.ok,
                "table_total": validation.table_total,
                "segment_sum": validation.segment_sum,
                "adjustment_sum": validation.adjustment_sum,
                "external_total": validation.external_total,
                "delta_pct": validation.delta_pct,
                "notes": validation.notes,
            }
            (t_art / "csv1_validation.json").write_text(json.dumps(validation_dict, indent=2), encoding="utf-8")
            _trace_append(t_art, {"stage": "csv1_validate", "validation": validation_dict, "table_id": table_id})

            # =================================================================
            # CSV1: New schema per csv1_segment_revenue_repo_aligned.md
            # =================================================================
            # Columns: Company Name, Ticker, Revenue Group, Revenue Line,
            #          Line Item description, Revenue (FY{year}, $m)
            
            if extraction_result and extraction_result.rows:
                # Step 1: Filter rows (no adjustments, no totals/subtotals)
                eligible_rows = [
                    r for r in extraction_result.rows
                    if r.row_type not in ("adjustment", "total")
                    and not is_total_row(r.item)
                    and not is_subtotal_row(r.item, ticker)
                    and r.value > 0
                ]
                
                # Step 2: Smart dimension selection to avoid duplication
                # Prefer granular items (revenue_source/product_service) but keep 
                # segment-level rows for segments that have no granular breakdown.
                #
                # Example (META): 
                #   - Family of Apps has granular breakdown (Advertising, Other revenue)
                #     → keep granular items, skip segment total
                #   - Reality Labs has NO granular breakdown
                #     → keep segment-level row
                
                # Group rows by segment
                granular_dims = {"revenue_source", "product_service", "end_market"}
                segments_with_granular = set()
                for r in eligible_rows:
                    if r.dimension in granular_dims and r.segment:
                        segments_with_granular.add(r.segment.lower().strip())
                        # Also check for variations (e.g., "Family of Apps (FoA)" vs "Family of Apps")
                        base_seg = r.segment.split("(")[0].strip().lower()
                        segments_with_granular.add(base_seg)
                
                # Filter: keep granular rows, or segment rows if segment has no granular breakdown
                filtered_rows = []
                for r in eligible_rows:
                    if r.dimension in granular_dims:
                        # Keep granular items
                        filtered_rows.append(r)
                    elif r.dimension == "segment":
                        # Keep segment row only if there's no granular breakdown for this segment
                        seg_lower = r.segment.lower().strip() if r.segment else ""
                        base_seg = seg_lower.split("(")[0].strip()
                        if seg_lower not in segments_with_granular and base_seg not in segments_with_granular:
                            filtered_rows.append(r)
                
                # Step 3: Build revenue lines list for description extraction
                revenue_lines_for_desc = [
                    {"item": r.item, "value": r.value}
                    for r in filtered_rows
                ]
                
                # Step 4: Get line item descriptions (LLM)
                # Use quality model for better company-language descriptions
                print(f"[{ticker}] extracting line item descriptions (LLM quality)...", flush=True)
                html_text = _html_text_for_llm(html_path)
                desc_result = describe_revenue_lines(
                    llm_quality,
                    ticker=ticker,
                    company_name=company_name,
                    fiscal_year=year,
                    revenue_lines=revenue_lines_for_desc,
                    table_context=accepted_table_context,
                    html_text=html_text,
                )
                
                # Create description lookup
                descriptions_by_line = {}
                for row in desc_result.get("rows", []):
                    line = row.get("revenue_line", "")
                    desc = row.get("description", "")
                    if line:
                        descriptions_by_line[line] = desc
                
                # Save descriptions artifact
                (t_art / "csv1_line_descriptions.json").write_text(
                    json.dumps(desc_result, indent=2, ensure_ascii=False), encoding="utf-8"
                )
                
                # Step 5: Build CSV1 rows with new schema
                for r in sorted(filtered_rows, key=lambda x: (-x.value, x.item)):
                    # Clean footnote markers from label
                    clean_label = _clean_revenue_line(r.item)
                    
                    # Determine Revenue Group using mappings
                    revenue_group = _get_revenue_group(ticker, clean_label, r.dimension, r.segment)
                    
                    # Get description
                    description = descriptions_by_line.get(r.item, "")
                    
                    csv1_rows.append(
                        {
                            "Company Name": company_name,
                            "Ticker": ticker,
                            "Fiscal Year": year,
                            "Revenue Group (Reportable Segment)": revenue_group,
                            "Revenue Line": clean_label,
                            "Line Item description (company language)": description,
                            "Revenue ($m)": _to_millions(r.value),
                        }
                    )

            # CSV2 + CSV3 via LLM
            html_text = _html_text_for_llm(html_path)
            seg_names = sorted(seg_totals.keys())
            
            # Extract revenue item names for grounding (from extraction_result)
            revenue_item_names = []
            if extraction_result and extraction_result.rows:
                revenue_item_names = [
                    r.item for r in extraction_result.rows 
                    if r.row_type != "adjustment" and r.item
                ]
            
            # Get the detected dimension for this extraction
            detected_dimension = extraction_result.dimension if extraction_result else "segment"
            
            # CSV2/CSV3 generation (skip if csv1_only mode)
            if not csv1_only:
                # Use quality model for segment descriptions
                print(f"[{ticker}] summarizing segment descriptions (LLM quality)...", flush=True)
                seg_desc = summarize_segment_descriptions(
                    llm_quality,
                    ticker=ticker,
                    company_name=company_name,
                    sec_doc_url=sec_doc_url,
                    html_text=html_text,
                    segment_names=seg_names,
                    revenue_items=revenue_item_names,
                    dimension=detected_dimension,
                    table_context=accepted_table_context,
                )
                (t_art / "csv2_llm.json").write_text(json.dumps(seg_desc, indent=2, ensure_ascii=False), encoding="utf-8")

                for r in (seg_desc.get("rows") or []):
                    csv2_rows.append(
                        {
                            "Company": company_name,
                            "Ticker": ticker,
                            "Segment": r.get("segment", ""),
                            "Segment description": r.get("segment_description", ""),
                            "Key products / services (keywords)": "; ".join(r.get("key_products_services", []) or []),
                            "Primary source": r.get("primary_source", "10-K segment/business description"),
                            "Link": sec_doc_url,
                        }
                    )

                # Use quality model for item extraction
                print(f"[{ticker}] expanding key items per segment (LLM quality)...", flush=True)
                csv3_payload = expand_key_items_per_segment(
                    llm_quality,
                    ticker=ticker,
                    company_name=company_name,
                    sec_doc_url=sec_doc_url,
                    segment_rows=(seg_desc.get("rows") or []),
                    html_text=html_text,
                    dimension=detected_dimension,
                )
                (t_art / "csv3_llm.json").write_text(
                    json.dumps(csv3_payload, indent=2, ensure_ascii=False), encoding="utf-8"
                )

                for r in (csv3_payload.get("rows") or []):
                    csv3_rows.append(
                        {
                            "Company Name": company_name,
                            "Business segment": r.get("segment", ""),
                            "Business item": r.get("business_item", ""),
                            "Description of Business item": r.get("business_item_short_description", ""),
                            "Textual description of the business item- Long form description": r.get(
                                "business_item_long_description", ""
                            ),
                            "Primary source": r.get("primary_source", "10-K segment/business description"),
                            "Link": sec_doc_url,
                        }
                    )

            per["ok"] = True
            per["segment_year"] = year
            per["n_segments"] = len(seg_totals)
            per["validation"] = validation_dict
            print(f"[{ticker}] done", flush=True)

        except Exception as e:
            per["errors"].append(f"{type(e).__name__}: {e}")

    # Write CSVs
    _write_csv(
        out_dir / "csv1_segment_revenue.csv",
        [
            "Company Name",
            "Ticker",
            "Fiscal Year",
            "Revenue Group (Reportable Segment)",
            "Revenue Line",
            "Line Item description (company language)",
            "Revenue ($m)",
        ],
        csv1_rows,
    )
    if not csv1_only:
        _write_csv(
            out_dir / "csv2_segment_descriptions.csv",
            [
                "Company",
                "Ticker",
                "Segment",
                "Segment description",
                "Key products / services (keywords)",
                "Primary source",
                "Link",
            ],
            csv2_rows,
        )
        _write_csv(
            out_dir / "csv3_segment_items.csv",
            [
                "Company Name",
                "Business segment",
                "Business item",
                "Description of Business item",
                "Textual description of the business item- Long form description",
                "Primary source",
                "Link",
            ],
            csv3_rows,
        )

    (out_dir / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _parse_args(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    import argparse

    p = argparse.ArgumentParser(description="Run ReAct revenue segmentation pipeline for tickers.")
    p.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. MSFT,AAPL,...")
    p.add_argument("--model-fast", default="gpt-4.1-mini", help="Fast model for table selection/layout")
    p.add_argument("--model-quality", default="gpt-4.1", help="Quality model for descriptions/discovery")
    p.add_argument("--max-react-iters", type=int, default=3)
    p.add_argument("--out-dir", default="data/outputs")
    p.add_argument("--csv1-only", action="store_true", help="Generate only CSV1 (skip CSV2/CSV3)")
    args = p.parse_args(argv)
    return {
        "tickers": [t.strip().upper() for t in args.tickers.split(",") if t.strip()],
        "model_fast": args.model_fast,
        "model_quality": args.model_quality,
        "max_react_iters": int(args.max_react_iters),
        "out_dir": Path(args.out_dir),
        "csv1_only": args.csv1_only,
    }


if __name__ == "__main__":
    cfg = _parse_args()
    run_pipeline(**cfg)

