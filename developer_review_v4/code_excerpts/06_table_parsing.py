"""
CODE EXCERPT: Table Parsing and Grid Extraction
File: revseg/table_candidates.py (lines 460-600)

PURPOSE: Convert HTML tables to normalized grids, handling iXBRL quirks
like split currency symbols, rowspan/colspan, and hidden rows.
"""

import re
from pathlib import Path
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup


def _table_index_from_id(table_id: str) -> int:
    """Convert table ID (e.g., 't0015') to integer index."""
    if not table_id.startswith("t"):
        raise ValueError(f"Invalid table_id: {table_id}")
    return int(table_id[1:])


def _clean_text(txt: str) -> str:
    """Basic text cleaning for cell content."""
    if not txt:
        return ""
    txt = re.sub(r'\s+', ' ', txt)
    return txt.strip()


def extract_table_grid_normalized(
    html_path: Path,
    table_id: str,
    *,
    max_rows: int = 250,
) -> List[List[str]]:
    """
    Extract a normalized cell grid for a specific table_id.
    
    Unlike the candidate preview, this attempts basic rowspan/colspan handling,
    which is important for downstream deterministic numeric extraction.
    
    iXBRL quirks handled:
    - Split currency symbols ("$" in one cell, "123" in next) → merged
    - Rowspan/colspan → properly expanded
    - Hidden rows (visibility: collapse) → included (may need filtering)
    - Trailing empty cells → trimmed
    """
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    tables = soup.find_all("table")

    idx = _table_index_from_id(table_id)
    if idx < 0 or idx >= len(tables):
        raise IndexError(f"table_id {table_id} out of range: {idx} (n_tables={len(tables)})")
    table = tables[idx]

    rows = table.find_all("tr")
    grid: List[List[str]] = []
    spans: Dict[Tuple[int, int], Tuple[str, int]] = {}  # (r,c) -> (text, remaining_rows)

    for r_i, tr in enumerate(rows[:max_rows]):
        out_row: List[str] = []
        col = 0

        def _drain_span_row(c: int) -> int:
            """Handle rowspan continuation from previous rows."""
            nonlocal out_row
            if (r_i, c) not in spans:
                return c
            txt, remaining = spans[(r_i, c)]
            out_row.append(txt)
            spans.pop((r_i, c), None)
            if remaining > 1:
                spans[(r_i + 1, c)] = (txt, remaining - 1)
            return c + 1

        # Drain spans at start of row
        while (r_i, col) in spans:
            col = _drain_span_row(col)

        for cell in tr.find_all(["th", "td"]):
            while (r_i, col) in spans:
                col = _drain_span_row(col)

            txt = _clean_text(cell.get_text(" ", strip=True))
            rowspan = int(cell.get("rowspan") or 1)
            colspan = int(cell.get("colspan") or 1)
            if colspan < 1:
                colspan = 1
            if rowspan < 1:
                rowspan = 1

            for _ in range(colspan):
                out_row.append(txt)
                if rowspan > 1:
                    spans[(r_i + 1, col)] = (txt, rowspan - 1)
                col += 1

        while (r_i, col) in spans:
            col = _drain_span_row(col)

        # Trim trailing empties
        while out_row and out_row[-1] == "":
            out_row.pop()
        grid.append(out_row)

    # Post-process: merge iXBRL split patterns
    grid = _merge_ixbrl_split_cells(grid)
    
    return grid


def _merge_ixbrl_split_cells(grid: List[List[str]]) -> List[List[str]]:
    """
    Merge common iXBRL split patterns where currency symbols are in separate cells.
    
    Examples:
    - ["$", "123,456"] → ["$123,456"]
    - ["(", "500", ")"] → ["(500)"]
    """
    def _looks_numeric_like(x: str) -> bool:
        if not x:
            return False
        t = x.strip()
        if t in {"-", "—", "–"}:
            return False
        return bool(re.match(r"^\d{1,3}(?:,\d{3})*(?:\.\d+)?$", t) or re.match(r"^\d+(?:\.\d+)?$", t))

    for r in grid:
        if not r:
            continue
        i = 0
        while i < len(r) - 1:
            cur = (r[i] or "").strip()
            nxt = (r[i + 1] or "").strip()
            
            # "$" split column: ["$", "123"] → ["$123"]
            if cur == "$" and _looks_numeric_like(nxt):
                r[i + 1] = "$" + nxt
                r[i] = ""
            
            # Parentheses split: ["(", "500", ")"] → ["(500)"]
            if cur == "(" and _looks_numeric_like(nxt):
                if i + 2 < len(r) and (r[i + 2] or "").strip() == ")":
                    r[i + 1] = f"({nxt})"
                    r[i] = ""
                    r[i + 2] = ""
            
            i += 1
    
    return grid


# =============================================================================
# NUMERIC VALUE PARSING
# =============================================================================

_PAREN_NEG_RE = re.compile(r"\(([^)]+)\)")  # (123) = negative


def parse_money_cell(cell: str, *, units_multiplier: int = 1) -> int:
    """
    Parse a cell value into an integer (in USD, scaled by units_multiplier).
    
    Handles:
    - Parentheses for negatives: "(500)" → -500
    - Currency symbols: "$1,234" → 1234
    - Commas and decimals: "1,234.56" → 1235 (rounded)
    - Dashes for zero: "-" or "—" → 0
    """
    if not cell:
        return 0
    
    txt = cell.strip()
    
    # Handle dash as zero
    if txt in {"-", "—", "–"}:
        return 0
    
    # Check for negative (parentheses)
    neg = False
    m = _PAREN_NEG_RE.match(txt)
    if m:
        neg = True
        txt = m.group(1)
    
    # Strip currency symbols and whitespace
    txt = txt.replace("$", "").replace(",", "").strip()
    
    try:
        val = float(txt)
    except ValueError:
        return 0
    
    if neg:
        val = -val
    
    return int(round(val * units_multiplier))


# =============================================================================
# ROW CLASSIFICATION
# =============================================================================

_TOTAL_ROW_PATTERNS = [
    re.compile(r"^total\s+(?:net\s+)?(?:revenue|sales)", re.IGNORECASE),
    re.compile(r"^total\s+net\s+sales", re.IGNORECASE),
    re.compile(r"^consolidated\s+total", re.IGNORECASE),
]

_SUBTOTAL_PATTERNS = [
    re.compile(r"^google\s+services\s*$", re.IGNORECASE),  # GOOGL subtotal
    re.compile(r"^google\s+advertising\s*$", re.IGNORECASE),  # GOOGL subtotal
    re.compile(r"^total\s+products\s*$", re.IGNORECASE),  # AAPL subtotal
    re.compile(r"^products\s+subtotal\s*$", re.IGNORECASE),
]

_ADJUSTMENT_PATTERNS = [
    re.compile(r"hedge|hedging", re.IGNORECASE),
    re.compile(r"corporate\s+(?:and\s+)?(?:reconciling|unallocated)", re.IGNORECASE),
    re.compile(r"intersegment\s+(?:elimination|revenue)", re.IGNORECASE),
    re.compile(r"reconciling\s+items", re.IGNORECASE),
]


def classify_row_type(label: str) -> str:
    """
    Classify a row label into: 'total', 'subtotal', 'adjustment', or 'item'.
    
    Returns:
        'total' - Final total row (e.g., "Total net revenue")
        'subtotal' - Intermediate subtotal (e.g., "Google Services")
        'adjustment' - Reconciliation item (e.g., "Hedging gains/losses")
        'item' - Regular revenue line item
    """
    label_clean = label.strip()
    
    for pattern in _TOTAL_ROW_PATTERNS:
        if pattern.match(label_clean):
            return "total"
    
    for pattern in _SUBTOTAL_PATTERNS:
        if pattern.match(label_clean):
            return "subtotal"
    
    for pattern in _ADJUSTMENT_PATTERNS:
        if pattern.search(label_clean):
            return "adjustment"
    
    return "item"
