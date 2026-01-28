"""
CODE EXCERPT: Layout Inference - Item Column Selection
File: revseg/react_agents.py (lines 47-156)

PURPOSE: Deterministically select the best label/item column in a table grid.
This overrides LLM column selection when the LLM picks a numeric column.
"""

import re
from typing import List, Optional, Tuple

# Pattern to detect currency/numeric values
_CURRENCY_NUM_RE = re.compile(
    r'^[($\s]*[\d,]+(?:\.\d+)?[)\s]*$|'  # Numbers with optional $ and parens
    r'^[($\s]*-[)\s]*$'  # Dash for negative/zero
)


def choose_item_col(
    grid: List[List[str]],
    header_rows: Optional[List[int]] = None,
    llm_proposed_col: Optional[int] = None,
) -> Tuple[int, str]:
    """
    Deterministically select the best label/item column in a table grid.
    
    Ranks columns by:
    1. numeric_ratio (lower = better for label column)
    2. alpha_ratio (higher = better)
    3. uniqueness (labels tend to be diverse)
    4. mean string length (labels typically > 2 chars)
    
    Args:
        grid: Table grid as list of lists
        header_rows: Row indices to skip (headers)
        llm_proposed_col: Column proposed by LLM (will validate)
    
    Returns:
        (best_col_index, reason_string)
    """
    if not grid:
        return (0, "empty grid, defaulting to 0")
    
    header_rows = set(header_rows or [])
    
    # Compute metrics for each column
    col_scores = []
    n_cols = max(len(row) for row in grid) if grid else 0
    
    for col_idx in range(n_cols):
        cells = []
        for row_idx, row in enumerate(grid):
            if row_idx in header_rows:
                continue
            if col_idx < len(row):
                cell = str(row[col_idx]).strip()
                if cell:
                    cells.append(cell)
        
        if not cells:
            col_scores.append({
                "col": col_idx,
                "numeric_ratio": 1.0,
                "alpha_ratio": 0.0,
                "uniqueness": 0.0,
                "mean_len": 0.0,
                "score": -999,
            })
            continue
        
        # Metric 1: numeric_ratio (lower = better for label column)
        numeric_count = sum(1 for c in cells if _CURRENCY_NUM_RE.match(c))
        numeric_ratio = numeric_count / len(cells)
        
        # Metric 2: alpha_ratio (contains letters, higher = better)
        alpha_count = sum(1 for c in cells if any(ch.isalpha() for ch in c))
        alpha_ratio = alpha_count / len(cells)
        
        # Metric 3: uniqueness (unique values / total, higher = better for labels)
        unique_values = len(set(c.lower() for c in cells))
        uniqueness = unique_values / len(cells) if cells else 0
        
        # Metric 4: mean string length (labels tend to be longer than numbers)
        mean_len = sum(len(c) for c in cells) / len(cells) if cells else 0
        
        # Combined score: 
        # - Penalize high numeric_ratio heavily (-5x weight)
        # - Reward alpha_ratio (+3x weight)
        # - Reward uniqueness slightly (+1x weight)
        # - Reward reasonable length (+0.05x weight)
        score = (-5 * numeric_ratio) + (3 * alpha_ratio) + (1 * uniqueness) + (0.05 * mean_len)
        
        col_scores.append({
            "col": col_idx,
            "numeric_ratio": round(numeric_ratio, 3),
            "alpha_ratio": round(alpha_ratio, 3),
            "uniqueness": round(uniqueness, 3),
            "mean_len": round(mean_len, 1),
            "score": round(score, 3),
        })
    
    if not col_scores:
        return (0, "no columns found, defaulting to 0")
    
    # Sort by score descending
    col_scores.sort(key=lambda x: x["score"], reverse=True)
    heuristic_best = col_scores[0]
    
    # Validate LLM's proposed column
    if llm_proposed_col is not None and 0 <= llm_proposed_col < len(col_scores):
        llm_col_data = next((c for c in col_scores if c["col"] == llm_proposed_col), None)
        
        if llm_col_data:
            # Accept LLM choice if:
            # 1. numeric_ratio < 0.5 (not mostly numbers)
            # 2. alpha_ratio > 0.3 (has some text)
            if llm_col_data["numeric_ratio"] < 0.5 and llm_col_data["alpha_ratio"] > 0.3:
                return (llm_proposed_col, f"LLM choice validated (num={llm_col_data['numeric_ratio']}, alpha={llm_col_data['alpha_ratio']})")
            else:
                # LLM choice failed validation, override with heuristic best
                return (
                    heuristic_best["col"],
                    f"LLM col {llm_proposed_col} OVERRIDDEN (num={llm_col_data['numeric_ratio']:.2f}, alpha={llm_col_data['alpha_ratio']:.2f}) → col {heuristic_best['col']} (score={heuristic_best['score']:.2f})"
                )
    
    # No LLM proposal or invalid index, use heuristic
    return (heuristic_best["col"], f"heuristic best (score={heuristic_best['score']:.2f})")
