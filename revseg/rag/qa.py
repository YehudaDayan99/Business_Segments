"""
QA artifact generation for CSV1 description coverage.

Features:
- Per-ticker coverage metrics
- Missing label tracking
- Retrieval tier breakdown
- Regression detection
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Any, Optional
from pathlib import Path

from .generation import DescriptionResult


@dataclass
class CSV1DescCoverage:
    """QA artifact for CSV1 description coverage."""
    ticker: str
    fiscal_year: int
    total_lines: int
    lines_with_description: int
    coverage_pct: float
    missing_labels: List[str]
    tier1_count: int = 0  # Lines from table-local tier
    tier2_count: int = 0  # Lines from full-filing tier
    failed_gate_count: int = 0  # Lines that failed evidence gate
    validation_failed_count: int = 0  # Lines that failed quote validation
    line_details: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_coverage(
    ticker: str,
    fiscal_year: int,
    results: List[DescriptionResult]
) -> CSV1DescCoverage:
    """
    Compute coverage metrics from description results.
    
    Args:
        ticker: Company ticker
        fiscal_year: Fiscal year
        results: List of DescriptionResult from RAG generation
    
    Returns:
        CSV1DescCoverage with detailed metrics
    """
    total = len(results)
    with_desc = sum(1 for r in results if r.description)
    missing = [r.revenue_line for r in results if not r.description]
    
    # Count by tier
    tier1 = sum(1 for r in results if r.retrieval_tier == "tier1_local" and r.description)
    tier2 = sum(1 for r in results if r.retrieval_tier == "tier2_full" and r.description)
    failed_gate = sum(1 for r in results if not r.evidence_gate_passed)
    validation_failed = sum(1 for r in results if not r.validated and r.evidence_gate_passed)
    
    # Per-line details
    line_details = []
    for r in results:
        line_details.append({
            "revenue_line": r.revenue_line,
            "has_description": bool(r.description),
            "retrieval_tier": r.retrieval_tier,
            "evidence_gate_passed": r.evidence_gate_passed,
            "validated": r.validated,
            "top_chunk_ids": r.evidence_chunk_ids[:3],
            "products_found": len(r.products_services_list),
            "description_preview": r.description[:100] if r.description else "",
        })
    
    return CSV1DescCoverage(
        ticker=ticker,
        fiscal_year=fiscal_year,
        total_lines=total,
        lines_with_description=with_desc,
        coverage_pct=round(100 * with_desc / total, 1) if total > 0 else 0.0,
        missing_labels=missing,
        tier1_count=tier1,
        tier2_count=tier2,
        failed_gate_count=failed_gate,
        validation_failed_count=validation_failed,
        line_details=line_details
    )


def write_csv1_qa_artifact(
    ticker: str,
    fiscal_year: int,
    results: List[DescriptionResult],
    output_dir: Path
) -> CSV1DescCoverage:
    """
    Write csv1_desc_coverage.json for regression testing.
    
    Example output:
    {
        "ticker": "NVDA",
        "fiscal_year": 2025,
        "total_lines": 6,
        "lines_with_description": 5,
        "coverage_pct": 83.3,
        "missing_labels": ["OEM and Other"],
        "tier1_count": 2,
        "tier2_count": 3,
        "failed_gate_count": 1,
        "validation_failed_count": 0,
        "line_details": [...]
    }
    
    Args:
        ticker: Company ticker
        fiscal_year: Fiscal year
        results: List of DescriptionResult from RAG generation
        output_dir: Directory to write artifact
    
    Returns:
        CSV1DescCoverage object
    """
    coverage = compute_coverage(ticker, fiscal_year, results)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write artifact
    output_path = output_dir / f"{ticker}_csv1_desc_coverage.json"
    output_path.write_text(
        json.dumps(coverage.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    
    return coverage


def load_coverage_artifact(path: Path) -> Optional[CSV1DescCoverage]:
    """Load a coverage artifact from disk."""
    if not path.exists():
        return None
    
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CSV1DescCoverage(**data)
    except Exception as e:
        print(f"Failed to load coverage artifact: {e}")
        return None


def run_regression_check(
    current: CSV1DescCoverage,
    baseline: CSV1DescCoverage
) -> Dict[str, Any]:
    """
    Compare current run against baseline for regression.
    
    Args:
        current: Coverage from current run
        baseline: Coverage from baseline run
    
    Returns:
        Dict with:
        - coverage_delta: change in coverage %
        - new_missing: labels that were covered but now missing
        - new_covered: labels that were missing but now covered
        - regression: True if quality degraded
    """
    baseline_covered = set(
        d["revenue_line"] for d in baseline.line_details if d["has_description"]
    )
    current_covered = set(
        d["revenue_line"] for d in current.line_details if d["has_description"]
    )
    
    new_missing = list(baseline_covered - current_covered)
    new_covered = list(current_covered - baseline_covered)
    
    return {
        "ticker": current.ticker,
        "coverage_delta": current.coverage_pct - baseline.coverage_pct,
        "baseline_coverage": baseline.coverage_pct,
        "current_coverage": current.coverage_pct,
        "new_missing": new_missing,
        "new_covered": new_covered,
        "regression": len(new_missing) > 0,
        "improvement": len(new_covered) > len(new_missing)
    }


def summarize_coverage(coverages: List[CSV1DescCoverage]) -> Dict[str, Any]:
    """
    Summarize coverage across multiple tickers.
    
    Args:
        coverages: List of CSV1DescCoverage objects
    
    Returns:
        Summary dict with aggregate metrics
    """
    if not coverages:
        return {
            "total_tickers": 0,
            "total_lines": 0,
            "total_with_desc": 0,
            "overall_coverage_pct": 0.0,
            "per_ticker": []
        }
    
    total_lines = sum(c.total_lines for c in coverages)
    total_with_desc = sum(c.lines_with_description for c in coverages)
    overall_pct = round(100 * total_with_desc / total_lines, 1) if total_lines > 0 else 0.0
    
    per_ticker = [
        {
            "ticker": c.ticker,
            "coverage_pct": c.coverage_pct,
            "lines": c.total_lines,
            "with_desc": c.lines_with_description,
            "missing": c.missing_labels
        }
        for c in coverages
    ]
    
    return {
        "total_tickers": len(coverages),
        "total_lines": total_lines,
        "total_with_desc": total_with_desc,
        "overall_coverage_pct": overall_pct,
        "tier1_total": sum(c.tier1_count for c in coverages),
        "tier2_total": sum(c.tier2_count for c in coverages),
        "failed_gate_total": sum(c.failed_gate_count for c in coverages),
        "per_ticker": per_ticker
    }
