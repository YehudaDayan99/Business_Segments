"""
RAG (Retrieval-Augmented Generation) module for revenue line descriptions.

This module implements Table-Local-First RAG with:
- Non-destructive TOC detection
- DOM-based table context extraction  
- Structure-aware chunking with metadata
- Two-tier FAISS index (table-local + full-filing)
- Threshold calibration
- Evidence gate + extractive-first generation
- QA artifact generation
"""

from .chunking import (
    Chunk,
    detect_toc_regions,
    is_toc_chunk,
    chunk_10k_structured,
    build_table_local_context_dom,
)
from .index import TwoTierIndex, embed_chunks, embed_query
from .generation import (
    DescriptionResult,
    extract_candidate_products,
    check_evidence_gate,
    generate_description_with_evidence,
    describe_revenue_lines_rag,
)
from .calibration import ThresholdCalibrator, CalibrationPair, CALIBRATION_PAIRS
from .qa import CSV1DescCoverage, write_csv1_qa_artifact, run_regression_check, summarize_coverage

__all__ = [
    # Chunking
    "Chunk",
    "detect_toc_regions",
    "is_toc_chunk", 
    "chunk_10k_structured",
    "build_table_local_context_dom",
    # Index
    "TwoTierIndex",
    "embed_chunks",
    "embed_query",
    # Generation
    "DescriptionResult",
    "extract_candidate_products",
    "check_evidence_gate",
    "generate_description_with_evidence",
    "describe_revenue_lines_rag",
    # Calibration
    "ThresholdCalibrator",
    "CalibrationPair",
    "CALIBRATION_PAIRS",
    # QA
    "CSV1DescCoverage",
    "write_csv1_qa_artifact",
    "run_regression_check",
]
