"""
Threshold calibration for RAG retrieval.

Calibrates retrieval thresholds using known-good query-chunk pairs,
setting thresholds at P90 of noise scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .index import TwoTierIndex


@dataclass
class CalibrationPair:
    """Known-good query-chunk pair for calibration."""
    ticker: str
    query: str
    expected_chunk_text: str  # Substring that should be in top results
    source: str  # "footnote", "table_local", "item1", "note_segment", etc.


# Known-good pairs from manual inspection
CALIBRATION_PAIRS = [
    # AMZN footnotes (should score high on table-local)
    CalibrationPair(
        ticker="AMZN",
        query="AMZN FY2024 revenue line 'Online stores' products services included",
        expected_chunk_text="product sales and digital media content",
        source="footnote"
    ),
    CalibrationPair(
        ticker="AMZN",
        query="AMZN FY2024 revenue line 'Third-party seller services' products services",
        expected_chunk_text="commissions and any related fulfillment and shipping fees",
        source="footnote"
    ),
    CalibrationPair(
        ticker="AMZN",
        query="AMZN FY2024 revenue line 'AWS' Amazon Web Services cloud products",
        expected_chunk_text="global sales of compute, storage, database",
        source="item1"
    ),
    
    # MSFT segment notes (should score high on note_segment)
    CalibrationPair(
        ticker="MSFT",
        query="MSFT FY2025 revenue line 'Intelligent Cloud' products services Azure",
        expected_chunk_text="Azure",
        source="note_segment"
    ),
    CalibrationPair(
        ticker="MSFT",
        query="MSFT FY2025 revenue line 'Productivity and Business Processes' Office Microsoft 365",
        expected_chunk_text="Office",
        source="note_segment"
    ),
    
    # AAPL footnotes
    CalibrationPair(
        ticker="AAPL",
        query="AAPL FY2025 revenue line 'iPhone' products smartphone",
        expected_chunk_text="iPhone",
        source="item1"
    ),
    CalibrationPair(
        ticker="AAPL",
        query="AAPL FY2025 revenue line 'Services' App Store AppleCare iCloud",
        expected_chunk_text="App Store",
        source="item1"
    ),
    
    # GOOGL
    CalibrationPair(
        ticker="GOOGL",
        query="GOOGL FY2024 revenue line 'Google Search' advertising",
        expected_chunk_text="Google Search",
        source="item1"
    ),
    CalibrationPair(
        ticker="GOOGL",
        query="GOOGL FY2024 revenue line 'Google Cloud' cloud services platform",
        expected_chunk_text="Google Cloud Platform",
        source="note_segment"
    ),
    
    # META
    CalibrationPair(
        ticker="META",
        query="META FY2024 revenue line 'Advertising' Facebook Instagram",
        expected_chunk_text="advertising",
        source="item1"
    ),
    
    # NVDA (hardest case - narrative style)
    CalibrationPair(
        ticker="NVDA",
        query="NVDA FY2025 revenue line 'Compute' Data Center GPU DGX",
        expected_chunk_text="Data Center",
        source="item1"
    ),
    CalibrationPair(
        ticker="NVDA",
        query="NVDA FY2025 revenue line 'Gaming' GeForce graphics cards",
        expected_chunk_text="GeForce",
        source="item1"
    ),
]


class ThresholdCalibrator:
    """
    Calibrate retrieval thresholds using known-good pairs.
    
    Method:
    1. For each known-good pair, compute score of expected chunk
    2. Compute score distribution of "noise" chunks (non-matching)
    3. Set threshold at P{noise_percentile} of noise scores
    """
    
    def __init__(self, noise_percentile: float = 90):
        """
        Args:
            noise_percentile: Percentile of noise scores to use as threshold.
                             Higher = more selective (fewer false positives).
                             Default 90 means we reject 90% of noise.
        """
        self.noise_percentile = noise_percentile
    
    def calibrate(
        self,
        index: 'TwoTierIndex',
        pairs: List[CalibrationPair],
        embed_func
    ) -> Tuple[float, float]:
        """
        Calibrate thresholds using known-good pairs.
        
        Args:
            index: Built TwoTierIndex to calibrate against
            pairs: List of CalibrationPair objects
            embed_func: Function to embed query strings
        
        Returns:
            (local_threshold, global_threshold)
        """
        local_good_scores = []
        local_noise_scores = []
        global_good_scores = []
        global_noise_scores = []
        
        # Filter pairs for this ticker
        ticker_pairs = [p for p in pairs if p.ticker == index.ticker]
        
        if not ticker_pairs:
            # No calibration data - use defaults
            return 0.70, 0.60
        
        for pair in ticker_pairs:
            try:
                query_emb = embed_func(pair.query)
            except Exception as e:
                print(f"[Calibration] Failed to embed query: {e}")
                continue
            
            query = np.array([query_emb], dtype=np.float32)
            
            # Normalize query
            import faiss
            faiss.normalize_L2(query)
            
            # Check local index
            if index.local_index is not None and index.local_index.ntotal > 0:
                k = min(50, index.local_index.ntotal)
                scores, indices = index.local_index.search(query, k)
                
                for idx, score in zip(indices[0], scores[0]):
                    if idx < 0 or idx >= len(index.local_chunks):
                        continue
                    chunk = index.local_chunks[idx]
                    
                    if pair.expected_chunk_text.lower() in chunk.text.lower():
                        local_good_scores.append(float(score))
                    else:
                        local_noise_scores.append(float(score))
            
            # Check full index
            if index.full_index is not None and index.full_index.ntotal > 0:
                k = min(100, index.full_index.ntotal)
                scores, indices = index.full_index.search(query, k)
                
                for idx, score in zip(indices[0], scores[0]):
                    if idx < 0 or idx >= len(index.full_chunks):
                        continue
                    chunk = index.full_chunks[idx]
                    
                    if pair.expected_chunk_text.lower() in chunk.text.lower():
                        global_good_scores.append(float(score))
                    else:
                        global_noise_scores.append(float(score))
        
        # Compute thresholds
        local_threshold = self._compute_threshold(
            local_good_scores, local_noise_scores, default=0.70
        )
        global_threshold = self._compute_threshold(
            global_good_scores, global_noise_scores, default=0.60
        )
        
        return local_threshold, global_threshold
    
    def _compute_threshold(
        self,
        good_scores: List[float],
        noise_scores: List[float],
        default: float
    ) -> float:
        """Compute threshold from score distributions."""
        if not noise_scores:
            return default
        
        # P{noise_percentile} of noise scores
        noise_threshold = np.percentile(noise_scores, self.noise_percentile)
        
        # Ensure threshold doesn't exceed minimum good score
        if good_scores:
            min_good = min(good_scores)
            # Threshold should be below minimum good score (with margin)
            noise_threshold = min(noise_threshold, min_good - 0.05)
        
        # Ensure reasonable bounds
        return max(0.45, min(0.85, noise_threshold))


def calibrate_index(
    index: 'TwoTierIndex',
    embed_func,
    pairs: List[CalibrationPair] = None
) -> Tuple[float, float]:
    """
    Convenience function to calibrate an index.
    
    Args:
        index: TwoTierIndex to calibrate
        embed_func: Function to embed queries
        pairs: Calibration pairs (defaults to CALIBRATION_PAIRS)
    
    Returns:
        (local_threshold, global_threshold)
    """
    if pairs is None:
        pairs = CALIBRATION_PAIRS
    
    calibrator = ThresholdCalibrator(noise_percentile=90)
    return calibrator.calibrate(index, pairs, embed_func)
