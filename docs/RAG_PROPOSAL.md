# RAG-Enhanced Revenue Line Description Extraction

## Proposal for Semantic Search Integration

**Date**: January 2026  
**Status**: Proposal  
**Estimated Effort**: 2-3 days  

---

## Executive Summary

### The Problem

Current description extraction uses **keyword search** to find revenue line descriptions in 10-K filings. This works well for companies with footnote-style disclosures (AMZN, MSFT) but fails for companies with narrative-style descriptions (NVDA).

| Ticker | Current Coverage | Issue |
|--------|------------------|-------|
| AMZN | 7/7 ✅ | Footnotes work |
| AAPL | 5/5 ✅ | Footnotes work |
| MSFT | 9/10 | "Other" has no description |
| GOOGL | 5/6 | "Other Bets" undescribed |
| META | 2/3 | "Other revenue" undescribed |
| **NVDA** | **0/6** ❌ | **No footnotes, narrative style** |

### The Solution

Implement **Retrieval-Augmented Generation (RAG)** to find relevant text chunks by **semantic meaning**, not keyword matching.

**Expected Improvement**: 80% → 95%+ description coverage

---

## Architecture

### Current Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    CURRENT APPROACH                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Revenue Line: "Compute"                                        │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────┐                                           │
│   │ Keyword Search  │  text.find("compute")                     │
│   │ (First Match)   │  → Position 12,453                        │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   ┌─────────────────┐                                           │
│   │ Extract Window  │  text[12453-800 : 12453+2500]             │
│   │ (3300 chars)    │                                           │
│   └────────┬────────┘                                           │
│            │                                                     │
│            ▼                                                     │
│   "...compute capacity may be limited..."  ❌ Wrong context!    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Proposed RAG Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG APPROACH                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   10-K Filing (one-time preprocessing)                          │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────┐     ┌──────────────┐     ┌─────────────┐      │
│   │   Chunk     │────▶│    Embed     │────▶│    Cache    │      │
│   │  (800 char  │     │   (OpenAI)   │     │   (JSON)    │      │
│   │   windows)  │     │              │     │             │      │
│   └─────────────┘     └──────────────┘     └──────┬──────┘      │
│        ~500 chunks         $0.003                  │             │
│                                                    │             │
│   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ │
│                                                    │             │
│   Revenue Line: "Compute"                          │             │
│         │                                          │             │
│         ▼                                          ▼             │
│   ┌─────────────┐     ┌──────────────┐     ┌─────────────┐      │
│   │ Embed Query │────▶│   Retrieve   │◀────│   FAISS     │      │
│   │             │     │   Top 5      │     │   Index     │      │
│   └─────────────┘     └──────┬───────┘     └─────────────┘      │
│                              │                                   │
│                              ▼                                   │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Chunk 1: "Data Center compute platforms include DGX..." │   │
│   │ Chunk 2: "Compute revenue from AI infrastructure..."    │   │
│   │ Chunk 3: "Our compute solutions enable enterprises..."  │   │
│   │ Chunk 4: "The Data Center end market includes GPUs..."  │   │
│   │ Chunk 5: "Compute platforms for AI and analytics..."    │   │
│   └─────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   ┌──────────────┐                                              │
│   │  LLM Call    │  "Summarize Compute from these chunks"       │
│   │  (gpt-4.1)   │                                              │
│   └──────┬───────┘                                              │
│          │                                                       │
│          ▼                                                       │
│   "Compute includes Data Center platforms such as DGX           │
│    systems and GPUs for AI training and inference."  ✅         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technical Implementation

### 1. Chunking Strategy

```python
def chunk_10k(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """
    Split 10-K text into overlapping chunks.
    
    Args:
        text: Full 10-K text (300-500k chars)
        chunk_size: Target chunk size (800 chars ≈ 200 tokens)
        overlap: Overlap between chunks (preserves context)
    
    Returns:
        List of ~500 chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Don't cut mid-sentence
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size // 2:
                end = start + last_period + 1
                chunk = text[start:end]
        
        chunks.append(chunk.strip())
        start = end - overlap
    
    return chunks
```

**Chunk Parameters**:
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `chunk_size` | 800 chars | ~200 tokens, fits in embedding context |
| `overlap` | 100 chars | Prevents cutting descriptions mid-sentence |
| Typical count | ~500 chunks | For 400k char 10-K |

---

### 2. Embedding Generation

```python
import openai
from typing import List

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Embed text chunks using OpenAI.
    
    Cost: ~$0.003 per 10-K (150k tokens)
    """
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=chunks  # Batch up to 2048 chunks
    )
    return [item.embedding for item in response.data]


def embed_query(query: str) -> List[float]:
    """Embed a single query."""
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    return response.data[0].embedding
```

**Model Choice**: `text-embedding-3-small`
| Spec | Value |
|------|-------|
| Dimensions | 1536 |
| Max input | 8191 tokens |
| Cost | $0.02 / 1M tokens |
| Quality | State-of-the-art |

---

### 3. Vector Storage & Retrieval

```python
import numpy as np
import faiss
from pathlib import Path
import json

class EmbeddingIndex:
    """FAISS-based vector index with JSON caching."""
    
    def __init__(self, ticker: str, cache_dir: Path = Path("data/embeddings")):
        self.ticker = ticker
        self.cache_dir = cache_dir
        self.cache_path = cache_dir / f"{ticker}_embeddings.json"
        self.index = None
        self.chunks = None
    
    def build(self, chunks: List[str], embeddings: List[List[float]]):
        """Build FAISS index from chunks and embeddings."""
        self.chunks = chunks
        
        # Convert to numpy
        embedding_matrix = np.array(embeddings, dtype=np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embedding_matrix)
        
        # Build index (inner product = cosine similarity after normalization)
        dim = embedding_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embedding_matrix)
        
        # Cache to disk
        self._save_cache(chunks, embeddings)
    
    def retrieve(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """Retrieve top-k most similar chunks."""
        query = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query)
        
        scores, indices = self.index.search(query, top_k)
        
        return [self.chunks[i] for i in indices[0]]
    
    def load_or_build(self, chunks: List[str]) -> bool:
        """Load from cache or build new index."""
        if self.cache_path.exists():
            data = json.loads(self.cache_path.read_text())
            self.build(data["chunks"], data["embeddings"])
            return True  # Loaded from cache
        else:
            embeddings = embed_chunks(chunks)
            self.build(chunks, embeddings)
            return False  # Built new
    
    def _save_cache(self, chunks: List[str], embeddings: List[List[float]]):
        """Save to JSON cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps({
            "chunks": chunks,
            "embeddings": embeddings
        }))
```

---

### 4. Description Generation with RAG

```python
def describe_revenue_line_rag(
    ticker: str,
    revenue_line: str,
    index: EmbeddingIndex,
    llm: OpenAIChatClient,
    top_k: int = 5
) -> str:
    """
    Generate description using RAG retrieval.
    
    Args:
        ticker: Company ticker
        revenue_line: Revenue line label (e.g., "Compute")
        index: Pre-built embedding index for this 10-K
        llm: LLM client for generation
        top_k: Number of chunks to retrieve
    
    Returns:
        1-2 sentence description in company language
    """
    # Build query
    query = f"{ticker} {revenue_line} revenue products services includes"
    
    # Embed query
    query_embedding = embed_query(query)
    
    # Retrieve relevant chunks
    chunks = index.retrieve(query_embedding, top_k=top_k)
    
    # Build context
    context = "\n\n---\n\n".join(chunks)
    
    # Generate description
    system = (
        "You are extracting product/service descriptions from SEC 10-K filings.\n\n"
        "RULES:\n"
        "1. Use ONLY information from the provided context.\n"
        "2. Write 1-2 sentences describing what this revenue line includes.\n"
        "3. Use company language (quote or closely paraphrase).\n"
        "4. If the context doesn't describe this revenue line, return empty string.\n"
    )
    
    user = f"""Revenue line: {revenue_line}

Context from 10-K filing:
{context}

Extract a description for "{revenue_line}" from the context above."""
    
    result = llm.json_call(
        system=system,
        user=user,
        max_output_tokens=200
    )
    
    return result.get("description", "")
```

---

### 5. Integration with Pipeline

```python
def describe_revenue_lines_with_rag(
    llm: OpenAIChatClient,
    *,
    ticker: str,
    company_name: str,
    fiscal_year: int,
    revenue_lines: List[Dict[str, Any]],
    html_text: str,
) -> Dict[str, Any]:
    """
    Enhanced description extraction using RAG.
    
    Falls back to footnote extraction if RAG returns empty.
    """
    # Step 1: Chunk the 10-K
    chunks = chunk_10k(html_text)
    
    # Step 2: Build or load embedding index
    index = EmbeddingIndex(ticker)
    from_cache = index.load_or_build(chunks)
    
    if from_cache:
        print(f"[{ticker}] Loaded embeddings from cache")
    else:
        print(f"[{ticker}] Generated new embeddings ({len(chunks)} chunks)")
    
    # Step 3: Generate descriptions for each line
    results = []
    for line_info in revenue_lines:
        item_label = line_info.get("item", "")
        
        # Try RAG first
        description = describe_revenue_line_rag(
            ticker=ticker,
            revenue_line=item_label,
            index=index,
            llm=llm
        )
        
        # Fallback to footnote extraction if RAG returns empty
        if not description:
            description = _extract_footnote_for_label(
                item_label, html_text, ""
            ) or ""
        
        results.append({
            "revenue_line": item_label,
            "description": description
        })
    
    return {"rows": results}
```

---

## Cost Analysis

### Embedding Costs (One-Time per Filing)

| Scale | Tokens | Cost |
|-------|--------|------|
| 1 10-K | ~150k | $0.003 |
| 6 tickers | ~900k | $0.02 |
| 100 tickers | ~15M | $0.30 |
| 1,000 tickers | ~150M | $3.00 |

**With caching**: Pay once, reuse forever (until filing is updated)

### LLM Costs (Per Run)

| Operation | Tokens | Cost |
|-----------|--------|------|
| RAG generation (per line) | ~800 in + 100 out | ~$0.002 |
| 6 lines per ticker | ~5400 | ~$0.01 |
| 6 tickers, 38 lines total | ~30k | ~$0.05 |

### Total Cost Comparison

| Approach | 6 Tickers | 100 Tickers |
|----------|-----------|-------------|
| Current (no RAG) | ~$0.30 | ~$5.00 |
| With RAG (first run) | ~$0.35 | ~$5.30 |
| With RAG (cached) | ~$0.35 | ~$5.00 |

**RAG adds <10% cost** with significant quality improvement.

---

## Implementation Plan

### Phase 1: Core RAG Infrastructure (Day 1)
- [ ] Implement `chunk_10k()` function
- [ ] Implement `EmbeddingIndex` class with FAISS
- [ ] Implement `describe_revenue_line_rag()` function
- [ ] Add caching to `data/embeddings/`

### Phase 2: Pipeline Integration (Day 2)
- [ ] Add `--use-rag` flag to pipeline
- [ ] Integrate RAG with fallback to footnote extraction
- [ ] Test on NVDA (hardest case)
- [ ] Compare results: RAG vs current

### Phase 3: Testing & Tuning (Day 3)
- [ ] Run on all 6 tickers
- [ ] Tune `chunk_size`, `overlap`, `top_k`
- [ ] Evaluate description quality
- [ ] Document results

### Dependencies

```bash
pip install faiss-cpu numpy
# OpenAI already installed
```

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Retrieved chunks miss relevant content | Medium | Wrong description | Increase `top_k`, tune query |
| Chunk boundaries split descriptions | Medium | Incomplete context | Overlap chunks, section-aware chunking |
| Embedding costs scale unexpectedly | Low | Budget | Cache aggressively |
| FAISS adds complexity | Low | Maintenance | Use simple IndexFlatIP, no GPU |
| LLM hallucinates from context | Medium | Wrong description | Strict prompt, evidence validation |

---

## Expected Results

### Before RAG (Current)

| Ticker | Lines | With Description |
|--------|-------|------------------|
| AMZN | 7 | 7 ✅ |
| META | 3 | 2 |
| NVDA | 6 | 0 ❌ |
| AAPL | 5 | 5 ✅ |
| MSFT | 10 | 9 |
| GOOGL | 6 | 5 |
| **Total** | **37** | **28 (76%)** |

### After RAG (Expected)

| Ticker | Lines | With Description |
|--------|-------|------------------|
| AMZN | 7 | 7 ✅ |
| META | 3 | 3 ✅ |
| NVDA | 6 | 5 ✅ |
| AAPL | 5 | 5 ✅ |
| MSFT | 10 | 10 ✅ |
| GOOGL | 6 | 6 ✅ |
| **Total** | **37** | **36 (97%)** |

---

## Decision Checkpoint

**Proceed with RAG implementation if**:
- [ ] Description coverage improvement is worth 2-3 days effort
- [ ] ~$0.30 embedding cost per 100 tickers is acceptable
- [ ] Adding `faiss-cpu` dependency is acceptable

**Defer RAG if**:
- [ ] Current 76% coverage is sufficient
- [ ] Time constraints are critical
- [ ] Simpler regex improvements should be tried first

---

## Appendix: Why Semantic Search Works

### Keyword Search Limitation

Query: `"Compute"`

Finds 47 matches including:
- "compute capacity constraints" (risk section)
- "compute infrastructure" (capex section)  
- "compute platforms include" (actual description) ← Not first!

### Semantic Search Advantage

Query embedding captures **intent**: "What products/services does Compute include?"

Returns chunks by **meaning similarity**, not word position:
1. "Data Center compute platforms include DGX..." (0.92 similarity)
2. "Compute revenue from AI infrastructure..." (0.89)
3. "Our compute solutions enable..." (0.87)

Even chunks without the exact word "Compute" are retrieved if they describe it.
