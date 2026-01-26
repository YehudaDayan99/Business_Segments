# Revenue Segmentation Pipeline

## Objective

> **Extract, for a given company's latest 10-K fiscal year, the complete set of revenue line items that are explicitly quantified in the filing and that represent products and/or services, and map each line item to the company's reported operating/business segments, producing a dataset that (a) is traceable to evidence in the filing and (b) reconciles to total revenue under a defined reconciliation policy.**

## Overview

This pipeline extracts revenue segmentation data from SEC 10-K filings using a combination of **LLM agents** and **deterministic extraction**. The goal is to produce structured output showing how a company's revenue breaks down by business segment and product/service line.

### Design Principles

1. **Evidence-based**: All extracted items must be traceable to the filing text
2. **Products/Services only**: Adjustments (hedging, corporate) are excluded from primary output
3. **Reconciliation**: Internal validation ensures extracted items sum to total revenue
4. **Company language**: Descriptions use direct quotes from 10-K footnotes and text

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE FLOW                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   10-K Filing (HTML)                                                         │
│         │                                                                    │
│         ▼                                                                    │
│   ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐              │
│   │   Scout     │────▶│   Discover   │────▶│  Table Select   │              │
│   │  (extract   │     │  (identify   │     │  (find best     │              │
│   │  headings,  │     │  segments)   │     │  revenue table) │              │
│   │  snippets)  │     │              │     │                 │              │
│   └─────────────┘     └──────────────┘     └────────┬────────┘              │
│                                                      │                       │
│         [LLM: gpt-4.1-mini - fast model]            ▼                       │
│                              ┌─────────────────────────────────────┐        │
│                              │         Layout Inference            │        │
│                              │   (identify columns, rows, units)   │        │
│                              └────────────────┬────────────────────┘        │
│                                               │                              │
│                                               ▼                              │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │                  Deterministic Extraction                        │       │
│   │   • Parse table grid (no LLM)                                    │       │
│   │   • Map items to segments (using mappings.py)                    │       │
│   │   • Extract revenue values for target fiscal year                │       │
│   │   • Validate against table total or SEC API                      │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                               │                              │
│                                               ▼                              │
│   ┌─────────────────────────────────────────────────────────────────┐       │
│   │              Description Extraction (Footnotes)                  │       │
│   │   • Extract table footnote definitions (1), (2), etc.            │       │
│   │   • Search 400k chars for "_____" separator + footnotes          │       │
│   │   • Fallback: LLM section search (Item 1, MD&A, Notes)           │       │
│   └─────────────────────────────────────────────────────────────────┘       │
│                                               │                              │
│         [LLM: gpt-4.1 - quality model]       ▼                              │
│   ┌─────────────┐     ┌──────────────┐     ┌─────────────────┐              │
│   │    CSV1     │     │    CSV2      │     │     CSV3        │              │
│   │  (revenue   │     │  (segment    │     │  (detailed      │              │
│   │  + descrip) │     │  descrip.)   │     │   items)        │              │
│   └─────────────┘     └──────────────┘     └─────────────────┘              │
│                        [optional]            [optional]                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 1: How It Works (Intuitive)

### The Problem
10-K filings contain revenue data in HTML tables, but these tables vary widely:
- Different structures (AAPL uses product categories, MSFT uses reportable segments)
- Different locations (Item 1, Item 7, Item 8 / Notes)
- Different formats (iXBRL with split cells, nested tables)
- **Footnotes** contain the descriptions but appear after tables with `___` separators

### The Solution: Agent-Based Approach

**1. Scout Agent** — Scans the document to extract:
- Section headings (to understand document structure)
- Text snippets containing revenue/segment keywords
- All table candidates with metadata (location, preview, numeric density)

**2. Discovery Agent** — Identifies business segments:
- Prompt asks: *"What are the primary business segments or product categories?"*
- Returns segment names (e.g., "Intelligent Cloud", "iPhone", "Google Services")
- Flags optional adjustment lines (e.g., "Corporate", "Hedging")

**3. Table Selection Agent** — Finds the best revenue table:
- Receives ranked table candidates with previews
- Prompt asks: *"Select the table that disaggregates revenue by these segments"*
- Prefers granular tables (product/service level) over segment totals
- Prefers Item 8 / Notes over Item 7 narrative

**4. Layout Inference Agent** — Understands table structure:
- Prompt asks: *"Which column has labels? Which columns have year data? What are the units?"*
- Returns: `item_col=0, year_cols={2024: 15}, units_multiplier=1000000`

**5. Deterministic Extraction** — No LLM, pure code:
- Parses the HTML table into a grid
- Uses `mappings.py` to assign items to segments (e.g., "LinkedIn" → "Productivity and Business Processes")
- Extracts values, handles accounting negatives `(500)`, validates totals

**6. Footnote Extraction** — Company-language descriptions:
- Detects footnote markers in labels: "Online stores (1)", "AWS (2)"
- Searches for `_____` separator followed by `(N) Includes...` pattern
- Extracts verbatim footnote text as description
- Fallback: LLM searches Item 1, MD&A, Notes sections

**7. Description Agents (Optional)** — Enrich with context:
- CSV2: Summarizes each segment from bounded 10-K text
- CSV3: Extracts detailed items with evidence validation

---

## Part 2: Technical Details

### Key Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Main orchestration, loops over tickers, CSV1 output |
| `react_agents.py` | All LLM agent functions + footnote extraction |
| `extraction/core.py` | Deterministic extraction logic |
| `extraction/matching.py` | Fuzzy segment name matching |
| `extraction/validation.py` | Revenue sum validation |
| `mappings.py` | Item-to-segment mappings per company |
| `table_candidates.py` | HTML parsing, table extraction |
| `table_kind.py` | Deterministic gates (reject unearned revenue, etc.) |

### LLM Configuration (Tiered Approach)

| Task | Model | Purpose |
|------|-------|---------|
| Scout, Discover, Table Select, Layout | `gpt-4.1-mini` | High volume, speed |
| Line descriptions (CSV1) | `gpt-4.1` | Quality descriptions |
| Segment descriptions (CSV2) | `gpt-4.1` | Quality summaries |
| Item expansion (CSV3) | `gpt-4.1` | Evidence-based extraction |

### Footnote Extraction Logic

```
1. Check if revenue line has footnote marker: "Online stores (1)"
2. Find label in 10-K text, look for "_____" separator within 3000 chars
3. Extract "(1) Includes..." pattern after separator (up to 600 chars)
4. If not found, search Item 8 section for separator + footnotes
5. Fallback: LLM searches Item 1 → MD&A → Notes → Full text
```

### Table Selection Logic

```
1. Extract all <table> elements from HTML
2. Score by: numeric_ratio, keyword_hits, location (Item 8 preferred)
3. Apply negative gates (reject: unearned revenue, leases, derivatives)
4. LLM selects best match from top 80 candidates
5. If validation fails, retry with next-best candidate (max 3 iterations)
```

### Validation Rules

**Revenue Extraction (internal)**:
```
|segment_sum + adjustment_sum - table_total| / table_total < 2%
```
- Adjustments (hedging, corporate) are included for validation only
- If no table total found, falls back to SEC CompanyFacts API

**CSV3 Evidence Validation**:
- Each item must have `evidence_span` found in source text (70%+ word match)
- OR item name must appear in source text
- Items failing validation are rejected and logged

### Output Schema

**CSV1** (primary output — revenue lines with descriptions):
```
Company Name, Ticker, Fiscal Year, Revenue Group (Reportable Segment), 
Revenue Line, Line Item description (company language), Revenue ($m)
```

Example row:
```csv
AMAZON COM INC,AMZN,2024,Product/Service disclosure,Online stores,
"Includes product sales and digital media content where we record revenue gross...",247029.0
```

**CSV2** (segment descriptions, optional):
```
Company, Ticker, Segment, Segment description, Key products/services, Primary source, Link
```

**CSV3** (detailed items, optional):
```
Company, Ticker, Segment, Business item, Short description, Long description, Evidence span, Link
```

### Adding New Companies

For companies where item-to-segment mapping isn't automatic:

1. Add mapping to `mappings.py`:
```python
NEWCO_ITEM_TO_SEGMENT = {
    "Product A": "Segment 1",
    "Product B": "Segment 2",
}
```

2. Update `get_segment_for_item()` to use it.

For most companies, the LLM agents handle mapping automatically.

---

## Running the Pipeline

```bash
# Single ticker (CSV1 only - fast mode)
python -m revseg.pipeline --tickers MSFT --out-dir data/outputs --csv1-only

# Multiple tickers with all outputs
python -m revseg.pipeline --tickers AAPL,MSFT,GOOGL --out-dir data/outputs

# Custom models
python -m revseg.pipeline --tickers AMZN --model-fast gpt-4.1-mini --model-quality gpt-4.1
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--tickers` | (required) | Comma-separated ticker symbols |
| `--out-dir` | `data/outputs` | Output directory |
| `--csv1-only` | `false` | Skip CSV2/CSV3 to save tokens |
| `--model-fast` | `gpt-4.1-mini` | Model for high-volume tasks |
| `--model-quality` | `gpt-4.1` | Model for quality-critical tasks |
| `--max-react-iters` | `3` | Max retries for table selection |

### Artifacts

Each run produces artifacts in `data/artifacts/{TICKER}/`:
- `scout.json` — Document structure analysis
- `disagg_layout.json` — Inferred table layout
- `disagg_extracted.json` — Raw extraction results
- `csv1_line_descriptions.json` — Footnote/LLM descriptions
- `csv2_llm.json`, `csv3_llm.json` — LLM responses (if not --csv1-only)
- `trace.jsonl` — Full execution trace for debugging

---

## Performance Characteristics

| Ticker Count | Mode | Approx. Time | LLM Calls |
|--------------|------|--------------|-----------|
| 1 | csv1-only | ~20 sec | ~5 |
| 1 | full | ~45 sec | ~8 |
| 6 | csv1-only | ~2 min | ~30 |
| 6 | full | ~4 min | ~48 |

*Times vary based on LLM response latency and 10-K document size.*
