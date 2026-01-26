# csv1_segment_revenue (Repo-Aligned Specification)

This document is an **implementation-oriented** specification for generating **csv1_segment_revenue** at scale (thousands of 10‑Ks) using the **current repo flow** in `revseg/`

**Scope:** Only `csv1_segment_revenue`. 

**Goal:** Produce an output structurally consistent with `Example_csv1_segment_revenue.xlsx`:
- **Company Name**
- **Revenue Group (Reportable Segment)**
- **Revenue Line**
- **Line Item description (company language)**
- **Revenue (FY{latest}, $m)**

---

## 1) Definitions 

### 1.1 Revenue Group (Reportable Segment)
A reportable operating segment under ASC 280 (reviewed by CODM; meets thresholds or disclosed voluntarily).

**Repo implication:** Only assign a segment name when the filing **explicitly attributes** a revenue line to a reportable segment (e.g., a table with a segment column, or an explicit mapping table/note).

### 1.2 Revenue Line (Product/Service With Explicit Revenue)
A product/service category with an **explicit numeric revenue** figure disclosed in the 10‑K for the **latest fiscal year**.

**Include only** rows with explicit revenue numbers for latest FY.

**Exclude:**
- qualitative statements without numeric revenue
- KPIs not expressed as revenue (users, subs, units)
- non‑GAAP / pro‑forma “revenue”
- inferred/estimated revenue
- **summary / aggregation rows** (e.g., Total revenue, Net sales, Net product sales, mapping subtotals)

### 1.3 Line Item Description
A concise description of what the revenue line includes, using **company language**; sourced from:
- the segment note / disaggregation note
- table footnotes/captions/nearby text
- MD&A revenue discussion
- Item 1 business overview

**Rule:** Do not infer scope beyond what is disclosed.

---

## 2) Output Contract (CSV1)

### 2.1 Column Schema (Recommended Stable Names)
To avoid dynamic column headers (e.g., “Revenue (FY2024, $m)”), store year in a separate column and write a presentation layer if needed.

**Recommended stable internal schema (CSV):**
- `Company Name`
- `Fiscal Year` (e.g., 2024)
- `Revenue Group (Reportable Segment)`
- `Revenue Group Type` = `segment` | `product_service_disclosure`
- `Revenue Line`
- `Line Item description (company language)`
- `Revenue ($m)`

**Presentation schema (to match Example):**
- same, but replace the last two columns with:
  - `Revenue (FY{Fiscal Year}, $m)`

### 2.2 Normalization
- Currency: USD (retain if disclosed otherwise, but this repo currently assumes USD).
- Units: normalize to **$m** (millions) using `units_multiplier` from layout inference.

---

## 3) Repo Flow (Current “Agent + Deterministic” Pipeline)

Entry point: `revseg/pipeline.py::run_pipeline()`.

The pipeline already implements a robust flow for **(a)** candidate table extraction, **(b)** LLM table selection, **(c)** layout inference, **(d)** deterministic numeric extraction, **(e)** validation, and **(f)** CSV writing.

### 3.1 Stages and Code Anchors (What Exists Today)

1. **Filing acquisition**
   - `revseg/sec_edgar.py`: `download_latest_10k()`
   - Output: `{filing_dir}/primary_document.html`, `filing_ref.json`, `submission.json`

2. **Table candidate extraction**
   - `revseg/table_candidates.py`: `extract_table_candidates_from_html()`
   - Output artifact: `{artifacts}/{TICKER}/{TICKER}_table_candidates.json`

3. **Context orientation**
   - `revseg/react_agents.py`: `document_scout()`
   - `revseg/react_agents.py`: `extract_keyword_windows()`
   - Output artifacts: `scout.json`, `retrieved_snippets.json`

4. **Business line discovery (LLM)**
   - `revseg/react_agents.py`: `discover_primary_business_lines()`
   - Output artifact: `business_lines.json`
   - **Purpose today:** provides a “business_lines” list used as a weak constraint for extraction.

5. **Candidate gating (deterministic)**
   - `revseg/table_kind.py`: `tablekind_gate()`
   - Output artifact: `tablekind_gate_rejects.json`

6. **Revenue table selection (LLM)**
   - `revseg/react_agents.py`: `select_revenue_disaggregation_table()`
   - Output artifact: `disagg_choice.json`

7. **Layout inference (LLM)**
   - `revseg/react_agents.py`: `infer_disaggregation_layout()`
   - Output artifact: `disagg_layout.json`

8. **Unified extraction (deterministic with fallbacks)**
   - `revseg/extraction/core.py`: `detect_dimension()`, row classification
   - `revseg/extraction/__init__.py`: `extract_with_layout_fallback()`
   - Output artifact: `disagg_extracted.json`

9. **Validation**
   - `revseg/extraction/validation.py`: `validate_extraction()`
   - Optional external check: `revseg/validate.py`: `fetch_companyfacts_total_revenue_usd()`
   - Output artifact: `csv1_validation.json`

10. **CSV writing**
   - `pipeline.py` writes `out_dir/csv1_segment_revenue.csv` using the repo’s *current* schema (Year/Dimension/Segment/Item/% etc.)

---

## 4) What Must Change for “Example‑Style” csv1_segment_revenue

The repo currently outputs **a distribution table** (Dimension/Segment/Item/Income/%).  
The Example requires a **granular revenue line table** with:
- reportable segment (where applicable) or entity-wide product/service disclosure label
- revenue line label
- **company-language description** for that line
- revenue amount (latest FY) normalized to $m

### 4.1 Transform Rules (Mapping from `ExtractionResult.rows` → Example Rows)

The unified extractor yields rows with:
- `r.segment` (may be empty)
- `r.item` (row label)
- `r.value` (int; base units using multiplier)
- `r.dimension` (segment/product_service/end_market/revenue_source/etc.)
- `r.row_type` (segment/adjustment/total/item/unknown)

**Rule A — row eligibility**
Include a row if:
- `r.row_type != "adjustment"`
- `r.row_type != "total"`
- row is not a subtotal/summary (see Rule D)
- numeric value exists and is > 0 (if issuer discloses negative revenue, keep but flag)

**Rule B — Revenue Group assignment**
- If `r.segment` is non-empty and the disclosure is segment-scoped:  
  `Revenue Group = r.segment`, `Revenue Group Type = segment`
- Else (no segment label / entity-wide product/service disaggregation):  
  `Revenue Group = "Product/Service disclosure"`, `Revenue Group Type = product_service_disclosure`

**Rule C — Revenue Line**
- `Revenue Line = r.item` (preserve company casing/punctuation as in the table)

**Rule D — Summary/subtotal removal**
Drop rows where `Revenue Line` is an aggregation of other included lines, including (non-exhaustive):
- Total revenue / Total net sales / Net sales / Revenues
- Net product sales / Net service sales
- Segment subtotals that repeat segment names where the table also provides sub-items (e.g., GOOGL “Google Services”)

**Implementation note:** You already have:
- `revseg/mappings.py::is_total_row()` and `is_subtotal_row()`
- ticker-specific subtotal sets (GOOGL, AMZN, NVDA, META)
Generalize this by adding **generic subtotal patterns** and using them even when ticker is unknown.

**Rule E — De-duplication**
If the same `(Revenue Group, Revenue Line, Fiscal Year)` appears multiple times due to multiple accepted tables:
- keep the row from the **most granular** table (prefer product_service > segment)
- keep the row with best validation and highest table score (see §6)

---

## 5) Line Item Description Extraction (New CSV1 Requirement)

CSV1 needs a **per revenue line** description, in company language. The current repo extracts descriptions for segments and business items (CSV2/CSV3), but not for revenue lines.

### 5.1 Deterministic Evidence Retrieval (Recommended)
For each output row `(Revenue Group, Revenue Line)`:

1. Start from the **accepted table candidate context** (already captured in `accepted_table_context` in `pipeline.py`):
   - `caption_text`
   - `heading_context`
   - `nearby_text_context`

2. Retrieve **additional text windows** from the filing around occurrences of:
   - the exact `Revenue Line` label
   - and/or footnote markers near that row label (if present)

**Suggested implementation:**
- Add a deterministic function similar to `extract_keyword_windows()`:
  - `extract_label_windows(html_path, labels=[...], window_chars=2000, max_windows_per_label=3)`
  - Prefer windows inside “Note—Revenue” / “Disaggregation of revenue” / “Net sales by…” sections if detectable.

3. Build an evidence bundle per line:
- `table_context_text` (caption + heading + nearby)
- `label_windows_text` (top N windows)

### 5.2 LLM Summarization (Constrained)
Add a dedicated agent (recommended) to produce `Line Item description (company language)`:

**New agent contract (suggested):**
`describe_revenue_lines(llm, ticker, company_name, fiscal_year, rows, evidence_by_line) -> {rows:[...]}`

**Constraints:**
- 1–2 sentences per line
- must be grounded in provided evidence (no invention)
- prefer exact company phrasing
- if no evidence, return empty string and set low confidence

**Output schema:**
```json
{
  "rows": [
    {
      "revenue_group": "...",
      "revenue_line": "...",
      "line_item_description": "...",
      "evidence_used": "short string (optional)"
    }
  ]
}
```

**Why this is necessary:**  
AMZN-style tables often define categories in footnotes; MSFT/META often define categories in nearby notes/MD&A; descriptions are not reliably derivable without text retrieval + summarization.

---

## 6) Table Selection and Dimension Policy (CSV1-Specific)

Your objective for CSV1 is: **most granular explicit revenue lines**.

### 6.1 Preferred Disclosure Dimension
- Prefer `product_service` disaggregation tables (ASC 280‑10‑50‑38) when present and numeric.
- Else accept segment-level tables or “significant product/service offerings” tables that provide explicit line revenues.

### 6.2 Excluding Geography
The repo already attempts to deprioritize geography, but CSV1 policy should be explicit:

**Hard exclude** tables where `detect_dimension(...) == "geography"` unless:
- the issuer provides **no other** explicit revenue disaggregation in the filing (rare; treat as fallback with explicit flag).

### 6.3 Avoiding Duplicates Like “AWS vs AWS (product/service line)”
This is solved by enforcing **one accepted table** for CSV1 (the repo already selects one table id), and by:
- not re-introducing additional tables into the CSV1 row set post-acceptance
- ensuring the transform step does not create separate synthetic rows

---

## 7) Validation Policy (CSV1)

Validation is used to confirm that the extracted line items are internally coherent.

### 7.1 Existing Validation
`validate_extraction()` compares:
- sum of extracted lines
- optional adjustments
- table total (if found)
- external total (companyfacts) within tolerance

### 7.2 CSV1-Specific Validation Recommendations
Because CSV1 excludes summary rows and often excludes adjustments:
- keep validation on the **raw extraction** (including adjustments if present) to determine table acceptance
- then produce CSV1 output after filtering adjustments/subtotals

Add a second validation artifact:
- `csv1_output_recon.json` containing:
  - sum of output rows
  - the table total (if available)
  - delta and rationale (“expected due to excluded adjustments”)

---

## 8) Implementation Checklist (Concrete Changes in Repo)

### 8.1 Modify CSV1 Writer to Match Example Columns
In `revseg/pipeline.py`, replace the current `csv1_rows.append(...)` blocks with a new builder that outputs:

- `Company Name`
- `Revenue Group (Reportable Segment)`
- `Revenue Line`
- `Line Item description (company language)`
- `Revenue (FY{year}, $m)`  (or stable internal schema per §2.1)

### 8.2 Add a “Revenue Group Type” Internal Field (Optional but Useful)
Even if not in the final example file, keep it internally to support QA and downstream logic:
- `segment` vs `product_service_disclosure`

### 8.3 Add Description Extraction Agent for CSV1
- Implement `extract_label_windows()` (deterministic)
- Implement `describe_revenue_lines()` (LLM)
- Persist artifacts:
  - `{ticker}/csv1_line_evidence.json`
  - `{ticker}/csv1_line_descriptions.json`

### 8.4 Strengthen Summary-Line Filtering (Generic)
Extend `revseg/mappings.py`:
- add generic subtotal/summary patterns
- apply for all tickers (not only those with hardcoded sets)

### 8.5 Keep Existing Caching
The pipeline already uses per-run caches for:
- HTML text
- candidates
- scout
- snippets
- business line discovery
Preserve this—CSV1 at scale depends on caching for cost and latency.

---

## 9) Common Failure Modes and How This Spec Addresses Them

1. **Wrong table selected (e.g., deferred revenue / RPO / contract liabilities)**
   - preserve `tablekind_gate`
   - strengthen table selection prompt to hard-exclude non-revenue tables

2. **Geography tables accidentally selected**
   - enforce `detect_dimension != geography` as hard reject for CSV1 selection

3. **MSFT-style tables lacking an explicit segment column**
   - current repo already uses `mappings.py` for MSFT (item → segment).  
   - generalize by supporting optional mapping when segment_col is null and the filing provides a mapping narrative.

4. **Summary lines included (Net product sales, totals)**
   - enforce Rule D filters + `is_total_row/is_subtotal_row` checks post-extraction

5. **Missing or noisy descriptions**
   - implement evidence windows per revenue line
   - constrain the description agent to evidence only (no invention)

---

## 10) “Definition‑Faithful” Outputs (Operational Criteria)

A CSV1 output is considered correct if:
- every row corresponds to a revenue line with an explicit numeric value for latest FY
- no geographic-only breakdown rows
- no summary / derived aggregates
- revenue group is a reportable segment only when explicitly attributed; otherwise “Product/Service disclosure”
- each description is grounded in filing language or left blank when evidence is missing (never invented)

