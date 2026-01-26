I reviewed the **new output** in `/mnt/data/extraction_results_v5.xlsx` (tabs: `csv1_segment_revenue`, `csv2_segment_descriptions`, `csv3_segment_items`) and the **revised code** in. The repo still uses **`gpt-4.1-mini`** (default in `revseg/llm_client.py`).

## Sign-off decision

**Not signed off yet.** The pipeline is materially closer (Dimension handling is partially implemented), but there are still correctness issues that will cause systematic errors at scale—particularly for **AMZN and META**—and **csv3 items** are not yet reliable.

---

# 1) csv1_segment_revenue: granular revenue extraction and reconciliation

## NVDA — OK for revenue extraction

* `Dimension=end_market` and totals reconcile to **100%**.
* Output is consistent with “Revenue by end market”.

**Status:** OK for csv1.

## AMZN — Still wrong dimension assignment for AWS line

AMZN is extracted from the same table (`t0069`) and includes:

* Online stores, Physical stores, Third-party seller services, Advertising services, Subscription services, Other
* **AWS is present but classified as `Dimension=segment`**, while the rest are `Dimension=product_service`.

**Evidence (from csv1):**

* `product_service` sum = **83.1406%** (6 rows)
* `segment` has only `AWS` = **16.8594%**
* This is a classification bug; AWS is part of the **product/service disaggregation** in that AMZN table.

**Root cause (code):**

* In `revseg/extraction/core.py`, `classify_row_dimension()` uses `SEGMENT_LABEL_PATTERNS` that includes `^\s*aws\s*$`, causing AWS rows to be forced into `segment` even when the **table dimension is product_service**.

**Fix:**

* Make row-level dimension classification conditional:

  * If `table_dimension == "product_service"` and ticker is AMZN, **do not reclassify “AWS”** to segment.
  * Easiest: remove the `^\s*aws\s*$` entry from `SEGMENT_LABEL_PATTERNS`, or gate it to `table_dimension == "segment"`.

**Status:** Not sign-off (AMZN does not reconcile within the intended dimension).

## META — Still missing the segment view (Family of Apps)

META output currently contains:

* `Dimension=revenue_source`: Advertising, Other revenue
* `Dimension=segment`: Reality Labs only
  **Missing:** `Family of Apps` segment revenue.

**Evidence (from csv1):**

* `revenue_source` total = **98.6954%**
* `segment` total = **1.3046%**
* This is not a valid “segment view”; segments should include **Family of Apps and Reality Labs** (and typically sum to 100%).

**Root cause (code):**

* `revseg/mappings.py` defines:

  * `META_SUBTOTAL_ITEMS` includes `"Family of Apps"` and `is_subtotal_row()` drops it.
  * In `revseg/extraction/core.py`, you call `is_subtotal_row(item_label, ticker)` **before** row-dimension classification, so “Family of Apps” is removed even though it is a reportable segment row.

**Fix:**

* Remove `"Family of Apps"` from `META_SUBTOTAL_ITEMS` (it is not a “subtotal to skip”).
* Additionally: in `extract_line_items_granular()` only apply subtotal skipping when the row is truly a subtotal *for the same dimension*. A robust fix is:

  * compute `row_dimension = classify_row_dimension(...)` first
  * only skip subtotals if `row_dimension == dimension` AND it’s not a reportable segment row.

**Status:** Not sign-off (META “segment” extraction is incomplete).

---

# 2) csv2_segment_descriptions: correctness and richness

## AMZN — Descriptions are contaminated (not grounded to category definitions)

**Example problem:**

* AMZN `Segment=AWS` keywords include: **“sponsored ads; display advertising; video advertising”**.
* That is not a faithful description of the AMZN product/service category definition; it reflects leakage from other parts of the filing.

**Root cause (code):**

* `summarize_segment_descriptions()` in `revseg/react_agents.py` builds segment/category snippets by finding occurrences of the label across the full HTML and selecting “best” occurrence using patterns like “segment information” / “notes”.
* This approach works for true reportable segments, but it is **not appropriate for product/service category labels** (e.g., AMZN’s net sales categories). Those should be described from the **table caption + footnotes immediately around the selected revenue table**, not from global text search.

**Fix:**

* When `dimension == "product_service"`:

  * Build `text_snippet` from **the table context** (caption/heading + footnotes around `table_id`, e.g., `t0069`), and optionally the row-footnote marker `(1)…(6)` mapping.
  * Do not use “segment info / note boundaries” heuristics for categories.

**Status:** Not sign-off (descriptions are not reliably grounded for classification).

## META — Dimension mixing persists in csv2

csv2 only includes “Advertising” and “Reality Labs”; it still fails to represent “Family of Apps”. Also “Advertising” is a revenue source, not a segment description entry.

**Fix:**

* csv2 should inherit the same `Dimension` concept as csv1 (or at least be split by dimension), and META should produce:

  * segment descriptions for **Family of Apps** and **Reality Labs**
  * revenue source descriptions for **Advertising** and **Other revenue** (optional, but keep separate)

---

# 3) csv3_segment_items: still not reliable

This is the largest remaining blocker.

## AMZN — Clear cross-category leakage

In csv3, `Business segment = AWS` includes:

* “sponsored ads”
* “video advertising”

These items should belong under “Advertising services,” not AWS.

**Root cause (code):**

* Although you added evidence checking, the evidence is derived from **LLM-produced `segment_description`** and the snippet extraction for categories is already contaminated.
* So the evidence gate is functioning, but it is validating against the wrong slice of text.

**Fix:**

* For csv3, extraction must be **snippet-first, not summary-first**:

  * Use `text_snippet` (ground truth context) as the extraction source.
  * Require evidence spans to be verbatim from that snippet.
  * For product_service dimensions, snippet must come from table-local footnotes (as above).

## NVDA — Items are too sparse / not “most granular”

Current NVDA items list is extremely thin (e.g., Data Center only “Hopper computing platform”; Gaming includes “CUDA programming model”). This is not rich enough for downstream classification.

**Fix:**

* Expand the bounded context for NVDA end markets to include:

  * the “end market” definition section (MD&A) plus the “revenue by end market” note text around the table.
* Adjust the item agent prompt to extract **multiple concrete offerings** per end market (hardware platforms, software suites, services) but still require snippet evidence.

---

# Developer-ready recommendations (targeted to current repo)

## A. Fix dimension logic (csv1)

1. **AMZN AWS classification**

   * Update `SEGMENT_LABEL_PATTERNS` in `revseg/extraction/core.py` to avoid forcing AWS into `segment` when table dimension is `product_service`.

2. **META Family of Apps being dropped**

   * Remove `"Family of Apps"` from `META_SUBTOTAL_ITEMS` in `revseg/mappings.py`.
   * Make subtotal skipping dimension-aware (row_dimension-aware) in `extract_line_items_granular()`.

## B. Make snippet construction dimension-aware (csv2/csv3)

1. In `revseg/react_agents.py::summarize_segment_descriptions()`:

   * If `dimension == product_service`, build snippets from **table-local context** (caption/heading + footnotes around the chosen table id).
   * Stop searching the entire filing for category labels.

2. Propagate `Dimension` into csv2 and csv3 schemas (or store it in the output Excel even if you keep csv format unchanged).

## C. Make csv3 extraction truly “extract-only”

In `expand_key_items_per_segment()`:

* Use `text_snippet` as the source of evidence and extraction, not `segment_description`.
* Reject items unless:

  * item appears in the snippet, and
  * evidence span is verbatim from that snippet and contains the item label.

---

# What I would sign off on today vs not

**Sign-off (partial):**

* NVDA **csv1** revenue extraction (end_market) looks correct and reconciles.

**Not signed off:**

* AMZN (csv1 still dimension-splits AWS; csv2/csv3 contamination remains).
* META (missing Family of Apps in segment view; csv2/csv3 dimension mixing).
* csv3 overall (leakage and insufficiently grounded snippets).

If you want a crisp “Definition of Done” for the developer: the minimum bar is that for each ticker and each extracted dimension, **Income % sums to 100% within that dimension**, and **no csv3 item can be found only outside the segment/category snippet** (i.e., snippet-scoped evidence must pass).
