I reviewed the **new `csv1_segment_revenue.csv`** and the **current `revseg.zip`** implementation that computes `Income %`.

## Sign-off status for csv1

**Partial sign-off.**

* **AMZN:** **Sign off** (now correct).
* **NVDA:** **Sign off** (correct).
* **META:** **Not signed off** (percent logic is still wrong when multiple dimensions are output).

---

## What is now correct

### AMZN (Dimension = `product_service`)

* 7 line-items (incl. **Other** and **AWS**) and `Income %` sums to **100.0000** within `product_service`.
* This fixes the prior issues (missing “Other”; AWS misclassified).

### NVDA (Dimension = `end_market`)

* 6 end-market line-items and `Income %` sums to **100.0000**.
* The Data Center split (Compute/Networking) is present and reconciles.

---

## What is still wrong (META)

### META output structure is now *directionally* right

You now have both dimensions present:

* `Dimension=revenue_source`: Advertising, Other revenue
* `Dimension=segment`: Family of Apps, Reality Labs

### But `Income %` is computed against the **combined total across both dimensions**

Current META rows show:

* Advertising **49.1449%**
* Other revenue **0.5268%**
* Family of Apps **49.6717%**
* Reality Labs **0.6566%**

Each dimension sums to ~50%, and **both dimensions together sum to 100%**. That is a clear sign the denominator is wrong, because **dimensions overlap** (each dimension should independently sum to 100% if you are expressing shares).

### Root cause in code (confirmed)

In `revseg/pipeline.py` you compute:

* `product_sum = sum(r.value for r in extraction_result.rows if r.row_type != "adjustment")`
* `total_for_pct = product_sum`

If `extraction_result.rows` contains multiple dimensions (as it now does for META), you are summing them all into a single denominator. That produces the “~50/50 split” behavior.

---

## Required fix (to sign off META)

### Option A (recommended): Percent is **within dimension**

In `pipeline.py`, compute totals per `(ticker, dimension)` and use that as the denominator:

* `pct = r.value / total_by_dimension[r.dimension] * 100`

This will make:

* `revenue_source` rows sum to 100%
* `segment` rows sum to 100%

This matches your requirement that “explicit line items summed to 100%” (per disclosed view).

### Option B: Keep percent as **of company total**, but then you must not output multiple overlapping dimensions in the same “100% expectation”

If you want `% of company total`, then revenue_source and segment will both sum to 100% (each), and you should not combine them into one reconciled set. You’d need a column like:

* `Income % of company total`
  and accept that multiple dimensions are alternative views.

Given your earlier stated expectation (“the output will include all explicit line items summed to 100%”), **Option A is the cleanest**.

---

## Decision

* **AMZN:** signed off for csv1.
* **NVDA:** signed off for csv1.
* **META:** not signed off until you change the `%` denominator logic to be **dimension-aware** (or you explicitly redefine what Income % means and update downstream expectations accordingly).

If you apply the dimension-aware denominator fix, re-run, and share the updated `csv1_segment_revenue.csv`, I can provide a final csv1 sign-off and we move to csv2/csv3.
