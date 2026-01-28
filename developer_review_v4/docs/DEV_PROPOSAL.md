# CSV1 Development Proposal — Post P5 Comprehensive Review

**Date**: January 28, 2026  
**Version**: 3.0  
**Status**: Ready for external review — 95% description coverage achieved

---

## Executive Summary

After completing Phase 5 (META critical fix + table-header rejection), the pipeline has achieved **95% description coverage** across all 6 test tickers. This document provides a systematic comparison against manual human extractions and identifies remaining gaps for reviewer evaluation.

### Performance Summary

| Ticker | Lines | With Description | Coverage | Grade | Key Issue |
|--------|-------|------------------|----------|-------|-----------|
| AAPL | 5 | 5 | **100%** | A | ✅ All correct |
| MSFT | 10 | 10 | **100%** | B+ | Text bleeding between sections |
| GOOGL | 6 | 6 | **100%** | A | ✅ All correct (YouTube ads, Google Network fixed) |
| AMZN | 7 | 7 | **100%** | A | ✅ All footnotes captured |
| META | 3 | 3 | **100%** | A | ✅ P5 fixed all lines |
| NVDA | 6 | 4 | **67%** | C+ | **Compute empty, Networking wrong context** |

**Overall: 35/37 lines = 95%** (2 gaps: NVDA Compute, NVDA OEM and Other)

---

## 1. Detailed Comparison: Pipeline vs. Manual Extraction

### 1.1 AAPL — 100% Match ✅

| Line | Manual Description | Pipeline Description | Match |
|------|-------------------|---------------------|-------|
| iPhone | Smartphones based on iOS | "iPhone is the Company's line of smartphones based on its iOS operating system..." | ✅ Exact |
| Mac | Personal computers based on macOS | "Mac is the Company's line of personal computers based on its macOS operating system..." | ✅ Exact |
| iPad | Multipurpose tablets based on iPadOS | "iPad is the Company's line of multipurpose tablets based on its iPadOS operating system..." | ✅ Exact |
| Wearables, Home | Wearables: watches, headphones; Home: Apple TV, HomePod | "Wearables includes smartwatches, wireless headphones and spatial computers. Home includes Apple TV 4K, HomePod..." | ✅ Exact |
| Services | Advertising, AppleCare, Cloud, Digital Content, Subscriptions, Payment Services | Full breakdown of all 6 sub-categories | ✅ Comprehensive |

**Assessment**: Perfect extraction. Heading-based method working well for AAPL's structured Item 1.

---

### 1.2 AMZN — 100% Match ✅

| Line | Source | Manual | Pipeline | Match |
|------|--------|--------|----------|-------|
| Online stores | Footnote (1) | Physical + digital goods, media products | "...consumable and durable goods that includes media products..." | ✅ |
| Physical stores | Footnote (2) | Sales in physical stores | "Includes sales from Amazon's physical stores..." | ✅ |
| Third-party seller services | Footnote (3) | Commissions, fulfillment fees | "Includes commissions and any related fulfillment and shipping fees..." | ✅ |
| Advertising services | Footnote (4) | Sponsored ads, display, video | "Includes sales of advertising services... through programs such as sponsored ads..." | ✅ |
| Subscription services | Footnote (5) | Prime memberships, digital content | "Includes annual and monthly fees associated with Amazon Prime memberships..." | ✅ |
| AWS | Note 2 | Compute, storage, database services | "amounts earned from global sales of compute, storage, database, and other services..." | ✅ |
| Other | Footnote (6) | Healthcare, licensing, shipping | "sales related to various other offerings, such as healthcare services..." | ✅ |

**Assessment**: Perfect extraction via DOM-based footnote capture (P2 fix).

---

### 1.3 META — 100% Match ✅ (Fixed in P5)

| Line | Manual Source | Manual Description | Pipeline Description | Match |
|------|---------------|-------------------|---------------------|-------|
| Advertising | Note 2 + Item 1 | Revenue from marketers advertising on Facebook, Instagram, Messenger | "Revenue from selling advertising placements on our family of apps to marketers... Ads on our platform enable marketers to reach people across a range of marketing objectives..." | ✅ |
| Other revenue | Note 2 | WhatsApp Business Platform, Meta Verified, Payments | "revenue from WhatsApp Business Platform, Meta Verified subscriptions, net fees we receive from developers using our Payments infrastructure..." | ✅ Exact |
| Reality Labs | Note 2 | Consumer hardware (Meta Quest, Ray-Ban Meta glasses) + software/content | "the delivery of consumer hardware products, such as Meta Quest and Ray-Ban Meta AI glasses, and related software and content." | ✅ Exact |

**Assessment**: P5 Note 2 paragraph extraction working perfectly.

---

### 1.4 GOOGL — 100% Match ✅ (Fixed in P5)

| Line | Manual Source | Manual Description | Pipeline Description | Match |
|------|---------------|-------------------|---------------------|-------|
| Google Search & other | MD&A bullets | Advertising on Google Search, Gmail, Maps, Play | "Google Search & other includes revenues generated on Google search properties...and other Google owned and operated properties like Gmail, Google Maps, and Google Play." | ✅ |
| YouTube ads | MD&A bullets | Advertising on YouTube properties | "YouTube ads includes revenues generated on YouTube properties." | ✅ |
| Google Network | MD&A bullets | AdMob, AdSense, Google Ad Manager | "Google Network includes revenues generated on Google Network properties participating in AdMob, AdSense, and Google Ad Manager." | ✅ |
| Google subscriptions, platforms, devices | MD&A bullets | YouTube services, Google One, Play, Pixel | "...consumer subscriptions...YouTube services, such as YouTube TV, YouTube Music and Premium...platforms...Google Play sales of apps and in-app purchases; devices...Pixel family of devices..." | ✅ |
| Google Cloud | MD&A | GCP + Workspace subscriptions | "Through our Google Cloud Platform and Google Workspace offerings..." | ✅ |
| Other Bets | Item 1 | Healthcare, internet services | "Revenues from Other Bets are generated primarily from the sale of healthcare-related services, and internet services." | ✅ |

**Assessment**: P5 heading-based extraction found all MD&A definitions.

---

### 1.5 MSFT — 100% Coverage, Quality Issues

| Line | Manual Description | Pipeline Description | Issue |
|------|-------------------|---------------------|-------|
| Microsoft 365 Commercial | Microsoft 365 Commercial cloud + on-premises | ✅ Correct core description | ⚠️ Contains "Microsoft 365 Consumer" text (bleeding) |
| Microsoft 365 Consumer | Consumer subscriptions + Office | ✅ Correct core description | ⚠️ Contains "LinkedIn" text (bleeding) |
| LinkedIn | Talent/Marketing/Premium/Sales Solutions | ✅ Correct core description | ⚠️ Contains "Dynamics" text (bleeding) |
| Dynamics | Cloud + on-premises ERP/CRM | ✅ Correct core description | ⚠️ Contains "Competition" text (bleeding) |
| Server products and cloud services | Azure + cloud services | ✅ Excellent Azure description | Clean |
| Enterprise and partner services | Enterprise Support, Industry Solutions | ✅ Correct | ⚠️ Contains "Competition" text |
| Windows and Devices | Windows OEM + Surface | ✅ Correct | Slightly long |
| Gaming | Xbox hardware + Game Pass + cloud | ✅ Correct | Clean |
| Search and news advertising | Bing + Edge + Copilot | ✅ Correct | Clean |
| Other | Cloud-based solutions | ✅ Correct | Clean |

**Root Cause**: MSFT's segment description document uses a continuous prose format. The heading-based extraction captures the right section but doesn't have clear boundary markers between product lines — text from subsequent sections "bleeds" in.

**Severity**: Low — descriptions are correct, just contain extra text.

---

### 1.6 NVDA — 67% Coverage ⚠️ CRITICAL GAP

| Line | Manual Description | Pipeline Description | Match |
|------|-------------------|---------------------|-------|
| **Compute** | GPUs, CPUs, DPUs for AI training/inference | **EMPTY** | ❌ **MISSING** |
| Networking | InfiniBand, Ethernet platforms, network adapters, DPUs, switches | "our Data Center accelerated computing platforms and AI solutions and software; networking; automotive platforms..." | ⚠️ **WRONG** (captured segment description, not Networking-specific) |
| Gaming | GeForce GPUs, GeForce NOW streaming | ✅ Correct | ✅ |
| Professional Visualization | Quadro/RTX GPUs, vGPU, Omniverse | ✅ Correct | ✅ |
| Automotive | DRIVE platform for AV/EV | ✅ Correct | ✅ |
| OEM and Other | (No definition in 10-K) | **EMPTY** | ✅ Expected |

**Root Cause Analysis**:
1. NVDA's revenue table uses "end_market" dimension (Compute, Networking, Gaming...), NOT the segment dimension (Compute & Networking, Graphics)
2. The text describes **segments** (Compute & Networking, Graphics), not individual **end markets**
3. "Compute" as a standalone revenue line item has NO explicit definition in the 10-K
4. The manual extraction synthesized the definition from segment-level text, but this requires understanding that "Compute" = "Data Center accelerated computing platforms"

**Manual Extraction (for Compute)**:
> "Part of NVIDIA's data-center-scale accelerated computing platform for AI (training + inferencing): 'full-stack' compute and networking solutions across processing units, interconnects, systems, and software, with compute explicitly including GPUs, CPUs, and DPUs."

This was synthesized from: `"Compute & Networking segment includes our Data Center accelerated computing platforms and AI solutions and software"`

---

## 2. Revenue Total Reconciliation

| Ticker | Pipeline Sum ($M) | Expected Total ($M) | Delta | Status |
|--------|-------------------|---------------------|-------|--------|
| AAPL | 416,161 | 416,161 | 0% | ✅ Exact |
| AMZN | 637,959 | 637,959 | 0% | ✅ Exact |
| META | 164,501 | 164,501 | 0% | ✅ Exact |
| GOOGL | 349,807 | 349,807 | 0% | ✅ Exact |
| MSFT | 279,317 | 279,317 | 0% | ✅ Exact |
| NVDA | 130,497 | 130,497 | 0% | ✅ Exact |

**Assessment**: All revenue totals reconcile perfectly. Numeric extraction is reliable.

---

## 3. Root Cause Analysis

### 3.1 NVDA Compute Gap — Why It's Hard

The NVDA case exposes a fundamental mismatch:

```
Table Structure (revenue disaggregation):
├── Compute                    ← Revenue line (no definition)
├── Networking                 ← Revenue line (no definition)
├── Gaming                     ← Revenue line (has definition in Graphics segment)
├── Professional Visualization ← Revenue line (has definition in Graphics segment)
├── Automotive                 ← Revenue line (has definition)
└── OEM and Other             ← Revenue line (no definition)

10-K Narrative Structure:
├── Compute & Networking segment includes:
│   ├── "Data Center accelerated computing platforms and AI solutions"
│   ├── "networking"
│   ├── "automotive platforms"
│   └── "DGX Cloud"
└── Graphics segment includes:
    ├── "GeForce GPUs for gaming"
    ├── "Quadro/NVIDIA RTX GPUs for enterprise"
    └── "vGPU software"
```

The problem: "Compute" and "Networking" appear as items in a bulleted list within the segment description, not as standalone headings with their own paragraphs.

### 3.2 MSFT Bleeding — Why It Happens

MSFT's segment descriptions in the 10-K look like:

```
Microsoft 365 Commercial products and cloud services
Microsoft 365 Commercial is an AI-powered business and productivity 
solutions platform... [description continues]

Microsoft 365 Consumer Products and Cloud Services
Microsoft 365 Consumer is designed to increase personal productivity...
```

The heading-based extraction finds "Microsoft 365 Commercial" as a heading, then captures text until it hits a "peer heading". But in MSFT's format:
- "Microsoft 365 Consumer Products and Cloud Services" looks like a continuation, not a peer
- The extraction captures too much text

---

## 4. Specific Questions for Reviewer

### Critical Questions

**Q1: NVDA Compute Definition**
The manual extraction for NVDA Compute is:
> "Part of NVIDIA's data-center-scale accelerated computing platform for AI..."

This was **synthesized** from segment-level text. Should the pipeline:
- A) Accept empty (no explicit definition exists)
- B) Synthesize from segment description (requires LLM inference)
- C) Something else?

**Q2: Revenue Total Validation**
All 6 tickers reconcile to 0% delta. Should we:
- A) Keep current fallback validation (accept if sum is reasonable)
- B) Require hard match to SEC API total (reject otherwise)
- C) Add bounds checking (e.g., 85% ≤ segment_sum/external_total ≤ 115%)

**Q3: MSFT Text Bleeding**
Descriptions contain extra text from neighboring sections. Should we:
- A) Accept as-is (descriptions are still correct, just long)
- B) Implement truncation at product name boundaries
- C) Use sentence count limit (e.g., max 4 sentences)

**Q4: Extraction Priority**
Current priority: Footnote → Heading → Note 2 paragraph → LLM
Should we:
- A) Keep current priority
- B) Add RAG as fallback after Note 2
- C) Other?

### Quality Questions

**Q5: Description Granularity**
For AAPL Services, we capture all 6 sub-categories. For MSFT, we capture product lines individually. Is this the right level?

**Q6: Evidence Traceability**
The provenance artifact tracks source_section and evidence_snippet. Is this sufficient for audit purposes?

---

## 5. Proposed Next Steps

### Tier 1: High Priority (if proceeding)

| Fix | Effort | Impact | Risk |
|-----|--------|--------|------|
| NVDA Compute: Add segment-to-item mapping | 2-3 hrs | +1 line | Medium (may over-extract) |
| NVDA Networking: Fix wrong context capture | 1-2 hrs | +1 quality | Low |
| MSFT bleeding: Add max_chars/sentence limit | 2 hrs | Quality improvement | Low |

### Tier 2: Medium Priority

| Fix | Effort | Impact | Risk |
|-----|--------|--------|------|
| Validation hardening (Phase 6) | 3-4 hrs | Scale safety | Low |
| Add symmetric bounds (85%-115%) | 1 hr | Reject bad tables | Low |
| Label sanity gate (reject >50% numeric) | 1 hr | Already implemented | None |

### Tier 3: Future (if scaling)

| Fix | Effort | Impact | Risk |
|-----|--------|--------|------|
| RAG fallback for missing descriptions | 4-6 hrs | Coverage for edge cases | Medium |
| item7_definitions pseudo-section | 3 hrs | GOOGL-style bullet definitions | Low |

---

## 6. Files for Code Review

The developer review package includes:

### Core Extraction Logic
- `revseg/react_agents.py`:
  - `choose_item_col()` — Deterministic label column selection (lines 47-156)
  - `validate_extracted_labels()` — Label sanity gate (lines 158-177)
  - `infer_disaggregation_layout()` — LLM layout inference (lines 1384+)
  - `_extract_heading_based_definition()` — Item 1 heading extraction (lines 410+)
  - `_extract_note2_paragraph_definition()` — Note 2 prose extraction (lines 430+)
  - `_is_table_header_contaminated()` — Table header rejection (lines 326+)
  - `strip_accounting_sentences()` — Driver/accounting filter (lines 285-304)

### Validation
- `revseg/extraction/validation.py`:
  - `validate_extraction()` — Primary validation logic
  - Table total → External total → Fallback acceptance

### Table Parsing
- `revseg/table_candidates.py`:
  - `_html_to_grid()` — HTML table to grid conversion
  - `extract_table_candidates()` — Candidate table extraction

### RAG (if enabled)
- `revseg/rag/generation.py`:
  - `extract_candidate_products()` — Extractive-first product enumeration
  - `build_rag_query()` — Query construction
  - `generate_description_rag()` — RAG-based description generation

---

## 7. Test Commands

```bash
# Run on all 6 tickers
python -m revseg.pipeline --tickers AAPL,MSFT,GOOGL,AMZN,META,NVDA --out data/outputs --csv1-only

# Run single ticker with verbose output
python -m revseg.pipeline --tickers NVDA --out data/outputs --csv1-only 2>&1 | tee nvda.log

# Check provenance artifacts
cat data/artifacts/NVDA/csv1_desc_provenance.json
```

---

## Sign-off Checklist

- [ ] Revenue totals reconcile (verified: 100%)
- [ ] Description quality acceptable for 95% of lines
- [ ] NVDA Compute gap acceptable or fix required?
- [ ] MSFT bleeding acceptable or fix required?
- [ ] Ready to scale to 100+ tickers?
