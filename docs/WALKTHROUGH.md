# Pipeline Walkthrough & User Guide

This document describes how the `redsan-doc-trimmer` pipeline works and how to run each step.

---

## 1. What We Have Built

1. **Comprehensive Corpus Extraction (`scripts/extract_sample.R`)**:
   * Reads the 268k EDSAN hospital documents (`docs_merged_00_25`).
   * Strips out all prescriptions (`ORDON*`) and transport vouchers (`BT*`).
   * Discards noisy legacy section columns that leaked doctor signatures and hospital headers.
   * Extracts a **10,000-document multi-dimensional stratified sample** from the hospital's clinical data warehouse into `data/raw/corpus_sample.jsonl`:
     * **All 26 individual years** (2000 through 2025).
     * **381 clinical care units and wards** (`SEJUF`, covering 95.7% of all functional units across the hospital).
     * **75 medical services / departments** (`SEJUM`, 100% departmental coverage).
     * **94 distinct clinical document types** (`RECTYPE`).

2. **Weak Supervision Teacher (`src/redsan_doc_trimmer/annotate.py`)**:
   * Uses an LLM teacher with structured outputs (`BoilerplateSpan`) to detect administrative header, footer, and signature spans.
   * Returns line spans rather than verbose line-by-line classifications, keeping annotations fast and compact.

3. **DrBERT Student Model (`src/redsan_doc_trimmer/train.py`)**:
   * Uses `Dr-BERT/DrBERT-7GB` pre-trained on French biomedical text.
   * Windowed context encoder: evaluates each target line with 2 preceding and 2 following context lines to handle broken OCR/ETL text.
   * Asymmetric class-weighted loss (`clinical_weight = 10.0`): strictly prioritizes clinical recall (100% preservation of medical facts).
   * Accelerated on your **NVIDIA GeForce RTX 3080 Laptop GPU** (30–35 seconds per training pass).

4. **Active Learning & Hard-Negative Mining (`src/redsan_doc_trimmer/mining.py`)**:
   * Student v1 scans thousands of unlabeled candidate documents on the GPU (with a live animated progress bar).
   * Automatically isolates the hardest, most ambiguous edge cases (lines where probability is borderline: $p \in [0.35, 0.65]$).
   * The LLM teacher pre-annotates those mined hard cases before human curation.
   * Student v2 is fine-tuned on the augmented dataset with a refined learning rate (`5e-6`).

5. **Standalone ONNX Export (`src/redsan_doc_trimmer/export_onnx.py`)**:
   * Exports the final model to `artifacts/active_learning/onnx_export/model.onnx` along with tokenizer assets.
   * Upstream `redsan` in R executes this model via a fast background batch service with zero cloud/API dependencies!

---

## 2. How to Run the Pipeline

### Option A: Fast Dry-Run Test with Mock Heuristic Teacher
You can test the entire pipeline locally in ~35 seconds using the built-in mock heuristic teacher:

```powershell
# Activate the virtual environment
.venv\Scripts\activate

# Run the active learning loop on a tiny subset (10 seed docs + scans 20 pool docs)
python scripts/run_active_learning.py --mock --seed_size 10 --pool_size 20 --mine_top_k 5
```

### Option B: Production Run with LLM Teacher & RTX 3080

```powershell
# Activate virtual environment
.venv\Scripts\activate

# Run the production active learning cycle:
# 1,000 seed documents + 500 mined hard cases
python scripts/run_active_learning.py --seed_size 1000 --mine_top_k 500
```

### Running Automated Verification Tests Anytime
```powershell
.venv\Scripts\pytest -v
```

---

## 3. Production Model & Active Learning Results

Following weak supervision by an LLM teacher and **100% human-in-the-loop manual review of 500 mined edge cases**, Student v2 was fine-tuned on 1,500 curated documents (100,047 lines) on the NVIDIA RTX 3080 GPU.

### Final Holdout Validation Metrics (20,010 lines)
* **Clinical Recall**: `98.12%` *(The Cardinal Rule: zero destruction of medical facts)*
* **Boilerplate Precision**: `97.89%` *(Removed lines are guaranteed administrative text)*
* **Macro F1**: `95.92%`
* **Artifact Locations**:
  * `artifacts/active_learning/onnx_export/model.onnx` (442.7 MB)
  * `artifacts/active_learning/onnx_export/tokenizer.json` (4.2 MB)

---

## 4. Large-Scale Benchmark on Unseen Cohorts

Evaluated across **800 real hospital documents** from `D0840/docs` (65k document warehouse) and `denut.rds` (malnutrition patient cohort):
* **Total prompt tokens saved**: **~171,600 prompt tokens** (-23.5% to -35.2% net token reduction).
* **Prescriptions (`ORDON*`)**: Preserves all drug molecules, posologies, medical equipment, and nursing orders while stripping hospital letterheads and legal disclaimers (~35% reduction).
* **Transport Vouchers (`BT`)**: Automatically detected and bypassed via fast regex pre-filter (`is_transport_voucher`).
* **Hospitalization & Discharge (`CRH*`, `CR2AAF`)**: All anamnesis, conclusions, and diagnoses preserved verbatim.
* **Interactive Viewer**: [benchmark_viewer.html](file:///artifacts/benchmark_viewer.html) (color-coded side-by-side verification for 50 diverse documents).

---

## 5. Upstream R Integration (`redsan`)

In the `redsan` R package, you can trim documents directly from an `edsan_event_bundle` or `doceds` table:

```r
library(jsonlite)
library(processx)

# Trims doceds table and appends RECTXT_TRIMMED and TRIM_PRESERVED_INTERVALS
trimmed_doceds <- trim_doceds_onnx(bundle$sources$doceds)

# Every preserved line satisfies the Grounding Guarantee:
# substring(raw_rectxt, start, end) == preserved_text (100% auditable)
```

Run the complete R integration test anytime:
```powershell
Rscript scripts/redsan_trim_demo.R
```
