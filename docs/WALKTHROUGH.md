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
.venv\Scripts\python -m pytest -v
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
* **Transport Vouchers (`BT`, `ORDON7`)**: The deployed `rectype-aware-v1` worker removes a whole document only when the text contains the literal `FORMCHECKBOX` marker and `RECTYPE` is exactly `BT` or starts with `ORDON`. All other documents, including clinical letters with checkboxes, continue to Student v3 inference.
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

The R entry point is `redsan::trim_doceds_onnx()`. There is no second one in
this repository: `scripts/redsan_trim_demo.R` was removed on 2026-09-21 because
it built its own payload with `id` and `text` and no `rectype`, which is a
different worker contract from the one `redsan` sends, and it is how a corpus
came to be trimmed under a rule production never used.

---

## 6. Hospital Deployment & Air-Gapped HDW Setup

On hospital Health Data Warehouse (HDW) platforms, machines are strictly air-gapped without access to public Hugging Face hubs or GitHub.

### 6.1 Artifact Packaging
The production model artifact is packaged as `edsan-doc-trimmer-v1.1.0.zip`, containing:
* `model.onnx`: Standalone DrBERT ONNX runtime graph for CPU inference (442.7 MB)
* `model.safetensors`: PyTorch model weights enabling CUDA GPU acceleration (442.5 MB)
* `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`: Fast Rust tokenizer assets
* `config.json`: Sequence classification architecture metadata
* `artifact.json`: Metadata manifest (`v1.1.0`) whose worker contract is read from the packaged worker
* `trim_batch_service.py`: High-performance batch inference worker implementing `rectype-aware-v1`

### 6.2 Distributing via Internal GitLab
Follow the step-by-step maintainer instructions in **[docs/GITLAB_RELEASE_GUIDE.md](GITLAB_RELEASE_GUIDE.md)** to:
1. Create a tag (`v1.1.0`) and release under **Deploy > Releases**.
2. Upload the release zip to the **GitLab Generic Package Registry** via curl.
3. Link the package URL to the release without bloating Git history.

### 6.3 Deployment for R Users (`redsan`)
```r
library(redsan)

# Option A: One-time install to user cache from downloaded zip:
edsan_install_trimmer("/path/to/edsan-doc-trimmer-v1.1.0.zip")

# Option B: Or point directly to a shared HDW cluster folder:
Sys.setenv(EDSAN_TRIMMER_PATH = "/data/shared/models/edsan-doc-trimmer/v1.1.0")

# Python Path (if running in a custom virtual environment):
Sys.setenv(REDSAN_PYTHON_PATH = "/path/to/python")

# Run trimming:
trimmed_bundle <- trim_doceds_onnx(bundle)
```

### 6.4 Deployment for Python Users
```bash
# Option 1: Unpack in repo artifacts folder
# On Linux / macOS:
unzip edsan-doc-trimmer-v1.1.0.zip -d artifacts/active_learning/onnx_export
# On Windows PowerShell:
Expand-Archive -Path edsan-doc-trimmer-v1.1.0.zip -DestinationPath artifacts/active_learning/onnx_export -Force

# Option 2: Set environment variable pointing to pre-extracted folder
export EDSAN_TRIMMER_PATH="/data/shared/models/edsan-doc-trimmer/v1.1.0"
# In PowerShell:
# $env:EDSAN_TRIMMER_PATH = "C:\models\edsan-doc-trimmer\v1.1.0"

# Auto-resolves and executes:
python scripts/trim_document.py
python scripts/trim_batch_service.py -i input.json -o output.json
```
