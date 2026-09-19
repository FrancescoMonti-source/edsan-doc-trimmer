# Pipeline Walkthrough & User Guide

This document describes how the `redsan-doc-trimmer` pipeline works and how to run each step.

---

## 1. What We Have Built

1. **Comprehensive Corpus Extraction (`scripts/extract_sample.R`)**:
   * Reads the 268k EDSAN hospital documents (`docs_merged_00_25`).
   * Strips out all prescriptions (`ORDON*`) and transport vouchers (`BT*`).
   * Discards noisy legacy section columns that leaked doctor signatures and hospital headers.
   * Extracts a **10,000-document multi-dimensional stratified sample** into `data/raw/corpus_sample.jsonl`:
     * **All 26 individual years** (2000 through 2025).
     * **381 hospital wards / clinics** (`SEJUF`, 95.7% of all hospital units).
     * **75 medical services** (`SEJUM`, 100% coverage).
     * **94 distinct clinical document types** (`RECTYPE`).

2. **Weak Supervision Teacher (`src/redsan_doc_trimmer/annotate.py`)**:
   * Uses `gpt-5.6-luna` with structured outputs (`BoilerplateSpan`) to detect administrative header, footer, and signature spans.
   * Cost: Only ~$0.30 per 1,000 documents (returns line spans rather than verbose JSON per line).

3. **DrBERT Student Model (`src/redsan_doc_trimmer/train.py`)**:
   * Uses `Dr-BERT/DrBERT-7GB` pre-trained on French biomedical text.
   * Windowed context encoder: evaluates each target line with 2 preceding and 2 following context lines to handle broken OCR/ETL text.
   * Asymmetric class-weighted loss (`clinical_weight = 10.0`): strictly prioritizes clinical recall (100% preservation of medical facts).
   * Accelerated on your **NVIDIA GeForce RTX 3080 Laptop GPU** (30–35 seconds per training pass).

4. **Active Learning & Hard-Negative Mining (`src/redsan_doc_trimmer/mining.py`)**:
   * Student v1 scans thousands of unlabeled candidate documents on the GPU (with a live animated progress bar).
   * Automatically isolates the hardest, most ambiguous edge cases (lines where probability is borderline: $p \in [0.35, 0.65]$).
   * GPT-5.6 Luna adjudicates only those mined hard cases.
   * Student v2 is fine-tuned on the augmented dataset with a refined learning rate (`5e-6`).

5. **Standalone ONNX Export (`src/redsan_doc_trimmer/export_onnx.py`)**:
   * Exports the final model to `artifacts/active_learning/onnx_export/model.onnx` along with tokenizer assets.
   * Upstream `redsan` in R executes this model in-process with zero Python dependencies!

---

## 2. How to Run the Pipeline

### Option A: Fast Dry-Run Test (Zero Cost, No API Key Required)
You can test the entire pipeline locally in ~35 seconds using the built-in mock heuristic teacher:

```powershell
# Activate the virtual environment
.venv\Scripts\activate

# Run the active learning loop on a tiny subset (10 seed docs + scans 20 pool docs)
python scripts/run_active_learning.py --mock --seed_size 10 --pool_size 20 --mine_top_k 5
```

### Option B: Production Run with GPT-5.6 Luna & RTX 3080
Because your `OPENAI_API_KEY` is already present in your environment variables, Python automatically detects it without needing to type or paste it!

```powershell
# Activate virtual environment
.venv\Scripts\activate

# Run the production active learning cycle:
# 1,000 seed documents + 500 mined hard cases (~$0.45 total API cost)
python scripts/run_active_learning.py --seed_size 1000 --mine_top_k 500
```

### Running Automated Verification Tests Anytime
```powershell
.venv\Scripts\pytest -v
```

---

## 3. Output Deliverables

After the pipeline finishes, the production deliverables for `redsan` are located at:
`artifacts/active_learning/onnx_export/`
* `model.onnx` — The compiled neural network ready for in-process inference.
* `tokenizer.json` — The standalone tokenizer configuration and vocabulary.
