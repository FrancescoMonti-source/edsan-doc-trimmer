# Pipeline Walkthrough & User Guide

This document describes how the `redsan-doc-trimmer` pipeline works and how to run each step.

---

## What We Have Built

1. **Corpus Extraction (`scripts/extract_sample.R`)**:
   * Reads the 268k EDSAN hospital documents (`docs_merged_00_25`).
   * Automatically strips out prescriptions (`ORDON*`) and transport vouchers (`BT*`).
   * Extracts a balanced 2,000-document sample into `data/raw/corpus_sample.jsonl`.

2. **Weak Supervision Teacher (`src/redsan_doc_trimmer/annotate.py`)**:
   * Calls `gpt-5.6-luna` with structured outputs to tag administrative header/footer line spans.
   * Protects clinical content: if a line appears in a confirmed clinical section (`CONCLUSIONDIAGNOSTIC`, `EXAMENCLINIQUE`), it is automatically rescued.
   * Cost: Only ~$0.30 per 1,000 documents.

3. **DrBERT Student Model (`src/redsan_doc_trimmer/train.py`)**:
   * Context-aware classifier (uses the target line + 2 preceding and 2 following lines).
   * Asymmetric loss (`clinical_weight = 10.0`): heavily penalizes false positives to guarantee high clinical recall.

4. **Active Learning & Hard-Negative Mining (`src/redsan_doc_trimmer/mining.py`)**:
   * Student v1 scans unlabeled documents to find its own hardest edge cases (uncertain lines and anchor conflicts).
   * Teacher adjudicates only those edge cases.
   * Student v2 is fine-tuned on the combined dataset.

5. **ONNX Export (`src/redsan_doc_trimmer/export_onnx.py`)**:
   * Compiles the trained model and tokenizer into `model.onnx` so the `redsan` R package can run it in-process without Python.

---

## How to Run the Pipeline (Step-by-Step)

### Option A: Fast Dry-Run Test (Zero Cost, No API Key Required)
You can test the entire pipeline locally in under 30 seconds with a mock teacher:

```powershell
# Activate the virtual environment
.venv\Scripts\activate

# Run the active learning loop on a tiny subset (10 seed docs + scan 20 pool docs)
python scripts/run_active_learning.py --mock --seed_size 10 --pool_size 20 --mine_top_k 5
```

### Option B: Real Run with GPT-5.6 Luna
Because `OPENAI_API_KEY` is already present in your environment variables, the script **automatically detects it** without having to enter or set it!

```powershell
# Run the full production active learning cycle (400 seed docs + 200 mined hard cases, ~$0.18 total)
python scripts/run_active_learning.py --seed_size 400 --mine_top_k 200
```


### Running Tests Anytime
```powershell
.venv\Scripts\pytest -v
```

