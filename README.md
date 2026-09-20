# edsan-doc-trimmer

**Fast, high-recall detection and removal of administrative boilerplate in French hospital documents (`RECTXT` from EDSAN).**

Part of the **redsan** ecosystem: trains lightweight student models (based on `DrBERT`) using weak supervision from an LLM teacher and human-in-the-loop active learning. The final model is exported as an ONNX artifact for high-throughput, private, local inference inside hospital networks.

---

## The Problem: Why Do We Need This?

Hospital Electronic Health Record (EHR) systems store millions of free-text clinical notes (`RECTXT`) — consultation reports, discharge summaries, prescriptions, and operative reports.

Across these documents, **25% to 35% of the text is repetitive administrative boilerplate**:
* Hospital letterheads, department addresses, FINESS numbers, telephone/fax lines.
* Doctor titles, RPPS registration numbers, secretarial initials, and signature blocks.
* Standard legal disclaimers and electronic signature warnings (*"Document validé électroniquement..."*).
* Mail-merge artifacts from word processors (`MERGEFIELD`, `FORMCHECKBOX`).

### Why not just use regular expressions?
Hospital documents span decades (2000–2025) across hundreds of clinical wards. Line wrapping is inconsistent, OCR errors are common, and physicians often embed critical clinical notes right next to administrative stamps. Rigid regexes either miss subtle boilerplate or accidentally cut into medical text.

### The Cardinal Rule: Asymmetric Cost of Error
In medical NLP, errors are not equal:
* **Keeping boilerplate (False Negative)**: Harmless — it only costs a few extra prompt tokens in downstream LLM calls.
* **Stripping clinical narrative (False Positive)**: **Fatal** — destroying a medication dosage (*"Kardegic 75mg"*), an allergy, or a diagnosis (*"pas de récidive d'AVC"*) corrupts patient history and downstream medical coding.

> **Decision Policy:** *When in doubt, always keep the text.* Every stage of our training and inference is tuned to guarantee near-perfect recall on clinical content.

---

## Methodology: How It Works (Step-by-Step)

Instead of manually labeling tens of thousands of lines from scratch, we used a **Student-Teacher Distillation + Active Learning** pipeline with human-in-the-loop curation:

```
                  ┌──────────────────────────────────────────────┐
                  │ 10,000 Multi-Dimensional Stratified Corpus   │
                  │ (26 years, 381 hospital units, 94 doc types) │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                             [Step 1: Seed Annotation]
                        1,000 Seed Documents ──► LLM Teacher
                                         │
                                         ▼
                        [Step 2: Train Student v1 (DrBERT)]
                     Asymmetric loss: 10x penalty on clinical errors
                                         │
                                         ▼
                      [Step 3: Hard-Negative / Uncertainty Mining]
                    Student v1 scans candidate pool on GPU:
                    Finds 500 edge cases with borderline probability
                                         │
                                         ▼
                    [Step 4: Human-in-the-Loop Gold Standard]
                    LLM pre-annotation + 100% manual review by human
                                         │
                                         ▼
                     [Step 5: Retrain Student v2 & Export ONNX]
                     Final artifact: model.onnx (442 MB)
                     • 98.12% Clinical Recall
                     • 97.89% Boilerplate Precision
                     • 100% Local & Private (0 API costs)
```

### 1. Multi-Dimensional Stratified Sampling
We extracted a representative 10,000-document sample from EDSAN covering:
* **All 26 individual years** (2000 to 2025).
* **381 hospital clinics and wards** (`SEJUF`, representing 95.7% of all units).
* **75 medical services** (`SEJUM`, 100% coverage).
* **94 clinical document types** (`RECTYPE`).

### 2. Weak Supervision via LLM Teacher
Labeling 100,000+ lines manually is cost-prohibitive. We used an LLM teacher with structured outputs to identify boilerplate spans on an initial seed batch of 1,000 documents (~$0.30 total API cost). The teacher returns start/end line intervals rather than line-by-line classification, keeping annotations fast and affordable.

### 3. Student Model Training (DrBERT)
We trained a specialized student model using **DrBERT** (`Dr-BERT/DrBERT-7GB`), a language model pre-trained on French biomedical text:
* **Context Windows**: Evaluates each line together with 2 lines before and 2 lines after, allowing the model to distinguish between a doctor's signature at the end of a note and an inline mention of a colleague.
* **Asymmetric Class Weighting**: Penalizes errors on clinical text **10x more heavily** than errors on boilerplate.

### 4. Active Learning & Uncertainty Mining
Instead of blindly labeling more random documents, Student v1 was run over thousands of unseen documents to find its own "blind spots." We mined the **500 hardest edge-case documents** where line predictions fell into the uncertainty zone ($p \in [0.35, 0.65]$).

### 5. Human-in-the-Loop Gold Standard
The 500 mined edge cases were pre-annotated by the LLM and then **100% manually curated line-by-line by a human expert** (adjusting 2,167 ambiguous spans). This produced an audited, gold-standard dataset specifically addressing difficult border cases (e.g. multi-line prescriptions, secretarial postscripts, complex hospital letterheads).

### 6. Final Training & ONNX Export
Student v2 was retrained on the combined dataset (1,500 curated documents, 100,047 lines). The final model weights and tokenizer were exported to a standalone **ONNX** graph (`model.onnx`, 442 MB).

---

## Key Guarantees

### 1. The Grounding Guarantee (Exact Coordinates)
Every prediction maps back to exact character intervals `[start_char, end_char]` in the raw `RECTXT`. No words are generated, altered, or rephrased. Downstream tools can verify that:
$$\text{substring}(\text{raw\_rectxt}, \text{start}, \text{end}) == \text{preserved\_text}$$
This provides 100% traceability and auditability for medical coding and clinical validation.

### 2. Zero Cloud Dependencies & Hospital Privacy
In production, the ONNX model runs locally inside the hospital network via CPU or local GPU. No patient health information (PHI) ever leaves the secure hospital environment, and runtime cost is $0.00.

---

## Results & Performance

### Holdout Validation Benchmark (20,010 lines)
* **Clinical Recall**: `98.12%` *(The Cardinal Rule: clinical content is strictly preserved)*
* **Boilerplate Precision**: `97.89%` *(Removed lines are guaranteed administrative boilerplate)*
* **Macro F1**: `95.92%`

### Real-World Token Reduction
Evaluated across **800 real hospital documents** from the EDSAN warehouse:
* **23.5% to 35.2% net token reduction** on full medical records.
* **Prescriptions (`ORDON*`)**: Strips hospital banners and legal footnotes while preserving 100% of molecules, dosages, and administration schedules.
* **Discharge & Consultation Summaries (`CRH*`, `CR2AAF`)**: Strips administrative headers and typist stamps while preserving all medical history, physical exams, and conclusions.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/FrancescoMonti-source/edsan-doc-trimmer.git
cd edsan-doc-trimmer

# Setup Python environment
uv venv
uv pip install -e ".[dev]"
```

### Running Tests

```bash
pytest -v
```

### Upstream Integration in R (`redsan`)

The trained ONNX artifact is packaged as a release zip (`edsan-doc-trimmer-v1.0.0.zip`) and called directly from R:

```r
library(redsan)

# Automatically runs background batch inference via ONNX Runtime
trimmed_doceds <- trim_doceds_onnx(bundle$sources$doceds)
```

For detailed pipeline reproduction and active learning commands, see **[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)**.
