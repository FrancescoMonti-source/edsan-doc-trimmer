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
Hospital documents span decades (2000–2025) across hundreds of clinical care units and medical services within the hospital. Line wrapping is inconsistent, OCR errors are common, and physicians often embed critical clinical notes right next to administrative stamps. Rigid regexes either miss subtle boilerplate or accidentally cut into medical text.

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
                  │ (26 years, 381 care units, 75 departments)   │
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
                     • 100% Local, Fast & On-Premises
```

### 1. Multi-Dimensional Stratified Sampling
All documents originate from the same hospital's clinical data warehouse (EDSAN). To capture the full diversity of medical writing and formatting across the establishment, we extracted a 10,000-document sample stratified across:
* **All 26 individual years** (2000 to 2025).
* **381 clinical care units and wards** (`SEJUF`, representing 95.7% of all functional units across the hospital).
* **75 medical services / departments** (`SEJUM`, 100% departmental coverage).
* **94 clinical document types** (`RECTYPE`).

### 2. Weak Supervision via LLM Teacher
Labeling 100,000+ lines manually is prohibitively time-consuming. We used an LLM teacher with structured outputs to identify boilerplate spans on an initial seed batch of 1,000 documents. The teacher returns start/end line intervals rather than line-by-line classification, keeping annotations fast and compact.

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

### 2. Fully On-Premises & Hospital Privacy
In production, the ONNX model runs completely on-premises inside the hospital network via CPU or local GPU. No patient health information (PHI) ever leaves the secure hospital infrastructure.

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
python -m pytest -v
```

---

## Setup in Hospital / Air-Gapped HDW Environments

### Why Doesn't `git clone` Include the Model?
The production ONNX model (`model.onnx`), tokenizer assets, and training checkpoints are **gitignored** (`artifacts/` in `.gitignore`).
Committing multi-hundred-megabyte binaries directly to Git would bloat repository history permanently, drastically slow down cloning and branching across hospital networks, and exceed enterprise GitLab push size limits.

Runtime archives are named `edsan-doc-trimmer-vX.Y.Z.zip` and are built by `scripts/package_artifact.py` (see **Releasing a new archive** below). The worker uses ONNX Runtime for every inference provider.

### Archive Contents
The provider-aware archive contains everything required for standalone, offline inference with zero internet access:
* **`model.onnx`** (442.7 MB): The single inference graph, executed through `onnxruntime` with CPU, CUDA, or OpenVINO providers.
* **`tokenizer.json`**, **`tokenizer_config.json`**, **`special_tokens_map.json`**: Fast Rust / Hugging Face tokenizer assets.
* **`config.json`**: Sequence classification architecture metadata.
* **`artifact.json`**: Manifest declaring the artifact version, the worker contract (`model-only-v1`) and the export date.
* **`trim_batch_service.py`**: Batch worker using one ONNX Runtime inference path and reporting the selected execution provider. PyTorch checkpoint weights are not included in the runtime archive.

### Device selection
The worker defaults to `EDSAN_TRIMMER_DEVICE=auto`. Auto selects CUDA for a detected NVIDIA GPU when `CUDAExecutionProvider` is available, OpenVINO for a detected Intel GPU when `OpenVINOExecutionProvider` is available, and otherwise CPU. Set the variable to `cpu`, `cuda`, `openvino`, `dml`, or `migraphx` to override the choice.

If an accelerator is detected but its ONNX Runtime provider is unavailable, the worker continues on CPU and emits a marked warning with an install suggestion, such as `onnxruntime-gpu` for CUDA or `onnxruntime-openvino` plus OpenVINO for Intel. R surfaces these warnings. Table results include `TRIM_EXECUTION_PROVIDER`; character-vector results carry the provider as an attribute.

### Releasing a new archive (maintainers)
The training export folder (`artifacts/active_learning/onnx_export`) is never modified by packaging. A release is built in a temporary staging folder from an allowlist of export files (`model.onnx`, the tokenizer files, `config.json`), the worker script and a generated `artifact.json`:

```bash
# 1. Train and export (writes the export folder)
# 2. Package; --version is required and must be plain X.Y.Z (numbers only,
#    since redsan installs each release into a folder named after it)
python scripts/package_artifact.py --version X.Y.Z
# -> artifacts/edsan-doc-trimmer-vX.Y.Z.zip
# 3. Tag the commit that contains the packaged worker
git tag vX.Y.Z
# 4. Upload the zip to GitLab (see docs/GITLAB_RELEASE_GUIDE.md)
# 5. Install into the versioned cache through redsan:
#    redsan::edsan_install_trimmer("artifacts/edsan-doc-trimmer-vX.Y.Z.zip")
```

An existing archive of the same name is refused unless `--force` is passed, and a warning is printed when tag `vX.Y.Z` already exists but does not point at `HEAD`. `artifact.json` takes `exported_at` from the modification date (UTC) of `model.onnx`. After zipping, the archive is re-opened and packaging fails unless it contains exactly the allowlisted files, which guards against shipping `model.safetensors` (~820 MB archives). The script prints the `redsan::edsan_install_trimmer("<zip>")` command when it finishes; it does not install anything itself.

### 1. Where to Get `edsan-doc-trimmer-vX.Y.Z.zip`
* **Hospital Internal GitLab**: Navigate to **Deploy > Releases** (tag `vX.Y.Z`) and download the attached asset `edsan-doc-trimmer-vX.Y.Z.zip`.
* **Shared HDW Server Drive**: Pre-extracted or stored under `/data/shared/models/edsan-doc-trimmer/`.
* *Maintainers*: See **[docs/GITLAB_RELEASE_GUIDE.md](docs/GITLAB_RELEASE_GUIDE.md)** for step-by-step instructions on creating the release and uploading the binary asset via the GitLab Generic Package Registry.

### 2. Step-by-Step for R Users (`redsan`)
In R, install the model directly using `redsan`:

```r
library(redsan)

# Option A: One-time install from the zip (the supported install path)
edsan_install_trimmer("C:/path/to/edsan-doc-trimmer-vX.Y.Z.zip")
# On Linux HDW: edsan_install_trimmer("/path/to/edsan-doc-trimmer-vX.Y.Z.zip")

# Option B: Point directly to a pre-extracted or shared HDW directory:
Sys.setenv(EDSAN_TRIMMER_PATH = "/data/shared/models/edsan-doc-trimmer/vX.Y.Z")

# Python Environment Configuration:
# redsan automatically detects virtual environments in edsan-doc-trimmer or system Python.
# If Python is in a custom location, specify REDSAN_PYTHON_PATH:
Sys.setenv(REDSAN_PYTHON_PATH = "C:/path/to/venv/Scripts/python.exe")
# On Linux HDW:
# Sys.setenv(REDSAN_PYTHON_PATH = "/path/to/venv/bin/python")
# Or permanently in ~/.Renviron:
# REDSAN_PYTHON_PATH=C:/path/to/venv/Scripts/python.exe (or /path/to/venv/bin/python)

# Verify it works immediately:
clean_text <- trim_doceds_onnx("Consultation du 12/03/2024. Patient vu pour fievre.")
cat(clean_text)

# Batch trim a clinical cohort bundle:
trimmed_bundle <- trim_doceds_onnx(bundle)
```

### 3. Step-by-Step for Python Users
Colleagues running standalone Python scripts (`scripts/trim_batch_service.py` or `scripts/trim_document.py`) have two options:

#### Option A: Install with redsan
Run `redsan::edsan_install_trimmer("edsan-doc-trimmer-vX.Y.Z.zip")` once (see above). The scripts find the installed version automatically.

Do not extract a release into `artifacts/active_learning/onnx_export`: that folder is training output, packaging reads it, and the scripts never search it.

**Automatic model discovery** (used when `--onnx_dir` is not given) follows redsan's versioned trimmer cache: `EDSAN_TRIMMER_PATH` (legacy `REDSAN_TRIMMER_PATH`), then the `EDSAN_TRIMMER_VERSION` pin, then the highest valid `<cache root>/<version>/` (the cache root is `tools::R_user_dir("edsan_doc_trimmer", "cache")`), then the legacy `<cache root>/v1` slot. A folder counts only if its name equals the `artifact_version` in its `artifact.json`. When `--onnx_dir` is not given, the scripts print the model they chose to stderr.

#### Option B: Extract to a folder and set `EDSAN_TRIMMER_PATH` (or pass `--onnx_dir`)
```bash
# Linux / macOS:
unzip edsan-doc-trimmer-vX.Y.Z.zip -d /data/shared/models/edsan-doc-trimmer/vX.Y.Z
export EDSAN_TRIMMER_PATH="/data/shared/models/edsan-doc-trimmer/vX.Y.Z"

# Windows PowerShell:
Expand-Archive -Path edsan-doc-trimmer-vX.Y.Z.zip -DestinationPath "C:\models\edsan-doc-trimmer\vX.Y.Z" -Force
$env:EDSAN_TRIMMER_PATH = "C:\models\edsan-doc-trimmer\vX.Y.Z"
```

Then run scripts without specifying paths:
```bash
# Test on a sample document
python scripts/trim_document.py

# Run batch worker
python scripts/trim_batch_service.py --input batch_docs.json --output trimmed_docs.json
```

### 4. Multi-User Shared Folder Setup on HDW Servers
To avoid duplicating large models for every data scientist on a shared server, administrators can configure a single shared model instance:

1. **Extract once to a shared location**:
   ```bash
   mkdir -p /data/shared/models/edsan-doc-trimmer/vX.Y.Z
   unzip edsan-doc-trimmer-vX.Y.Z.zip -d /data/shared/models/edsan-doc-trimmer/vX.Y.Z/
   chmod -R a+rX /data/shared/models/edsan-doc-trimmer/
   ```

2. **Configure environment variables for all users**:
   * In `/etc/environment` or `/etc/profile.d/edsan.sh`:
     ```bash
     export EDSAN_TRIMMER_PATH="/data/shared/models/edsan-doc-trimmer/vX.Y.Z"
     ```
   * Or in users' `~/.Renviron`:
     ```
     EDSAN_TRIMMER_PATH=/data/shared/models/edsan-doc-trimmer/vX.Y.Z
     ```

Both R (`redsan`) and Python (`trim_batch_service.py`, `trim_document.py`) will automatically discover and load the shared model!

---

For detailed pipeline reproduction and active learning commands, see **[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md)**.
For hospital GitLab release setup and binary hosting, see **[docs/GITLAB_RELEASE_GUIDE.md](docs/GITLAB_RELEASE_GUIDE.md)**.
