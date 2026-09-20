# Working in redsan-doc-trimmer

## Scope

This repository trains and exports machine learning models (specifically DrBERT-based sequence/span taggers) to distinguish clinical narrative from non-clinical administrative boilerplate in French hospital documents (`RECTXT` from EDSAN).

- **Who calls this**: Used to train models and export ONNX artifacts for upstream consumption by `redsan`.
- **Output artifact**: The primary deliverable of this repository is an exported ONNX model (`model.onnx`) and tokenizer metadata packaged into `edsan-doc-trimmer-v1.0.0.zip`, executed by `redsan` via a background Python batch worker (`trim_batch_service.py`).
- **Language & Stack**: Python 3.11+, PyTorch, Hugging Face `transformers`, `onnx`, `onnxruntime`.

## The Cardinal Rule: Asymmetric Cost of Error

- **False Negative (keeping boilerplate)**: Harmless — costs a few extra prompt tokens in downstream LLM calls.
- **False Positive (stripping clinical content)**: Fatal — destroys medical facts, diagnoses, or measurements needed for downstream coding and validation.
- **Decision policy**: In case of ambiguity, always keep the text. Thresholds must heavily favor high recall on clinical content.

## Coordinates & Grounding Guarantee

Every trimming prediction must map back to exact character intervals `[start_char, end_char]` in the raw `RECTXT` document, so downstream citation and quote verification in `redsan` and `redsan-coding` remains fully traceable.

## Agent skills

### Issue tracker

GitHub Issues (`gh` CLI) with PR triage disabled. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context (`CONTEXT.md` at root, ADRs in `docs/adr/`). See `docs/agents/domain.md`.
