# redsan-doc-trimmer

Training and ONNX export pipeline for detecting and trimming non-clinical administrative boilerplate in French hospital documents (`RECTXT` from EDSAN).

Part of the **redsan** ecosystem: trains student models (using `DrBERT/DrBERT-7GB`) with weak supervision from LLM teachers, exporting lightweight ONNX artifacts for high-throughput, in-process inference in `redsan` without Python dependencies in production.

## Pipeline Architecture

1. **Weak Supervision (LLM Teacher)**: Annotates a sample of raw `RECTXT` documents to detect administrative frames (headers, footers, phone numbers, legal disclaimers, signature blocks) while preserving character offset coordinates.
2. **Context-Aware Sequence Tagging (DrBERT Student)**: Handles arbitrarily broken post-ETL lines by utilizing sliding windows of surrounding context.
3. **Asymmetric Risk Optimization**: Penalizes false positives on clinical content to prioritize high clinical recall over aggressive boilerplate removal.
4. **ONNX Export**: Exports the trained weights to an ONNX graph consumed upstream by `redsan::trim_doceds_text()`.

## Getting Started

```bash
# Install with uv or pip
uv venv
uv pip install -e ".[dev]"
```
