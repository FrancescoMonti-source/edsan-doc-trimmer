#!/usr/bin/env python3
"""Cross-platform verification script.

Verifies that:
1. Python in-process `trim_batch` (CPU ONNX)
2. Python standalone worker CLI (`trim_batch_service.py`)
3. Upstream R execution via `redsan::trim_doceds_onnx`
produce 100% identical trimmed text, character counts, and intervals.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# Add scripts and tests to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_transport_voucher import DISCHARGE_LETTER, VOUCHER
from trim_batch_service import trim_batch

DOCS = [
    {"id": "doc_discharge", "text": DISCHARGE_LETTER, "rectype": "CRH2AB"},
    {"id": "doc_voucher", "text": VOUCHER, "rectype": "ORDON7"},
]

def test_in_process_python():
    print("--- Testing Python in-process trim_batch ---")
    results = trim_batch(DOCS)
    print(f"Discharge letter: raw={results[0]['raw_chars']}, trimmed={results[0]['trimmed_chars']}, reduction={results[0]['reduction_pct']}%")
    print(f"Voucher: raw={results[1]['raw_chars']}, trimmed={results[1]['trimmed_chars']}, reduction={results[1]['reduction_pct']}%")
    return results

def test_cli_worker(in_process_results):
    print("\n--- Testing Python CLI trim_batch_service.py ---")
    input_file = REPO_ROOT / "artifacts" / "test_input.json"
    output_file = REPO_ROOT / "artifacts" / "test_output.json"
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(DOCS, f)

    cmd = [
        str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"),
        str(REPO_ROOT / "scripts" / "trim_batch_service.py"),
        "--input", str(input_file),
        "--output", str(output_file),
    ]
    subprocess.run(cmd, check=True)

    with open(output_file, "r", encoding="utf-8") as f:
        cli_results = json.load(f)

    for r_in, r_cli in zip(in_process_results, cli_results):
        assert r_in["trimmed_text"] == r_cli["trimmed_text"], f"Mismatch in trimmed text for {r_in['id']}"
        assert r_in["raw_chars"] == r_cli["raw_chars"]
        assert r_in["trimmed_chars"] == r_cli["trimmed_chars"]
        assert r_in["reduction_pct"] == r_cli["reduction_pct"]
        assert r_in["preserved_intervals"] == r_cli["preserved_intervals"]
    print("Python CLI results are 100% IDENTICAL to in-process trim_batch!")
    return cli_results

def test_r_execution(in_process_results):
    print("\n--- Testing R redsan::trim_doceds_onnx ---")
    rscript = r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe"
    r_test_file = REPO_ROOT / "artifacts" / "test_redsan.R"
    r_output_json = REPO_ROOT / "artifacts" / "r_output.json"
    input_file = REPO_ROOT / "artifacts" / "test_input.json"

    # Write R test script
    r_code = f"""
devtools::load_all("C:/Users/franc/Documents/Git/redsan", quiet = TRUE)
library(jsonlite)

Sys.setenv(REDSAN_PYTHON_PATH = "{str(REPO_ROOT / '.venv' / 'Scripts' / 'python.exe').replace(chr(92), '/')}")

docs_json <- read_json("{str(input_file).replace(chr(92), '/')}", simplifyVector = TRUE)
docs_df <- data.frame(
  id = docs_json$id,
  RECTXT = docs_json$text,
  RECTYPE = docs_json$rectype,
  stringsAsFactors = FALSE
)

trimmed_df <- trim_doceds_onnx(docs_df, text_col = "RECTXT")

out_list <- list(
  list(
    id = trimmed_df$id[1],
    trimmed_text = trimmed_df$RECTXT_TRIMMED[1],
    trimmed_chars = nchar(trimmed_df$RECTXT_TRIMMED[1]),
    reduction_pct = trimmed_df$TRIM_REDUCTION_PCT[1],
    preserved_intervals = fromJSON(trimmed_df$TRIM_PRESERVED_INTERVALS[1], simplifyVector = FALSE)
  ),
  list(
    id = trimmed_df$id[2],
    trimmed_text = trimmed_df$RECTXT_TRIMMED[2],
    trimmed_chars = nchar(trimmed_df$RECTXT_TRIMMED[2]),
    reduction_pct = trimmed_df$TRIM_REDUCTION_PCT[2],
    preserved_intervals = fromJSON(trimmed_df$TRIM_PRESERVED_INTERVALS[2], simplifyVector = FALSE)
  )
)

writeLines(toJSON(out_list, auto_unbox = TRUE, pretty = TRUE), "{str(r_output_json).replace(chr(92), '/')}")
"""
    with open(r_test_file, "w", encoding="utf-8") as f:
        f.write(r_code)

    res = subprocess.run([rscript, str(r_test_file)], capture_output=True, text=True)
    if res.returncode != 0:
        print("R script failed with stderr:")
        print(res.stderr)
        print("stdout:")
        print(res.stdout)
        sys.exit(1)

    with open(r_output_json, "r", encoding="utf-8") as f:
        r_results = json.load(f)

    for py_res, r_res in zip(in_process_results, r_results):
        print(f"Comparing {py_res['id']}...")
        assert py_res["trimmed_text"] == r_res["trimmed_text"], f"Mismatch between Python and R for {py_res['id']}:\nPY:\n{py_res['trimmed_text']}\n\nR:\n{r_res['trimmed_text']}"
        assert py_res["trimmed_chars"] == r_res["trimmed_chars"]
        assert abs(py_res["reduction_pct"] - r_res["reduction_pct"]) < 0.01
        assert len(py_res["preserved_intervals"]) == len(r_res["preserved_intervals"])
        for py_iv, r_iv in zip(py_res["preserved_intervals"], r_res["preserved_intervals"]):
            assert py_iv["start"] == r_iv["start"]
            assert py_iv["end"] == r_iv["end"]
            assert py_iv["text"] == r_iv["text"]

    print("R redsan::trim_doceds_onnx results are 100% IDENTICAL to Python execution!")

if __name__ == "__main__":
    py_res = test_in_process_python()
    test_cli_worker(py_res)
    test_r_execution(py_res)
    print("\nALL CONSISTENCY CHECKS PASSED PERFECTLY!")
