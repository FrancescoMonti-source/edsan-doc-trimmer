#!/usr/bin/env python3
"""Generate an interactive HTML visualizer for 50 diverse documents from D0840 and denut.rds."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from redsan_doc_trimmer.dataset import build_windowed_samples, extract_lines_with_offsets


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>redsan-doc-trimmer: Step 2 Visual Inspector (50 Diverse Documents)</title>
<style>
  :root {
    --bg-main: #0f172a;
    --bg-card: #1e293b;
    --bg-code: #090d16;
    --text-main: #e2e8f0;
    --text-muted: #94a3b8;
    --border: #334155;
    --accent-blue: #38bdf8;
    --accent-green: #10b981;
    --accent-purple: #a855f7;
    --accent-red: #ef4444;
    --accent-amber: #f59e0b;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 0;
    background: var(--bg-main);
    color: var(--text-main);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  header {
    background: #020617;
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .badge-model {
    background: #064e3b;
    color: #6ee7b7;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: bold;
    text-transform: uppercase;
  }
  .top-stats { display: flex; gap: 16px; align-items: center; font-size: 13px; color: var(--text-muted); }
  .stat-pill {
    background: var(--bg-card);
    border: 1px solid var(--border);
    padding: 4px 10px;
    border-radius: 6px;
    display: flex;
    gap: 6px;
  }
  .stat-pill strong { color: var(--accent-blue); }

  /* Navigation Bar */
  .nav-bar {
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    padding: 8px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    flex-shrink: 0;
  }
  .nav-group { display: flex; align-items: center; gap: 8px; }
  select, button {
    background: #090d16;
    color: var(--text-main);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 13px;
    cursor: pointer;
    outline: none;
  }
  select:focus, button:focus { border-color: var(--accent-blue); }
  button:hover { background: #1e293b; border-color: #475569; }
  .btn-primary { background: #0284c7; color: #fff; border-color: #0284c7; font-weight: 600; }
  .btn-primary:hover { background: #0369a1; }
  
  /* Filter pills */
  .filter-btn { padding: 4px 8px; font-size: 12px; border-radius: 4px; }
  .filter-btn.active { background: #38bdf8; color: #020617; font-weight: bold; }

  /* Main Workspace */
  .workspace {
    flex: 1;
    display: flex;
    overflow: hidden;
  }
  .panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border-right: 1px solid var(--border);
  }
  .panel:last-child { border-right: none; }
  .panel-header {
    background: #1e293b;
    border-bottom: 1px solid var(--border);
    padding: 10px 16px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 13px;
    font-weight: 600;
  }
  .panel-body {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 12.5px;
    line-height: 1.5;
    background: var(--bg-code);
  }

  /* Line Row */
  .line-row {
    display: flex;
    align-items: flex-start;
    padding: 2px 4px;
    border-radius: 4px;
    margin-bottom: 2px;
    transition: background 0.1s;
  }
  .line-row:hover { background: #1e293b; }
  .line-num {
    width: 38px;
    color: #475569;
    font-size: 11px;
    user-select: none;
    text-align: right;
    padding-right: 8px;
    flex-shrink: 0;
  }
  .line-badge {
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: bold;
    text-transform: uppercase;
    width: 90px;
    text-align: center;
    margin-right: 10px;
    flex-shrink: 0;
    user-select: none;
  }
  .badge-clin { background: #064e3b; color: #6ee7b7; }
  .badge-bp { background: #4c1d95; color: #d8b4fe; }
  .prob-score {
    width: 48px;
    font-size: 10.5px;
    color: #94a3b8;
    text-align: right;
    margin-right: 12px;
    flex-shrink: 0;
    user-select: none;
  }
  .line-text {
    flex: 1;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .line-text.clin { color: #f1f5f9; }
  .line-text.bp { color: #64748b; text-decoration: line-through; opacity: 0.65; }

  /* Summary Card */
  .doc-summary {
    background: #0f172a;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 16px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  .doc-summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 12px;
    margin-top: 10px;
  }
  .summary-item {
    background: #1e293b;
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid #334155;
  }
  .summary-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; }
  .summary-val { font-size: 15px; font-weight: bold; margin-top: 2px; }

  /* Output Preview panel */
  .clean-output {
    white-space: pre-wrap;
    word-break: break-word;
    color: #e2e8f0;
  }

  .view-toggle { display: flex; gap: 4px; }
</style>
</head>
<body>

<header>
  <div class="brand">
    <h2 style="margin: 0; font-size: 17px; letter-spacing: -0.02em;">redsan-doc-trimmer</h2>
    <span class="badge-model">Student v2 ONNX</span>
    <span style="font-size: 13px; color: var(--text-muted);">Step 2: Interactive Verification (50 Real Documents)</span>
  </div>
  <div class="top-stats">
    <div class="stat-pill">Dataset: <strong>D0840 & denut.rds</strong></div>
    <div class="stat-pill">Target Recall: <strong>&ge; 98%</strong></div>
    <div class="stat-pill">Prescriptions: <strong>Preserved</strong></div>
  </div>
</header>

<div class="nav-bar">
  <div class="nav-group">
    <button onclick="prevDoc()" title="Previous document (Arrow Left / A)">&larr; Prev</button>
    <select id="doc-select" onchange="renderDoc(this.value)" style="min-width: 380px;"></select>
    <button onclick="nextDoc()" title="Next document (Arrow Right / D)">Next &rarr;</button>
    <span id="doc-counter" style="font-size: 12px; color: var(--text-muted); margin-left: 8px;"></span>
  </div>

  <div class="nav-group">
    <span style="font-size: 12px; color: var(--text-muted);">Filter:</span>
    <button class="filter-btn active" onclick="filterType('ALL', this)">All (50)</button>
    <button class="filter-btn" onclick="filterType('ORDON', this)">Prescriptions (ORDON*)</button>
    <button class="filter-btn" onclick="filterType('CONSULT', this)">Consult / Letters</button>
    <button class="filter-btn" onclick="filterType('DENUT', this)">DENUT Cohort</button>
    <button class="filter-btn" onclick="filterType('EXTREME', this)">High Trim (&gt;50%)</button>
  </div>
</div>

<div class="workspace">
  <!-- Left Panel: Detailed Line Classification -->
  <div class="panel" style="flex: 1.15;">
    <div class="panel-header">
      <span>1. Line-by-Line Classification (Green = Preserved Clinical, Purple = Trimmed Boilerplate)</span>
      <span style="font-size: 11px; color: var(--text-muted);">Click line to inspect offsets</span>
    </div>
    <div class="panel-body" id="left-panel"></div>
  </div>

  <!-- Right Panel: Final Clean Clinical Narrative (Downstream LLM View) -->
  <div class="panel" style="flex: 0.85;">
    <div class="panel-header">
      <span>2. Cleaned Document Deliverable (What downstream LLM & coding receive)</span>
      <span id="reduction-badge" class="badge-model" style="background: #0284c7; color: #e0f2fe;"></span>
    </div>
    <div class="panel-body" id="right-panel"></div>
  </div>
</div>

<script>
const docsData = DOCS_DATA_PLACEHOLDER;
let filteredIndices = docsData.map((_, i) => i);
let currentFilteredIdx = 0;

function populateSelect() {
  const sel = document.getElementById('doc-select');
  sel.innerHTML = '';
  filteredIndices.forEach((docIdx, fIdx) => {
    const d = docsData[docIdx];
    const opt = document.createElement('option');
    opt.value = fIdx;
    opt.textContent = `[${d.dataset}] ${d.rectype.padEnd(10)} | Reduc: ${d.reduction_pct.toFixed(1)}% | ${d.doc_id}`;
    sel.appendChild(opt);
  });
}

function filterType(category, btn) {
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');

  if (category === 'ALL') {
    filteredIndices = docsData.map((_, i) => i);
  } else if (category === 'ORDON') {
    filteredIndices = docsData.map((d, i) => d.rectype.toUpperCase().includes('ORDON') ? i : -1).filter(i => i !== -1);
  } else if (category === 'CONSULT') {
    filteredIndices = docsData.map((d, i) => (d.rectype.includes('CR2') || d.rectype.includes('CRH') || d.rectype.includes('DICT')) ? i : -1).filter(i => i !== -1);
  } else if (category === 'DENUT') {
    filteredIndices = docsData.map((d, i) => d.dataset === 'DENUT' ? i : -1).filter(i => i !== -1);
  } else if (category === 'EXTREME') {
    filteredIndices = docsData.map((d, i) => d.reduction_pct >= 50.0 ? i : -1).filter(i => i !== -1);
  }

  populateSelect();
  currentFilteredIdx = 0;
  renderDoc(0);
}

function prevDoc() {
  if (currentFilteredIdx > 0) {
    currentFilteredIdx--;
    renderDoc(currentFilteredIdx);
  }
}

function nextDoc() {
  if (currentFilteredIdx < filteredIndices.length - 1) {
    currentFilteredIdx++;
    renderDoc(currentFilteredIdx);
  }
}

function renderDoc(fIdx) {
  currentFilteredIdx = parseInt(fIdx, 10);
  const realIdx = filteredIndices[currentFilteredIdx];
  const doc = docsData[realIdx];
  if (!doc) return;

  document.getElementById('doc-select').value = currentFilteredIdx;
  document.getElementById('doc-counter').textContent = `Doc ${currentFilteredIdx + 1} of ${filteredIndices.length}`;

  const reducBadge = document.getElementById('reduction-badge');
  reducBadge.textContent = `-${doc.reduction_pct.toFixed(1)}% Tokens (${doc.trimmed_chars} / ${doc.raw_chars} chars)`;

  // Render Left Panel (Lines)
  let linesHtml = `
    <div class="doc-summary">
      <div style="display: flex; justify-content: space-between; align-items: baseline;">
        <span style="font-weight: bold; font-size: 15px; color: #f8fafc;">${doc.doc_id}</span>
        <span style="font-size: 12px; color: var(--accent-blue);">${doc.dataset}</span>
      </div>
      <div class="doc-summary-grid">
        <div class="summary-item">
          <div class="summary-label">Document Type</div>
          <div class="summary-val" style="color: #38bdf8;">${doc.rectype}</div>
        </div>
        <div class="summary-item">
          <div class="summary-label">Token Savings</div>
          <div class="summary-val" style="color: #10b981;">-${doc.reduction_pct.toFixed(1)}%</div>
        </div>
        <div class="summary-item">
          <div class="summary-label">Preserved Lines</div>
          <div class="summary-val" style="color: #6ee7b7;">${doc.clinical_lines} / ${doc.total_lines}</div>
        </div>
        <div class="summary-item">
          <div class="summary-label">Stripped Lines</div>
          <div class="summary-val" style="color: #d8b4fe;">${doc.bp_lines} lines</div>
        </div>
      </div>
    </div>
  `;

  for (let i = 0; i < doc.lines.length; i++) {
    const l = doc.lines[i];
    const isBp = l.is_bp;
    const badgeCls = isBp ? 'badge-bp' : 'badge-clin';
    const badgeText = isBp ? 'BOILERPLATE' : 'CLINICAL';
    const textCls = isBp ? 'bp' : 'clin';
    const probText = `p=${l.prob_bp.toFixed(2)}`;
    const safeText = l.text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const coords = `[${l.start_char}..${l.end_char}]`;

    linesHtml += `
      <div class="line-row" title="Offsets in raw text: ${coords}">
        <div class="line-num">${i}</div>
        <div class="line-badge ${badgeCls}">${badgeText}</div>
        <div class="prob-score">${probText}</div>
        <div class="line-text ${textCls}">${safeText}</div>
      </div>
    `;
  }
  document.getElementById('left-panel').innerHTML = linesHtml;

  // Render Right Panel (Clean output)
  const cleanHtml = `
    <div style="margin-bottom: 12px; font-size: 11.5px; color: #94a3b8;">
      Exact character intervals preserved: <b>${doc.clinical_lines} intervals</b> grounded in raw <code>RECTXT</code>.
    </div>
    <div class="clean-output">${doc.trimmed_text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')}</div>
  `;
  document.getElementById('right-panel').innerHTML = cleanHtml;
}

// Keyboard shortcuts
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowLeft' || e.key.toLowerCase() === 'a') prevDoc();
  if (e.key === 'ArrowRight' || e.key.toLowerCase() === 'd') nextDoc();
});

// Init
populateSelect();
renderDoc(0);
</script>
</body>
</html>
"""


def generate_benchmark_viewer(
    sample_file: str = "data/processed/benchmark_sample.jsonl",
    model_dir: str = "artifacts/active_learning/student_v2/best_model",
    n_docs: int = 50,
    output_html: str = "artifacts/benchmark_viewer.html",
):
    print("=" * 80)
    print(f"GENERATING INTERACTIVE BENCHMARK VIEWER FOR {n_docs} DOCUMENTS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    # Load documents from sample file
    d0840_docs = []
    denut_docs = []
    with open(sample_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("dataset") == "DENUT":
                denut_docs.append(d)
            else:
                d0840_docs.append(d)

    print(f"Found in pool: {len(d0840_docs)} D0840 docs and {len(denut_docs)} DENUT docs.")

    # Stratified selection to get a diverse representative set of 50 docs:
    # We want:
    # - 15 Prescriptions (ORDON7, ORDON6, ORDONACTE2) from both datasets
    # - 15 Consultation / Hospitalization (CR2AAF, CRH2AB, CTDEFAUT, DICT)
    # - 5 Nursing handovers (XWAYTC, XWAY)
    # - 5 Imaging / Exams (IMAG, CRRAD)
    # - 10 Other types (TU, DAVINCI, DIANE, LDL2024, etc.)
    selected_docs = []

    def pick_docs(pool, condition, count):
        matched = [d for d in pool if condition(d)]
        return matched[:count]

    # Prescriptions
    selected_docs.extend(pick_docs(denut_docs, lambda d: "ORDON" in d.get("rectype", "").upper(), 8))
    selected_docs.extend(pick_docs(d0840_docs, lambda d: "ORDON" in d.get("rectype", "").upper(), 7))

    # Narrative Consultations & Hospitalizations
    selected_docs.extend(pick_docs(d0840_docs, lambda d: d.get("rectype") in ["CR2AAF", "CRH2AB", "CTDEFAUT"], 8))
    selected_docs.extend(pick_docs(denut_docs, lambda d: d.get("rectype") in ["DICT", "CRH2AB", "TU"], 7))

    # Handover & Nursing
    selected_docs.extend(pick_docs(denut_docs, lambda d: "XWAY" in d.get("rectype", "").upper(), 5))

    # Imaging
    selected_docs.extend(pick_docs(d0840_docs, lambda d: d.get("rectype") in ["IMAG", "CRRAD"], 3))
    selected_docs.extend(pick_docs(denut_docs, lambda d: d.get("rectype") in ["IMAG"], 2))

    # Specialty & Edge cases
    selected_docs.extend(pick_docs(d0840_docs, lambda d: d.get("rectype") in ["DAVINCI", "DIANE", "LDL2024"], 5))
    selected_docs.extend(pick_docs(denut_docs, lambda d: d.get("rectype") in ["DAVINCI", "TAS2AA"], 5))

    # Fill up to 50 if needed
    seen_ids = {d["doc_id"] for d in selected_docs}
    for d in d0840_docs + denut_docs:
        if len(selected_docs) >= n_docs:
            break
        if d["doc_id"] not in seen_ids:
            selected_docs.append(d)
            seen_ids.add(d["doc_id"])

    selected_docs = selected_docs[:n_docs]
    print(f"Selected {len(selected_docs)} diverse documents across both datasets.")

    # Process all selected docs
    docs_payload = []

    with torch.no_grad():
        for idx, doc in enumerate(selected_docs):
            doc_id = doc["doc_id"]
            dataset = doc.get("dataset", "UNKNOWN")
            rectype = doc.get("rectype", "UNKNOWN")
            raw_text = doc.get("rectxt", "")

            spans = extract_lines_with_offsets(raw_text)
            samples = build_windowed_samples(spans, window_size=2)

            if not samples:
                continue

            line_is_bp = [False] * len(spans)
            line_probs = [0.0] * len(spans)

            # Batched inference
            batch_size = 64
            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]
                targets = [s.target_text for s in batch]
                contexts = [f"{s.context_before} \n {s.context_after}".strip() for s in batch]

                enc = tokenizer(
                    targets, contexts, padding=True, truncation=True, max_length=256, return_tensors="pt"
                ).to(device)

                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                for s, p in zip(batch, probs, strict=False):
                    p_bp = float(p[1])
                    line_probs[s.line_index] = p_bp
                    if p_bp > 0.50:
                        line_is_bp[s.line_index] = True

            clinical_chunks = []
            lines_data = []
            bp_count = 0
            clin_count = 0

            for i, span in enumerate(spans):
                is_bp = line_is_bp[i]
                if is_bp:
                    bp_count += 1
                else:
                    clin_count += 1
                    clinical_chunks.append(span.text)

                lines_data.append({
                    "line_index": i,
                    "is_bp": is_bp,
                    "prob_bp": round(line_probs[i], 3),
                    "start_char": span.start_char,
                    "end_char": span.end_char,
                    "text": span.text,
                })

            trimmed_text = "\n".join(c for c in clinical_chunks if c.strip())
            raw_chars = len(raw_text)
            trimmed_chars = len(trimmed_text)
            reduc = ((raw_chars - trimmed_chars) / max(1, raw_chars)) * 100.0

            docs_payload.append({
                "doc_id": doc_id,
                "dataset": dataset,
                "rectype": rectype,
                "raw_chars": raw_chars,
                "trimmed_chars": trimmed_chars,
                "reduction_pct": reduc,
                "total_lines": len(spans),
                "clinical_lines": clin_count,
                "bp_lines": bp_count,
                "trimmed_text": trimmed_text,
                "lines": lines_data,
            })

            print(f"[{idx+1:2d}/{len(selected_docs)}] {doc_id:15s} ({rectype:10s}, {dataset:5s}) -> Saved: {reduc:4.1f}% | Preserved: {clin_count}/{len(spans)} lines")

    # Generate HTML
    html_content = HTML_TEMPLATE.replace("DOCS_DATA_PLACEHOLDER", json.dumps(docs_payload))
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"\nHTML visualizer generated at: {output_html} ({len(html_content):,} bytes)")

    # Also copy to brain artifact directory for convenience
    brain_artifact = Path("C:/Users/franc/.gemini/antigravity/brain/3956609e-7690-4e6c-b9a8-f45a4aeeb305/benchmark_viewer.html")
    try:
        shutil.copyfile(output_html, brain_artifact)
        print(f"Copied to brain artifacts: {brain_artifact}")
    except Exception as e:
        print(f"Brain copy notice: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_file", default="data/processed/benchmark_sample.jsonl")
    parser.add_argument("--model_dir", default="artifacts/active_learning/student_v2/best_model")
    parser.add_argument("--n_docs", type=int, default=50)
    parser.add_argument("--output_html", default="artifacts/benchmark_viewer.html")
    args = parser.parse_args()

    generate_benchmark_viewer(
        sample_file=args.sample_file,
        model_dir=args.model_dir,
        n_docs=args.n_docs,
        output_html=args.output_html,
    )

