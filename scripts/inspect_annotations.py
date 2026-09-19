#!/usr/bin/env python3
"""Interactive inspector, persistent editor, and HTTP server for LLM teacher annotations."""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import urllib.parse
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def load_doc_by_index(jsonl_path: str, target_idx: int) -> dict | None:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == target_idx:
                return json.loads(line)
    return None


def load_doc_by_id(jsonl_path: str, doc_id: str) -> dict | None:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            if doc.get("doc_id") == doc_id:
                return doc
    return None


def clean_mail_merge_artifacts(input_jsonl: str, output_jsonl: str | None = None) -> tuple[int, int]:
    """Sanitizes Word publipostage / mail-merge template placeholders in an annotated JSONL file."""
    if output_jsonl is None:
        output_jsonl = input_jsonl

    pattern = re.compile(
        r"(«|MERGEFIELD|«Adresse|«Libellé|«Code_postal|«Prénom»|«Nom»|«Commune»|«Titre»)",
        re.IGNORECASE,
    )
    docs_modified = 0
    lines_cleaned = 0

    records = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            doc = json.loads(line)
            modified = False

            new_bp_indices = set()
            for l in doc.get("lines", []):
                if l.get("is_boilerplate"):
                    new_bp_indices.add(l["line_index"])
                elif pattern.search(l.get("text", "")):
                    l["is_boilerplate"] = True
                    new_bp_indices.add(l["line_index"])
                    lines_cleaned += 1
                    modified = True

            if modified:
                docs_modified += 1
                if new_bp_indices:
                    sorted_indices = sorted(new_bp_indices)
                    new_spans = []
                    start = sorted_indices[0]
                    prev = sorted_indices[0]
                    for idx in sorted_indices[1:]:
                        if idx == prev + 1:
                            prev = idx
                        else:
                            new_spans.append({
                                "start_line": start,
                                "end_line": prev,
                                "category": "form_metadata",
                                "reason": "Mail-merge / Word placeholder",
                            })
                            start = idx
                            prev = idx
                    new_spans.append({
                        "start_line": start,
                        "end_line": prev,
                        "category": "form_metadata",
                        "reason": "Mail-merge / Word placeholder",
                    })
                    old_spans = doc.get("boilerplate_spans", [])
                    for ns in new_spans:
                        for os in old_spans:
                            if os["start_line"] <= ns["start_line"] <= os["end_line"]:
                                ns["category"] = os["category"]
                                ns["reason"] = os["reason"]
                                break
                    doc["boilerplate_spans"] = new_spans

            records.append(doc)

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return docs_modified, lines_cleaned


def print_doc_terminal(doc: dict, max_lines: int | None = None):
    doc_id = doc.get("doc_id")
    rectype = doc.get("rectype")
    recdate = doc.get("recdate")
    sejum = doc.get("sejum")
    lines = doc.get("lines", [])
    spans = doc.get("boilerplate_spans", [])

    print("=" * 80)
    print(f"DOCUMENT ID: {doc_id}")
    print(f"Type: {rectype} | Date: {recdate} | Service: {sejum} | Total lines: {len(lines)}")
    print("=" * 80)
    print("IDENTIFIED BOILERPLATE SPANS:")
    if not spans:
        print("  (None - Document is 100% clinical content)")
    for s in spans:
        cat = s.get("category")
        start = s.get("start_line")
        end = s.get("end_line")
        reason = s.get("reason", "")
        print(f"  Lines [{start:3d}..{end:3d}] -> {cat:18s} | {reason}")

    print("\n" + "-" * 80)
    print("DOCUMENT CONTENT (Lines marked with [BP: Category] or [CLINICAL]):")
    print("-" * 80)

    line_to_bp = {}
    for s in spans:
        for idx in range(s.get("start_line", 0), s.get("end_line", -1) + 1):
            line_to_bp[idx] = s

    for i, l in enumerate(lines):
        if max_lines and i >= max_lines:
            print(f"... and {len(lines) - max_lines} more lines (use --all to see full document) ...")
            break

        text = l.get("text", "")
        if i in line_to_bp:
            span = line_to_bp[i]
            cat = span.get("category")
            print(f"[{i:3d}] [BP: {cat:16s}] {text}")
        else:
            print(f"[{i:3d}] [    CLINICAL    ] {text}")
    print("=" * 80)


def build_html_content(jsonl_path: str, num_docs: int = 1000) -> str:
    """Builds the HTML page with localStorage persistence and optional live server sync."""
    docs = []
    categories = Counter()
    total_lines = 0
    total_bp_lines = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            doc = json.loads(line)
            total_lines += doc.get("line_count", 0)
            for s in doc.get("boilerplate_spans", []):
                categories[s.get("category")] += 1
            for l in doc.get("lines", []):
                if l.get("is_boilerplate"):
                    total_bp_lines += 1
            if len(docs) < num_docs:
                docs.append(doc)

    pct_bp = (total_bp_lines / max(1, total_lines)) * 100
    pct_clin = 100 - pct_bp

    html_content = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>redsan-doc-trimmer — Persistent Teacher Annotations Editor</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; padding-bottom: 90px; }}
  .container {{ max-width: 1350px; margin: 0 auto; }}
  h1 {{ font-size: 24px; margin-bottom: 4px; color: #38bdf8; }}
  .header-desc {{ color: #94a3b8; font-size: 14px; margin-bottom: 12px; }}
  .keyboard-cheatsheet {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 12px; color: #94a3b8; background: rgba(30, 41, 59, 0.8); padding: 8px 14px; border-radius: 6px; margin-bottom: 14px; border: 1px solid #334155; align-items: center; }}
  .kbd {{ background: #334155; color: #38bdf8; padding: 2px 7px; border-radius: 4px; font-family: monospace; font-size: 11px; font-weight: bold; border: 1px solid #475569; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin-bottom: 14px; }}
  .stat-card {{ background: #1e293b; border-radius: 8px; padding: 12px 16px; border: 1px solid #334155; }}
  .stat-val {{ font-size: 22px; font-weight: bold; color: #f1f5f9; }}
  .stat-lbl {{ font-size: 11px; color: #94a3b8; text-transform: uppercase; margin-top: 4px; }}
  .persist-bar {{ background: #1e293b; border: 1px solid #38bdf8; border-radius: 8px; padding: 10px 16px; margin-bottom: 14px; display: flex; justify-content: space-between; align-items: center; font-size: 13px; }}
  .selector-bar {{ display: flex; gap: 10px; margin-bottom: 16px; align-items: center; background: #1e293b; padding: 10px 16px; border-radius: 8px; flex-wrap: wrap; }}
  select {{ background: #0f172a; color: #f8fafc; border: 1px solid #475569; padding: 8px 12px; border-radius: 6px; font-size: 13px; flex-grow: 1; min-width: 320px; }}
  button {{ background: #0284c7; color: white; border: none; padding: 8px 14px; border-radius: 6px; cursor: pointer; font-weight: 500; font-size: 13px; transition: all 0.15s; user-select: none; }}
  button:hover {{ background: #0369a1; }}
  button.btn-success {{ background: #059669; }}
  button.btn-success:hover {{ background: #047857; }}
  button.btn-danger {{ background: #dc2626; }}
  button.btn-danger:hover {{ background: #b91c1c; }}
  .doc-meta {{ background: #1e293b; border-radius: 8px 8px 0 0; padding: 16px; border: 1px solid #334155; border-bottom: none; }}
  .doc-view {{ background: #0b1120; border: 1px solid #334155; border-radius: 0 0 8px 8px; font-family: 'Consolas', 'Monaco', monospace; font-size: 13px; max-height: 650px; overflow-y: auto; user-select: none; }}
  .line-row {{ display: flex; padding: 3px 12px; line-height: 1.5; border-bottom: 1px solid rgba(255,255,255,0.03); align-items: center; cursor: pointer; }}
  .line-row:hover {{ background: rgba(56, 189, 248, 0.08) !important; }}
  .line-num {{ width: 45px; color: #64748b; flex-shrink: 0; text-align: right; margin-right: 16px; user-select: none; }}
  .line-badge {{ width: 140px; flex-shrink: 0; font-size: 11px; padding: 2px 6px; border-radius: 4px; margin-right: 12px; text-align: center; cursor: pointer; user-select: none; transition: transform 0.1s; }}
  .line-badge:hover {{ transform: scale(1.05); }}
  .badge-clin {{ background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }}
  .badge-bp-header {{ background: #7f1d1d; color: #fca5a5; border: 1px solid #dc2626; }}
  .badge-bp-sig {{ background: #78350f; color: #fcd34d; border: 1px solid #d97706; }}
  .badge-bp-footer {{ background: #4c1d95; color: #d8b4fe; border: 1px solid #7c3aed; }}
  .badge-bp-form {{ background: #312e81; color: #a5b4fc; border: 1px solid #4f46e5; }}
  .badge-bp-other {{ background: #374151; color: #d1d5db; border: 1px solid #6b7280; }}
  .line-text {{ white-space: pre-wrap; word-break: break-word; flex-grow: 1; }}
  .bp-text {{ color: #94a3b8; text-decoration: line-through 1px rgba(239, 68, 68, 0.4); }}
  .clin-text {{ color: #ffffff; font-weight: 500; }}
  .span-summary {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }}
  .span-pill {{ font-size: 12px; padding: 4px 10px; border-radius: 12px; background: #334155; color: #cbd5e1; }}
  .badge-modified {{ border: 2px solid #38bdf8 !important; box-shadow: 0 0 6px rgba(56, 189, 248, 0.5); }}
  .floating-dock {{ position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); background: rgba(15, 23, 42, 0.94); backdrop-filter: blur(12px); border: 1px solid #38bdf8; border-radius: 36px; padding: 8px 20px; display: flex; align-items: center; gap: 12px; box-shadow: 0 12px 35px rgba(0,0,0,0.75); z-index: 1000; }}
  .floating-dock button {{ padding: 7px 16px; border-radius: 20px; font-size: 13px; }}
</style>
</head>
<body>
<div class="container">
  <h1>Persistent Annotations Inspector & Editor</h1>
  <div class="header-desc">
    Reviewing: <code>{Path(jsonl_path).name}</code>. Click any badge or <b>drag across multiple lines</b> to paint.
  </div>

  <div class="keyboard-cheatsheet">
    <span><span class="kbd">&larr;</span> / <span class="kbd">&rarr;</span> or <span class="kbd">A</span> / <span class="kbd">D</span> : Prev / Next Doc</span>
    <span><span class="kbd">Mouse Drag</span> : Paint multiple lines</span>
    <span><span class="kbd">Space</span> or <span class="kbd">F</span> : Flip hovered line</span>
    <span><span class="kbd">Ctrl+S</span> : Save All</span>
  </div>

  <div class="persist-bar">
    <div id="persist-status">
      💾 <b>Auto-Save Active:</b> All line toggles are automatically saved in browser storage. Refreshing (Ctrl+R) or closing your browser will <b>never</b> lose your work.
    </div>
    <div style="display: flex; gap: 8px;">
      <button onclick="resetCurrentDoc()" style="background: #475569; font-size: 11px;">Reset Current Doc</button>
      <button onclick="clearAllOverrides()" class="btn-danger" style="font-size: 11px;">Clear All Overrides</button>
    </div>
  </div>

  <div class="stats-grid">
    <div class="stat-card">
      <div class="stat-val">{len(docs)}</div>
      <div class="stat-lbl">Documents</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color: #6ee7b7;">{pct_clin:.1f}%</div>
      <div class="stat-lbl">Clinical Lines</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color: #fca5a5;">{pct_bp:.1f}%</div>
      <div class="stat-lbl">Boilerplate Lines</div>
    </div>
    <div class="stat-card">
      <div class="stat-val" style="color: #38bdf8;" id="ovr-counter">0</div>
      <div class="stat-lbl">Manual Overrides Saved</div>
    </div>
  </div>

  <div class="selector-bar">
    <select id="doc-select" onchange="renderDoc(this.value)">
"""
    for i, d in enumerate(docs):
        d_id = d.get("doc_id")
        d_type = d.get("rectype", "UNKNOWN")
        d_date = d.get("recdate", "")
        d_um = d.get("sejum", "")
        n_spans = len(d.get("boilerplate_spans", []))
        html_content += f'      <option value="{i}">#{i+1:03d} | {d_id} | {d_type} | {d_date} | {d_um} ({n_spans} spans)</option>\n'

    html_content += """    </select>
    <button onclick="prevDoc()">&larr; Prev (A)</button>
    <button onclick="nextDoc()">Next (D) &rarr;</button>
    <button onclick="randomDoc()">🎲 Random</button>
    <button class="btn-success" id="btn-save-disk" onclick="saveToDiskOrDownload()">💾 Save & Export Dataset</button>
  </div>

  <div id="viewer-container"></div>
</div>

<!-- Bottom Floating Dock -->
<div class="floating-dock">
  <button onclick="prevDoc()">&larr; Prev (A / &larr;)</button>
  <span id="dock-counter" style="font-weight: 600; font-size: 13px; color: #f8fafc; min-width: 140px; text-align: center;">Doc 1 / 500</span>
  <button onclick="nextDoc()">Next (D / &rarr;) &rarr;</button>
  <button class="btn-success" onclick="saveToDiskOrDownload()">💾 Save All (Ctrl+S)</button>
</div>

<script>
const docsData = """
    html_content += json.dumps(docs, ensure_ascii=False) + ";\n"

    html_content += """
const STORAGE_KEY = 'redsan_trimmer_overrides_' + window.location.pathname;
let currentDocIndex = 0;
let userOverrides = {};
let isMouseDown = false;
let paintTargetIsBp = null;
let hoveredLineIdx = null;

try {
  userOverrides = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}');
} catch (e) {
  console.error('Failed to load overrides from localStorage', e);
}

function applyStoredOverrides() {
  for (const doc of docsData) {
    const docOvr = userOverrides[doc.doc_id];
    if (docOvr) {
      for (const [lineIdxStr, isBp] of Object.entries(docOvr)) {
        const idx = parseInt(lineIdxStr, 10);
        if (doc.lines && doc.lines[idx]) {
          doc.lines[idx].is_boilerplate = isBp;
        }
      }
      rebuildSpans(doc);
    }
  }
  updateStatsBanner();
}

function updateStatsBanner() {
  const counterEl = document.getElementById('ovr-counter');
  let totalLineOverrides = 0;
  for (const docId in userOverrides) {
    const lines = Object.keys(userOverrides[docId]);
    totalLineOverrides += lines.length;
  }
  if (counterEl) {
    counterEl.textContent = totalLineOverrides;
  }
}

function getBadgeClass(cat) {
  if (!cat) return 'badge-clin';
  if (cat.includes('header')) return 'badge-bp-header';
  if (cat.includes('sig')) return 'badge-bp-sig';
  if (cat.includes('footer')) return 'badge-bp-footer';
  if (cat.includes('form') || cat.includes('meta')) return 'badge-bp-form';
  return 'badge-bp-other';
}

function rebuildSpans(doc) {
  const bpIndices = [];
  doc.lines.forEach((l, idx) => {
    if (l.is_boilerplate) bpIndices.push(idx);
  });

  if (bpIndices.length === 0) {
    doc.boilerplate_spans = [];
    return;
  }

  const newSpans = [];
  let start = bpIndices[0];
  let prev = bpIndices[0];

  for (let i = 1; i < bpIndices.length; i++) {
    const idx = bpIndices[i];
    if (idx === prev + 1) {
      prev = idx;
    } else {
      newSpans.push({
        start_line: start,
        end_line: prev,
        category: 'form_metadata',
        reason: 'Manually refined'
      });
      start = idx;
      prev = idx;
    }
  }
  newSpans.push({
    start_line: start,
    end_line: prev,
    category: 'form_metadata',
    reason: 'Manually refined'
  });

  doc.boilerplate_spans = newSpans;
}

function applySingleLineState(lineIdx, targetIsBp) {
  const doc = docsData[currentDocIndex];
  if (!doc || !doc.lines || !doc.lines[lineIdx]) return;
  const line = doc.lines[lineIdx];
  if (line.is_boilerplate === targetIsBp) return;

  line.is_boilerplate = targetIsBp;

  // Persist to userOverrides
  if (!userOverrides[doc.doc_id]) userOverrides[doc.doc_id] = {};
  userOverrides[doc.doc_id][lineIdx] = targetIsBp;

  // If live server is active, send update
  if (window.location.protocol.startsWith('http')) {
    fetch('/api/save_line', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        doc_id: doc.doc_id,
        line_index: lineIdx,
        is_boilerplate: targetIsBp
      })
    }).catch(err => console.warn('Server sync error:', err));
  }

  // Update DOM in-place
  const badge = document.getElementById(`badge-${lineIdx}`);
  const textEl = document.getElementById(`text-${lineIdx}`);
  if (badge && textEl) {
    if (targetIsBp) {
      badge.className = 'line-badge badge-bp-form badge-modified';
      badge.textContent = 'BOILERPLATE';
      textEl.className = 'line-text bp-text';
    } else {
      badge.className = 'line-badge badge-clin badge-modified';
      badge.textContent = 'CLINICAL';
      textEl.className = 'line-text clin-text';
    }
  }
}

function startDrag(e, lineIdx) {
  e.preventDefault();
  isMouseDown = true;
  const doc = docsData[currentDocIndex];
  if (!doc || !doc.lines || !doc.lines[lineIdx]) return;
  paintTargetIsBp = !doc.lines[lineIdx].is_boilerplate;
  applySingleLineState(lineIdx, paintTargetIsBp);
}

function onLineHover(lineIdx) {
  hoveredLineIdx = lineIdx;
  if (isMouseDown && paintTargetIsBp !== null) {
    applySingleLineState(lineIdx, paintTargetIsBp);
  }
}

function endDrag() {
  if (isMouseDown) {
    isMouseDown = false;
    paintTargetIsBp = null;
    const doc = docsData[currentDocIndex];
    if (doc) {
      rebuildSpans(doc);
      updateSpanSummary(doc);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(userOverrides));
      updateStatsBanner();
    }
  }
}

window.addEventListener('mouseup', endDrag);

function toggleLine(lineIdx) {
  const doc = docsData[currentDocIndex];
  if (!doc || !doc.lines || !doc.lines[lineIdx]) return;
  const targetIsBp = !doc.lines[lineIdx].is_boilerplate;
  applySingleLineState(lineIdx, targetIsBp);
  rebuildSpans(doc);
  updateSpanSummary(doc);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(userOverrides));
  updateStatsBanner();
}

function updateSpanSummary(doc) {
  const summaryEl = document.getElementById('span-summary-container');
  if (!summaryEl) return;
  const spans = doc.boilerplate_spans || [];
  let spansHtml = '';
  if (spans.length === 0) {
    spansHtml = '<span class="span-pill" style="background: #064e3b; color: #6ee7b7;">Pure Clinical (0 boilerplate)</span>';
  } else {
    for (const s of spans) {
      spansHtml += `<span class="span-pill">Lines [${s.start_line}..${s.end_line}] <b>${s.category}</b>: <i>${s.reason || ''}</i></span>`;
    }
  }
  summaryEl.innerHTML = spansHtml;
}

function renderDoc(index) {
  currentDocIndex = parseInt(index, 10);
  const doc = docsData[currentDocIndex];
  if (!doc) return;

  const sel = document.getElementById('doc-select');
  if (sel) sel.selectedIndex = currentDocIndex;

  const dockCounter = document.getElementById('dock-counter');
  if (dockCounter) dockCounter.textContent = `Doc ${currentDocIndex + 1} / ${docsData.length}`;

  const spans = doc.boilerplate_spans || [];
  const lines = doc.lines || [];
  const docOvr = userOverrides[doc.doc_id] || {};

  let spansHtml = '';
  if (spans.length === 0) {
    spansHtml = '<span class="span-pill" style="background: #064e3b; color: #6ee7b7;">Pure Clinical (0 boilerplate)</span>';
  } else {
    for (const s of spans) {
      spansHtml += `<span class="span-pill">Lines [${s.start_line}..${s.end_line}] <b>${s.category}</b>: <i>${s.reason || ''}</i></span>`;
    }
  }

  let linesHtml = '';
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    const isBp = l.is_boilerplate === true;
    const cat = isBp ? 'form_metadata' : null;
    const isOverridden = (i in docOvr);
    const modCls = isOverridden ? 'badge-modified' : '';
    const badgeCls = isBp ? getBadgeClass(cat) : 'badge-clin';
    const badgeText = isBp ? 'BOILERPLATE' : 'CLINICAL';
    const textCls = isBp ? 'bp-text' : 'clin-text';
    const safeText = l.text ? l.text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;') : '';

    linesHtml += `
      <div class="line-row" id="row-${i}" onmouseenter="onLineHover(${i})" onmouseleave="if(hoveredLineIdx===${i}) hoveredLineIdx=null;">
        <div class="line-num">${i}</div>
        <div class="line-badge ${badgeCls} ${modCls}" id="badge-${i}" onmousedown="startDrag(event, ${i})" onclick="toggleLine(${i})" title="Click or drag to paint">${badgeText}</div>
        <div class="line-text ${textCls}" id="text-${i}">${safeText}</div>
      </div>
    `;
  }

  const container = document.getElementById('viewer-container');
  container.innerHTML = `
    <div class="doc-meta">
      <div style="display: flex; justify-content: space-between; align-items: baseline;">
        <h3 style="margin: 0; color: #f8fafc;">${doc.doc_id}</h3>
        <span style="font-size: 13px; color: #94a3b8;">${lines.length} lines | ${doc.char_length} chars</span>
      </div>
      <div style="font-size: 13px; color: #cbd5e1; margin-top: 6px;">
        <b>Document Type:</b> ${doc.rectype || 'N/A'} | <b>Date:</b> ${doc.recdate || 'N/A'} | <b>Service:</b> ${doc.sejum || 'N/A'}
      </div>
      <div class="span-summary" id="span-summary-container">${spansHtml}</div>
    </div>
    <div class="doc-view">${linesHtml}</div>
  `;
}

// Global Keyboard Navigation
window.addEventListener('keydown', (e) => {
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;

  if (e.key === 'ArrowRight' || e.key.toLowerCase() === 'd') {
    e.preventDefault();
    nextDoc();
  } else if (e.key === 'ArrowLeft' || e.key.toLowerCase() === 'a') {
    e.preventDefault();
    prevDoc();
  } else if (e.code === 'Space' || e.key.toLowerCase() === 'f') {
    if (hoveredLineIdx !== null) {
      e.preventDefault();
      toggleLine(hoveredLineIdx);
    }
  } else if (e.key.toLowerCase() === 's' && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    saveToDiskOrDownload();
  }
});

function prevDoc() {
  if (currentDocIndex > 0) {
    renderDoc(currentDocIndex - 1);
  }
}

function nextDoc() {
  if (currentDocIndex < docsData.length - 1) {
    renderDoc(currentDocIndex + 1);
  }
}

function randomDoc() {
  const randIdx = Math.floor(Math.random() * docsData.length);
  renderDoc(randIdx);
}

function resetCurrentDoc() {
  const doc = docsData[currentDocIndex];
  if (!doc) return;
  if (userOverrides[doc.doc_id]) {
    delete userOverrides[doc.doc_id];
    localStorage.setItem(STORAGE_KEY, JSON.stringify(userOverrides));
    window.location.reload();
  }
}

function clearAllOverrides() {
  if (confirm('Clear all manual overrides? This will reset all documents to original teacher labels.')) {
    localStorage.removeItem(STORAGE_KEY);
    window.location.reload();
  }
}

function clearAllOverrides() {
  if (confirm('Clear all manual overrides? This will reset all documents to original teacher labels.')) {
    localStorage.removeItem(STORAGE_KEY);
    window.location.reload();
  }
}

function saveToDiskOrDownload() {
  if (window.location.protocol.startsWith('http')) {
    const btn = document.getElementById('btn-save-disk');
    btn.textContent = 'Saving to disk...';
    fetch('/api/save_all', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(docsData)
    })
    .then(r => r.json())
    .then(data => {
      btn.textContent = '✓ Saved Directly to Disk!';
      setTimeout(() => { btn.textContent = '💾 Save & Export Refined Dataset'; }, 3000);
      alert('Successfully saved all annotations directly to disk!');
    })
    .catch(err => {
      alert('Error saving to server. Falling back to browser download.');
      exportRefinedJsonl();
    });
  } else {
    exportRefinedJsonl();
  }
}

function exportRefinedJsonl() {
  let content = '';
  for (const doc of docsData) {
    content += JSON.stringify(doc) + '\\n';
  }
  const blob = new Blob([content], { type: 'application/jsonl;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', 'refined_annotations.jsonl');
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

window.onload = () => {
  applyStoredOverrides();
  renderDoc(0);
};
</script>
</body>
</html>
"""
    return html_content


def generate_html_viewer(jsonl_path: str, output_html: str, num_docs: int = 1000):
    content = build_html_content(jsonl_path, num_docs=num_docs)
    out_file = Path(output_html)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"HTML Inspector generated: {out_file.resolve()}")


def run_local_server(jsonl_path: str, port: int = 8080):
    """Runs a local Python web server that directly saves human annotations to disk on every click."""
    jsonl_p = Path(jsonl_path).resolve()
    print(f"Loading documents from {jsonl_p}...")

    class AnnotationServer(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ("/", "/index.html"):
                html_text = build_html_content(str(jsonl_p), num_docs=1000)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html_text.encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length).decode("utf-8")

            if self.path == "/api/save_line":
                data = json.loads(post_data)
                doc_id = data.get("doc_id")
                line_idx = data.get("line_index")
                is_bp = data.get("is_boilerplate")

                # Update in file
                updated_records = []
                with open(jsonl_p, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        if rec.get("doc_id") == doc_id:
                            if "lines" in rec and line_idx < len(rec["lines"]):
                                rec["lines"][line_idx]["is_boilerplate"] = is_bp
                                # Rebuild spans
                                bp_indices = [i for i, l in enumerate(rec["lines"]) if l.get("is_boilerplate")]
                                if not bp_indices:
                                    rec["boilerplate_spans"] = []
                                else:
                                    sorted_idx = sorted(bp_indices)
                                    spans = []
                                    s = sorted_idx[0]
                                    p = sorted_idx[0]
                                    for idx in sorted_idx[1:]:
                                        if idx == p + 1:
                                            p = idx
                                        else:
                                            spans.append({"start_line": s, "end_line": p, "category": "form_metadata", "reason": "Manually refined"})
                                            s = idx
                                            p = idx
                                    spans.append({"start_line": s, "end_line": p, "category": "form_metadata", "reason": "Manually refined"})
                                    rec["boilerplate_spans"] = spans
                        updated_records.append(rec)

                with open(jsonl_p, "w", encoding="utf-8") as f:
                    for rec in updated_records:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')

            elif self.path == "/api/save_all":
                updated_docs = json.loads(post_data)
                with open(jsonl_p, "w", encoding="utf-8") as f:
                    for doc in updated_docs:
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, format, *args):
            # Keep console clean
            pass

    server_address = ("", port)
    httpd = HTTPServer(server_address, AnnotationServer)
    print(f"\n========================================================")
    print(f" Live Annotation Server Running at: http://localhost:{port}")
    print(f" Target File on Disk: {jsonl_p}")
    print(f" Every click in the browser saves directly to disk!")
    print(f" Press Ctrl+C in terminal to stop.")
    print(f"========================================================\n")
    try:
        import webbrowser
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect & refine teacher annotations")
    parser.add_argument(
        "--file",
        type=str,
        default="data/processed/mined_hard_cases_annotated_500.jsonl",
        help="Path to annotated JSONL",
    )
    parser.add_argument("--doc_id", type=str, default=None, help="Inspect specific document ID")
    parser.add_argument("--index", type=int, default=None, help="Inspect document by 0-indexed position")
    parser.add_argument("--random", action="store_true", help="Inspect a random document")
    parser.add_argument("--all", action="store_true", help="Print all lines without truncation")
    parser.add_argument("--html", type=str, default=None, help="Generate interactive HTML report file")
    parser.add_argument("--num_docs", type=int, default=1000, help="Number of docs to include in HTML viewer")
    parser.add_argument(
        "--clean_mail_merge",
        action="store_true",
        help="Automatically sanitize Word mail-merge template placeholders in JSONL",
    )
    parser.add_argument("--output", type=str, default=None, help="Output path for sanitized JSONL")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run local web server with automatic live disk sync on every click",
    )
    parser.add_argument("--port", type=int, default=8080, help="Port for local web server")
    args = parser.parse_args()

    if args.clean_mail_merge:
        target_out = args.output or args.file
        docs_mod, lines_mod = clean_mail_merge_artifacts(args.file, target_out)
        print(f"Successfully cleaned mail-merge placeholders: {lines_mod} lines across {docs_mod} documents updated in {target_out}.")
    elif args.serve:
        run_local_server(args.file, port=args.port)
    elif args.html:
        generate_html_viewer(args.file, args.html, num_docs=args.num_docs)
    elif args.doc_id:
        doc = load_doc_by_id(args.file, args.doc_id)
        if doc:
            print_doc_terminal(doc, max_lines=None if args.all else 60)
        else:
            print(f"Document {args.doc_id} not found.")
    elif args.random:
        with open(args.file, "r", encoding="utf-8") as f:
            total = sum(1 for line in f if line.strip())
        rand_idx = random.randint(0, total - 1)
        doc = load_doc_by_index(args.file, rand_idx)
        print(f"(Displaying random document #{rand_idx + 1} of {total})")
        print_doc_terminal(doc, max_lines=None if args.all else 60)
    else:
        idx = args.index if args.index is not None else 0
        doc = load_doc_by_index(args.file, idx)
        if doc:
            print_doc_terminal(doc, max_lines=None if args.all else 60)
        else:
            print("No document found.")
