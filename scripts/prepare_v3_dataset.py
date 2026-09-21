"""Builds the enriched dataset for Student v3 DrBERT fine-tuning.

Includes:
1. Genuine transport vouchers (ORDON7 + BT) with 100% boilerplate labeling.
2. Clinical letters with FORMCHECKBOX (CRH2AB, LDL2024) with strict preservation
   of clinical narrative and clinical anchors (Poids, Motif, Conclusion).
3. Targeted trailing metadata lines (POST_BP_PATTERNS: copies, typist initials, lone page numbers).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from redsan_doc_trimmer.dataset import extract_lines_with_offsets

POST_BP_PATTERNS = [
    re.compile(r"^\s*Copie\s+(?:adress[ée]e|transmise)\s+à\b.*$", re.IGNORECASE),
    re.compile(r"^\s*[A-Z]{2,4}\s*/\s*[A-Z]{2,4}\s*$"),  # Typist initials e.g. FM / AB
    re.compile(r"^\s*(?:Page\s+)?\d+\s*(?:/\s*\d+)?\s*$", re.IGNORECASE),  # Lone page numbers
]

CLINICAL_ANCHOR_PATTERNS = [
    re.compile(r"\bpoids\b", re.IGNORECASE),
    re.compile(r"\btaille\b", re.IGNORECASE),
    re.compile(r"\bmotif\b", re.IGNORECASE),
    re.compile(r"\bconclusion\b", re.IGNORECASE),
    re.compile(r"\bdiagnostic\b", re.IGNORECASE),
    re.compile(r"\bant[eé]c[eé]dent", re.IGNORECASE),
    re.compile(r"\btraitement\b", re.IGNORECASE),
    re.compile(r"\bexamen\b", re.IGNORECASE),
    re.compile(r"\bbiologie\b", re.IGNORECASE),
    re.compile(r"\bhospitalis[eé]", re.IGNORECASE),
    re.compile(r"\bsortie\b", re.IGNORECASE),
    re.compile(r"\bclinique\b", re.IGNORECASE),
]

HEADER_REGEX = re.compile(
    r"(H[oô]pitau|Centre Hospitalier|CHRU|H[oô]pital|P[oô]le |D[eé]partement de |Service d|CH |Adresse postale|Fax\s*:|T[eé]l\s*:|\[PHONE\]|\[ADDRESS)",
    re.IGNORECASE,
)
SIG_REGEX = re.compile(
    r"^(Docteur|Dr\b|Pr\b|Professeur|Secr[eé]tariat|Chef de Clinique|Bien confraternellement|Confraternellement|Sinc[eè]res salutations)",
    re.IGNORECASE,
)

def annotate_voucher(doc: dict) -> dict:
    """Every non-empty line of a transport voucher is administrative boilerplate."""
    text = doc["text"]
    spans = extract_lines_with_offsets(text)
    lines_out = []
    for i, s in enumerate(spans):
        lines_out.append({
            "line_index": i,
            "start_char": s.start_char,
            "end_char": s.end_char,
            "text": s.text,
            "is_boilerplate": True,
        })
    return {
        "doc_id": f"voucher_{doc['id']}",
        "rectype": doc.get("rectype", "ORDON7"),
        "char_length": len(text),
        "line_count": len(spans),
        "lines": lines_out,
    }

def annotate_clinical_letter(doc: dict) -> dict:
    """Annotates clinical letters: headers, tick boxes, signatures, and typist initials

    are boilerplate. All clinical narrative and anchors are strictly False.
    """
    text = doc["text"]
    spans = extract_lines_with_offsets(text)
    n = len(spans)
    max_header = min(n, max(5, int(n * 0.25)))
    min_footer = max(0, int(n * 0.80))

    lines_out = []
    for i, s in enumerate(spans):
        line = s.text.strip()
        is_bp = False

        if not line:
            is_bp = False
        else:
            # Check for strong clinical anchors first: if present, NEVER boilerplate
            has_clinical_anchor = any(p.search(line) for p in CLINICAL_ANCHOR_PATTERNS)

            if has_clinical_anchor:
                is_bp = False
            elif i < max_header and (HEADER_REGEX.search(line) or "FORMCHECKBOX" in line):
                is_bp = True
            elif i >= min_footer and (SIG_REGEX.search(line) or "[DOCTOR]" in line):
                is_bp = True
            elif any(pat.search(line) for pat in POST_BP_PATTERNS):
                is_bp = True
            else:
                is_bp = False

        lines_out.append({
            "line_index": i,
            "start_char": s.start_char,
            "end_char": s.end_char,
            "text": s.text,
            "is_boilerplate": is_bp,
        })

    return {
        "doc_id": f"letter_{doc['id']}",
        "rectype": doc.get("rectype", "CRH2AB"),
        "char_length": len(text),
        "line_count": len(spans),
        "lines": lines_out,
    }

def build_v3_dataset(
    payload_path: str = r"C:\Users\franc\AppData\Local\Temp\denut_docs_payload.json",
    seed_v2_path: str = "data/processed/augmented_dataset_v2.jsonl",
    output_path: str = "data/processed/augmented_dataset_v3.jsonl",
    n_vouchers: int = 250,
    n_letters: int = 80,
):
    print(f"Loading raw payload from {payload_path}...")
    with open(payload_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    # 1. Select genuine vouchers (ORDON7 + FORMCHECKBOX)
    vouchers = [
        d for d in docs
        if d.get("rectype") == "ORDON7" and "FORMCHECKBOX" in d.get("text", "")
    ][:n_vouchers]
    print(f"Annotating {len(vouchers)} transport vouchers as 100% boilerplate...")

    annotated_vouchers = [annotate_voucher(v) for v in vouchers]

    # 2. Select clinical letters with FORMCHECKBOX (CRH2AB, LDL2024)
    letters = [
        d for d in docs
        if d.get("rectype") in ("CRH2AB", "LDL2024") and "FORMCHECKBOX" in d.get("text", "")
    ][:n_letters]
    print(f"Annotating {len(letters)} clinical letters with anchor protection...")
    annotated_letters = [annotate_clinical_letter(l) for l in letters]

    # 3. Create synthetic edge cases for POST_BP_PATTERNS (to reinforce typist initials & copies)
    synthetic_docs = []
    synthetic_templates = [
        ("Copie adressée à : Dr Martin, CHU de Rouen", True),
        ("Copie transmise à : Docteur Dupont", True),
        ("Copie adressée au médecin traitant", True),
        ("FM / AB", True),
        ("SR / CD", True),
        ("Page 1/2", True),
        ("Page 2/2", True),
        ("1/3", True),
        ("2/3", True),
        ("3/3", True),
    ]

    for idx, (target_line, is_bp) in enumerate(synthetic_templates):
        doc_text = f"Compte-rendu d'hospitalisation.\nPatient suivi pour pneumopathie.\n{target_line}\nFin de document."
        spans = extract_lines_with_offsets(doc_text)
        s_lines = []
        for i, s in enumerate(spans):
            s_lines.append({
                "line_index": i,
                "start_char": s.start_char,
                "end_char": s.end_char,
                "text": s.text,
                "is_boilerplate": (s.text.strip() == target_line),
            })
        synthetic_docs.append({
            "doc_id": f"synthetic_post_{idx}",
            "rectype": "SYNTHETIC",
            "char_length": len(doc_text),
            "line_count": len(spans),
            "lines": s_lines,
        })

    new_docs = annotated_vouchers + annotated_letters + synthetic_docs
    new_slice_path = Path("data/processed/voucher_and_metadata_annotated.jsonl")
    with open(new_slice_path, "w", encoding="utf-8") as f:
        for d in new_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Wrote {len(new_docs)} new annotated documents to {new_slice_path}")

    # 4. Merge with augmented_dataset_v2.jsonl
    out_file = Path(output_path)
    total_docs = 0
    with open(out_file, "w", encoding="utf-8") as out_f:
        if Path(seed_v2_path).exists():
            print(f"Reading existing seed v2 from {seed_v2_path}...")
            with open(seed_v2_path, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if line.strip():
                        out_f.write(line.strip() + "\n")
                        total_docs += 1
        for d in new_docs:
            out_f.write(json.dumps(d, ensure_ascii=False) + "\n")
            total_docs += 1

    print(f"Successfully generated {out_file} with {total_docs} total annotated documents!")

if __name__ == "__main__":
    build_v3_dataset()
