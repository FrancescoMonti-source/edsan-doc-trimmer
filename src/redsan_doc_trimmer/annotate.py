"""LLM-based teacher annotation module for generating weak supervision labels."""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from redsan_doc_trimmer.dataset import DocumentSpan, extract_lines_with_offsets

logger = logging.getLogger(__name__)

DEFAULT_TEACHER_MODEL = "gpt-5.6-luna"

TEACHER_SYSTEM_PROMPT = """Tu es un expert en traitement de documents médicaux hospitaliers français (dossiers patients EDSAN / RECTXT).
Ton rôle est d'identifier EXCLUSIVEMENT les blocs de lignes qui relèvent STRICTEMENT du cadre administratif et de la mise en page (boilerplate / en-têtes / pieds de page), SANS JAMAIS marquer le contenu clinique.

RÈGLE D'OR MÉDICALE (COÛT ASYMÉTRIQUE DES ERREURS) :
- Supprimer une ligne clinique détruit un fait médical essentiel. C'est inacceptable.
- Conserver une ligne administrative est inoffensif.
- TOUT ce qui relève de : antécédents, anamnèse, mode de vie, symptômes, constantes, examens cliniques, biologie, imagerie, diagnostics, conclusions, prescriptions, résumés opératoires et consignes de sortie DOIT RESTER HORS DES BLOCS BOILERPLATE.
- EN CAS DE DOUTE ou si une ligne est mixte (clinique + administratif), NE PAS LA MARQUER comme boilerplate.

ÉLÉMENTS À MARQUER DANS boilerplate_spans ([start_line, end_line] inclusifs) :
1. hospital_header : En-têtes hospitaliers (CHU, hôpitaux, pôles, adresses postales de l'établissement, numéros de téléphone/fax/email du standard ou secrétariat).
2. signature_block : Blocs de signature isolés en bas de document (ex: "Secrétariat médical", nom/titre du médecin isolé sans texte clinique associé).
3. page_footer : Pieds de page ("Page 1/2", "Tourner SVP", mentions légales de transmission et de confidentialité).
4. form_metadata : Identifiants techniques purs de transmission, codes-barres textuels.

Si le document ne contient aucun élément administratif, retourne une liste boilerplate_spans vide.
"""


class BoilerplateCategory(str, Enum):
    HOSPITAL_HEADER = "hospital_header"
    SIGNATURE_BLOCK = "signature_block"
    PAGE_FOOTER = "page_footer"
    LEGAL_DISCLAIMER = "legal_disclaimer"
    FORM_METADATA = "form_metadata"
    OTHER_BOILERPLATE = "other_boilerplate"


class BoilerplateSpan(BaseModel):
    start_line: int = Field(
        ..., description="Index de la première ligne du bloc de boilerplate (inclusif)"
    )
    end_line: int = Field(
        ..., description="Index de la dernière ligne du bloc de boilerplate (inclusif)"
    )
    category: str = Field(
        default="other_boilerplate",
        description="Catégorie (hospital_header, signature_block, page_footer, legal_disclaimer, form_metadata)",
    )
    reason: str = Field(
        default="",
        description="Brève justification (ex: 'En-tête CHU et coordonnées standard')",
    )


class DocBoilerplateAnnotation(BaseModel):
    doc_id: str = Field(..., description="Identifiant unique du document")
    boilerplate_spans: list[BoilerplateSpan] = Field(
        default_factory=list,
        description="Liste des intervalles de lignes identifiés comme boilerplate",
    )

    def get_boilerplate_line_indices(self, total_lines: int | None = None) -> set[int]:
        """Returns the set of 0-indexed line numbers classified as boilerplate."""
        indices: set[int] = set()
        for span in self.boilerplate_spans:
            start = max(0, span.start_line)
            end = span.end_line
            if total_lines is not None:
                end = min(total_lines - 1, end)
            for idx in range(start, end + 1):
                indices.add(idx)
        return indices

    def apply_to_spans(
        self,
        spans: list[DocumentSpan],
        label_source: str = "llm_teacher",
    ) -> list[DocumentSpan]:
        """Applies boilerplate labels to raw document spans, preserving exact coordinates."""
        bp_indices = self.get_boilerplate_line_indices(len(spans))
        annotated: list[DocumentSpan] = []
        for i, span in enumerate(spans):
            is_bp = i in bp_indices
            annotated.append(
                DocumentSpan(
                    start_char=span.start_char,
                    end_char=span.end_char,
                    text=span.text,
                    is_boilerplate=is_bp,
                    label_source=label_source if is_bp else "clinical",
                )
            )
        return annotated


# Legacy line-by-line schema for backwards compatibility
class LineAnnotation(BaseModel):
    line_index: int = Field(..., description="Index de la ligne (0-indexé)")
    is_boilerplate: bool = Field(
        ..., description="Vrai si la ligne est purement administrative/boilerplate"
    )
    reason: str = Field(..., description="Brève justification")


class DocumentAnnotationResult(BaseModel):
    doc_id: str
    lines: list[LineAnnotation]


def format_doc_for_llm(spans: list[DocumentSpan]) -> str:
    """Formats lines with bracketed line indices for LLM prompt context."""
    formatted_lines: list[str] = []
    for i, span in enumerate(spans):
        text = span.text.strip()
        if not text:
            continue
        formatted_lines.append(f"[{i}] {text}")
    return "\n".join(formatted_lines)


def annotate_document_with_openai(
    doc_id: str,
    raw_text: str,
    client: Any | None = None,
    model: str = DEFAULT_TEACHER_MODEL,
) -> DocBoilerplateAnnotation:
    """Annotates a single document using OpenAI structured outputs with gpt-5.6-luna."""
    if client is None:
        try:
            import openai

            client = openai.OpenAI()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize OpenAI client. Ensure OPENAI_API_KEY is set or provide a client: {e}"
            ) from e

    spans = extract_lines_with_offsets(raw_text)
    formatted_text = format_doc_for_llm(spans)

    user_prompt = f"Document ID: {doc_id}\nNombre total de lignes: {len(spans)}\n\nCONTENU DU DOCUMENT :\n{formatted_text}"

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format=DocBoilerplateAnnotation,
    )

    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError(f"Failed to parse structured output for doc {doc_id}")
    return result


def mock_heuristic_annotate(doc_id: str, raw_text: str) -> DocBoilerplateAnnotation:
    """Heuristic / rule-based annotator for offline testing and development without API keys."""
    spans = extract_lines_with_offsets(raw_text)
    bp_indices: set[int] = set()

    # Common French hospital boilerplate patterns
    header_regex = re.compile(
        r"(H[oô]pitau|Centre Hospitalier|CHRU|H[oô]pital|P[oô]le |D[eé]partement de |Service d|CH |Adresse postale|Fax\s*:|T[eé]l\s*:|\[PHONE\]|\[ADDRESS)",
        re.IGNORECASE,
    )
    footer_regex = re.compile(
        r"(Page \d+/\d+|\bTourner SVP\b|confidentiel|secret professionnel)",
        re.IGNORECASE,
    )
    sig_regex = re.compile(
        r"^(Docteur|Dr\b|Pr\b|Professeur|Secr[eé]tariat|Chef de Clinique|Bien confraternellement|Confraternellement|Sinc[eè]res salutations)",
        re.IGNORECASE,
    )

    n = len(spans)
    # Check header lines in first 30% of document
    max_header_line = min(n, max(5, int(n * 0.35)))
    for i in range(max_header_line):
        line = spans[i].text.strip()
        if header_regex.search(line):
            bp_indices.add(i)

    # Check footer / signature in last 25% of document
    min_footer_line = max(0, int(n * 0.75))
    for i in range(min_footer_line, n):
        line = spans[i].text.strip()
        if footer_regex.search(line) or sig_regex.search(line) or "[DOCTOR]" in line:
            bp_indices.add(i)

    # Group contiguous indices into spans
    if not bp_indices:
        return DocBoilerplateAnnotation(doc_id=doc_id, boilerplate_spans=[])

    sorted_indices = sorted(bp_indices)
    grouped_spans: list[BoilerplateSpan] = []
    start = sorted_indices[0]
    prev = sorted_indices[0]

    for idx in sorted_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            cat = "hospital_header" if start < max_header_line else "signature_block"
            grouped_spans.append(
                BoilerplateSpan(
                    start_line=start,
                    end_line=prev,
                    category=cat,
                    reason="Heuristic match",
                )
            )
            start = idx
            prev = idx
    cat = "hospital_header" if start < max_header_line else "signature_block"
    grouped_spans.append(
        BoilerplateSpan(
            start_line=start, end_line=prev, category=cat, reason="Heuristic match"
        )
    )

    return DocBoilerplateAnnotation(doc_id=doc_id, boilerplate_spans=grouped_spans)


def validate_against_clinical_anchors(
    annotation: DocBoilerplateAnnotation,
    spans: list[DocumentSpan],
    clinical_anchors: dict[str, str],
) -> tuple[DocBoilerplateAnnotation, list[str]]:
    """Validates that boilerplate annotations never suppress confirmed clinical anchor text.

    If an overlap is detected, unmarks the line to preserve clinical recall (Cardinal rule).
    """
    if not clinical_anchors:
        return annotation, []

    bp_indices = annotation.get_boilerplate_line_indices(len(spans))
    rescued_lines: list[str] = []
    valid_bp_indices: set[int] = set(bp_indices)

    for idx in bp_indices:
        line_text = spans[idx].text.strip()
        if len(line_text) < 10:
            continue
        # Check against each clinical anchor
        for anchor_name, anchor_content in clinical_anchors.items():
            if not isinstance(anchor_content, str):
                continue
            if line_text in anchor_content:
                # Overlap with confirmed clinical section! Rescue the line
                valid_bp_indices.discard(idx)
                rescued_lines.append(
                    f"Line [{idx}] rescued from {anchor_name}: '{line_text[:50]}...'"
                )
                break

    if len(valid_bp_indices) == len(bp_indices):
        return annotation, []

    # Reconstruct spans from filtered indices
    if not valid_bp_indices:
        return DocBoilerplateAnnotation(
            doc_id=annotation.doc_id, boilerplate_spans=[]
        ), rescued_lines

    sorted_indices = sorted(valid_bp_indices)
    new_spans: list[BoilerplateSpan] = []
    start = sorted_indices[0]
    prev = sorted_indices[0]
    for idx in sorted_indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            new_spans.append(
                BoilerplateSpan(
                    start_line=start,
                    end_line=prev,
                    category="filtered_boilerplate",
                    reason="Validated",
                )
            )
            start = idx
            prev = idx
    new_spans.append(
        BoilerplateSpan(
            start_line=start,
            end_line=prev,
            category="filtered_boilerplate",
            reason="Validated",
        )
    )

    return DocBoilerplateAnnotation(
        doc_id=annotation.doc_id, boilerplate_spans=new_spans
    ), rescued_lines


def annotate_corpus_file(
    input_jsonl_path: str,
    output_jsonl_path: str,
    max_docs: int | None = None,
    client: Any | None = None,
    model: str = DEFAULT_TEACHER_MODEL,
    use_mock: bool = False,
    resume: bool = True,
) -> int:
    """Processes a raw corpus JSONL file, producing annotated JSONL lines with resumption support."""
    out_path = Path(output_jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids: set[str] = set()
    if resume and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        completed_ids.add(record["doc_id"])
                    except json.JSONDecodeError as err:
                        logger.debug("Skipping unparseable line during resume: %s", err)
        logger.info(
            "Resuming from existing annotations: %d docs already processed.",
            len(completed_ids),
        )

    processed_count = 0
    with (
        open(input_jsonl_path, "r", encoding="utf-8") as in_f,
        open(out_path, "a", encoding="utf-8") as out_f,
    ):
        for line in in_f:
            if max_docs is not None and processed_count >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc["doc_id"]
            if doc_id in completed_ids:
                continue

            rectxt = doc["rectxt"]
            spans = extract_lines_with_offsets(rectxt)

            if use_mock:
                annotation = mock_heuristic_annotate(doc_id, rectxt)
            else:
                annotation = annotate_document_with_openai(
                    doc_id, rectxt, client=client, model=model
                )

            # Enforce clinical anchor safety
            clinical_anchors = doc.get("clinical_anchors", {})
            if isinstance(clinical_anchors, dict) and clinical_anchors:
                annotation, rescued = validate_against_clinical_anchors(
                    annotation, spans, clinical_anchors
                )
                if rescued:
                    logger.warning(
                        "Doc %s: Rescued %d lines from clinical anchors",
                        doc_id,
                        len(rescued),
                    )

            annotated_spans = annotation.apply_to_spans(spans)

            # Export record
            out_record = {
                "doc_id": doc_id,
                "rectype": doc.get("rectype"),
                "recdate": doc.get("recdate"),
                "sejum": doc.get("sejum"),
                "char_length": len(rectxt),
                "line_count": len(spans),
                "boilerplate_spans": [
                    s.model_dump() for s in annotation.boilerplate_spans
                ],
                "lines": [
                    {
                        "line_index": i,
                        "start_char": s.start_char,
                        "end_char": s.end_char,
                        "text": s.text,
                        "is_boilerplate": s.is_boilerplate,
                    }
                    for i, s in enumerate(annotated_spans)
                ],
            }
            out_f.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            out_f.flush()
            completed_ids.add(doc_id)
            processed_count += 1

    return processed_count
