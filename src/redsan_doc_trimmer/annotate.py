"""LLM-based teacher annotation module for generating weak supervision labels."""

from __future__ import annotations

import json
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from redsan_doc_trimmer.dataset import DocumentSpan, extract_lines_with_offsets


TEACHER_SYSTEM_PROMPT = """Tu es un expert en traitement de documents médicaux hospitaliers français (dossiers patients EDSAN / RECTXT).
Ton rôle est d'identifier les lignes qui relèvent STRICTEMENT du cadre administratif et de la mise en page (boilerplate / en-têtes / pieds de page), SANS JAMAIS supprimer le contenu clinique.

ÉLÉMENTS À MARQUER COMME BOILERPLATE (is_boilerplate = true) :
- En-têtes d'hôpitaux, logos textuels, adresses postales, numéros de téléphone/fax/email de l'établissement
- Titres de formulaires vides, mentions de transmission, codes-barres textuels, identifiants techniques de document
- Lignes de pied de page ("Page 1/3", avertissements légaux de confidentialité)
- Blocs de signature isolés (ex: "Secrétariat médical", nom du praticien seul en bas de page sans phrase clinique)

RÈGLE D'OR MÉDICALE :
- Tout ce qui touche à l'histoire de la maladie, aux symptômes, diagnostics, examens physiques, constantes (poids, taille, IMC), résultats de biologie, ordonnances, comptes-rendus opératoires ou conclusions DOIT ÊTRE MARQUÉ comme contenu clinique (is_boilerplate = false).
- En cas de doute ou si une ligne mélange administratif et texte clinique, NE PAS MARQUER comme boilerplate.
"""


class LineAnnotation(BaseModel):
    line_index: int = Field(..., description="Index de la ligne (0-indexé)")
    is_boilerplate: bool = Field(..., description="Vrai si la ligne est purement administrative/boilerplate")
    reason: str = Field(..., description="Brève justification (ex: 'en-tête CHU', 'pied de page', 'texte clinique')")


class DocumentAnnotationResult(BaseModel):
    doc_id: str
    lines: List[LineAnnotation]


def format_doc_for_llm(spans: List[DocumentSpan]) -> str:
    """Formats a list of lines with line indices for LLM prompting."""
    formatted_lines = []
    for i, span in enumerate(spans):
        text = span.text.strip()
        if not text:
            continue
        formatted_lines.append(f"[{i}] {text}")
    return "\n".join(formatted_lines)
