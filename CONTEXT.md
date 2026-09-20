# Context: redsan-doc-trimmer Domain Model

## Core Concepts

### RECTXT
The raw, unstructured clinical document text extracted from the EDSAN clinical data warehouse. It contains arbitrary line wrapping, legacy transcription headers, mail-merge tags, physician signatures, and administrative boilerplate mixed with clinical narrative.

### The Cardinal Rule (Asymmetric Cost of Error)
The governing principle of the document trimmer:
- **False Negative (keeping boilerplate)**: Harmless — costs a few extra prompt tokens in downstream LLM calls.
- **False Positive (stripping clinical content)**: Fatal — destroys medical facts, diagnoses, measurements, or posologies needed for medical coding and clinical validation.
- **Decision Policy**: When in doubt, always keep the text. Class-weighted loss heavily favors clinical recall ($\ge 98\%$).

### Grounding Guarantee
Every trimmed prediction must map back to exact character intervals `[start_char, end_char]` in the raw `RECTXT` document, maintaining auditability and quote verification for downstream coding pipelines.

### Boilerplate Span
A contiguous span of lines in a document containing purely administrative or non-clinical text:
- `hospital_header`: Hospital letterhead, department banners, FINESS numbers, telephone/fax lines.
- `doctor_header`: Prescribing or consulting physician titles, RPPS/ADELI registration numbers.
- `patient_address`: Outpatient mailing addresses, postal codes, telephone contacts.
- `signature_block`: Doctor or resident signature lines, secretarial typist initials, validation stamps.
- `legal_notice`: Standard disclaimers ("ne rien inscrire sous ce trait", RGPD warnings, prescription validity).
- `form_metadata`: Word mail-merge artifacts (`«Libellé...»`, `MERGEFIELD`), check-boxes (`FORMCHECKBOX`).

### Clinical Narrative
Any text conveying patient history, symptoms, physical findings, vital signs, lab/imaging interpretations, surgical procedures, diagnoses, medications, posologies, or therapeutic plans.

### Active Learning Mining Margin
The uncertainty window where the student model's predicted probability of boilerplate is borderline ($p \in [0.35, 0.65]$). Documents containing lines in this margin represent hard negative candidates selected for teacher re-annotation and human curation.

