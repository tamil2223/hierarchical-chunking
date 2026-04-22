# hierarchical-chunking

Small **PDF → outline-linked chunks** pipeline for RAG: one pass over the document, parent/child links between sections, and a ready-made `retrieval_text` field per chunk.

Extraction uses **[pdfplumber](https://github.com/jsvine/pdfplumber)** (word-level geometry, font size, bold). Heading detection is heuristic, not perfect on every layout.

## What it does

1. **Stream** each page into visual lines (words grouped by baseline; large vertical gaps become paragraph separators).
2. **Guess titles** in a fixed order: numbered outline (`1`, `3.2`, `A.1`, …) → larger-than-body font → short ALL CAPS (with a space) → short bold line.
3. **Attach body text** to the current section (or to a leading `para_*` block before the first real heading).
4. **Emit JSON** with `parent_id` / `children_ids`, plus `parent_context` and `retrieval_text` for embedding.

**Outline depth** is capped at **two tiers** by default (`level` 1 = top heading, `level` 2 = subsection under it). Deeper PDF numbering (e.g. `3.2.1`) still gets its own chunk, but its parent becomes the **current top section** (e.g. `3`), not `3.2`—so the linked tree matches common two-level RAG practice. Raise **`PipelineThresholds.max_outline_depth`** (e.g. to `3`) if you want deeper parent chains.

Tuning knobs live in **`PipelineThresholds`** (`pdf_rag_pipeline.py`): word clustering, paragraph gap ratio, heading font delta, title line length cap, orphan preview words, parent snippet length, outline depth. Replace the module-level **`THRESHOLDS`** instance if your PDFs need different defaults.

## Requirements

- Python 3.9+

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## CLI

```bash
python3 hierarchical_chunking.py path/to/paper.pdf
```

Writes **`{YYYYMMDD_HHMMSS}_chunks.json`** next to the PDF (resolved path) and prints outline reports to stdout.

Explicit output path:

```bash
python3 hierarchical_chunking.py path/to/paper.pdf ./out/chunks.json
```

## Output (JSON)

Each element is one chunk: section headings and body spans share the same schema; leading material before the first numbered heading often appears as `para_*` chunks.

```json
{
  "chunk_id": "sec_3_2_1",
  "section_number": "3.2.1",
  "heading": "Scaled Dot-Product Attention",
  "content": "… body text …",
  "level": 3,
  "page": 5,
  "parent_id": "sec_3_2",
  "children_ids": [],
  "parent_context": "[3.2: Attention] First ~400 chars of parent body…",
  "retrieval_text": "3.2.1 Scaled Dot-Product Attention\n… body text …"
}
```

- **`chunk_id`**: stable id (`sec_*` for numbered sections, `heading_*` for synthetic `H1`-style labels, `para_*` for floating preface blocks).
- **`parent_context`**: short parent excerpt for retrieval; empty if there is no parent.
- **`retrieval_text`**: single string meant to be embedded (section label + heading + content).

## Python API

**Facade** (`hierarchical_chunking.py`):

```python
from hierarchical_chunking import (
    chunk_pdf_for_rag,
    default_export_path,
    materialize_chunks,
    rag_record_for,
)

records = chunk_pdf_for_rag("paper.pdf", export_json_path=None)  # no file if path is None
nodes = materialize_chunks("paper.pdf")
vault = {c.chunk_id: c for c in nodes}
row = rag_record_for(nodes[0], vault)
```

**Tuning** (optional):

```python
import pdf_rag_pipeline as prp

prp.THRESHOLDS = prp.PipelineThresholds(
    word_cluster_pt=2.5,
    paragraph_gap_ratio=0.85,
    heading_font_above_body_pt=0.5,
    max_outline_depth=3,
)
```

**Pipeline module** (`pdf_rag_pipeline.py`): `TextRow`, `Chunk`, `stream_rows_from_pdf`, `font_size_ranks`, `emit_branch_report`, etc. Older names are still available as aliases (`assemble_chunks`, `build_rag_payload`, `print_parent_children`, …).

## Project layout

| File                       | Role                                           |
| -------------------------- | ---------------------------------------------- |
| `hierarchical_chunking.py` | CLI, JSON export, re-exports                   |
| `pdf_rag_pipeline.py`      | Extraction, outline logic, reports, thresholds |

## Future scope

Possible directions that fit this codebase without throwing away the current pipeline:

**Layout and extraction**

- **Two-column and reading order**: column detection, left-then-right or block-ordered text so headings and body stay aligned.
- **Second backend** (e.g. PyMuPDF): optional fast path or block-based paragraphs; compare or merge with pdfplumber output.
- **Noise masks**: skip or tag headers/footers, page numbers, and repeated running titles using position or repetition heuristics.
- **Figures and tables**: detect captions (`Figure 3`, `Table 2`) and emit separate chunks or metadata pointers instead of mixing them into body text.
- **Scanned PDFs**: optional OCR path when extracted text is empty or too short.

**Chunking and RAG output**

- **Within-section splitting**: max token/character targets, sliding windows, or sentence/paragraph grouping _under_ each heading so long sections become multiple retrievable units.
- **Richer graph fields**: explicit `breadcrumb` (ancestor titles), `sibling_ids`, page ranges, or confidence scores for heading guesses.
- **Configurable `retrieval_text`**: templates (e.g. include only title path vs full parent chain) for different embedding models.

**CLI and configuration**

- **`argparse`**: PDF path, output path, `--quiet`, optional `--config` YAML/TOML mapping to `PipelineThresholds`.
- **Batch mode**: directory of PDFs → one JSONL or one folder of JSON files.
- **Optional download**: `--url` + cache path (with clear SSL/docs), if you want parity with ad-hoc `curl` workflows.

**Quality and maintenance**

- **Golden tests**: small fixture PDFs (or snapshots of `materialize_chunks` output) to lock heading and parent/child behavior when thresholds change.
- **Lint rules for chunks**: flag overlong sections, empty parents, or suspicious `para_*` volume for manual review.

## Limitations

- **Layout-dependent**: two-column papers, equations, and footnotes can confuse line grouping and title rules.
- **No URL download** in the current CLI (use `curl`/`wget` and pass a local path).
- **Heuristic headings**: false positives/negatives happen; adjust **`THRESHOLDS`** or post-filter chunks for production.
