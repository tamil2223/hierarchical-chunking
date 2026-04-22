#!/usr/bin/env python3
"""
CLI and thin API over the PDF → outline → RAG-record pipeline.

The heavy lifting lives in `pdf_rag_pipeline`: row streaming, title detection,
single-pass outline assembly, and console reports. This file only wires I/O.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from pdf_rag_pipeline import (
    Chunk,
    PdfLine,
    TextRow,
    assemble_chunks,
    build_rag_payload,
    emit_branch_report,
    emit_floater_report,
    emit_outline_stats,
    materialize_chunks,
    print_parent_children,
    print_standalone_paragraphs,
    print_summary,
    rag_record_for,
)

__all__ = [
    "Chunk",
    "PdfLine",
    "TextRow",
    "assemble_chunks",
    "build_rag_payload",
    "chunk_pdf_for_rag",
    "default_export_path",
    "emit_branch_report",
    "emit_floater_report",
    "emit_outline_stats",
    "hierarchical_chunk",
    "materialize_chunks",
    "print_parent_children",
    "print_standalone_paragraphs",
    "print_summary",
    "rag_record_for",
]


def default_export_path(pdf_file_path: str) -> str:
    """Resolve beside the input PDF: `{YYYYMMDD_HHMMSS}_chunks.json`."""
    path = Path(pdf_file_path).expanduser().resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(path.with_name(f"{stamp}_chunks.json"))


def chunk_pdf_for_rag(
    pdf_file_path: str,
    export_json_path: Optional[str] = None,
) -> list[dict]:
    """
    Run the pipeline: chunks from `pdf_file_path`, optional JSON export,
    stdout reports, return embedding-ready dicts (includes parent_context +
    retrieval_text).
    """
    nodes = materialize_chunks(pdf_file_path)
    vault = {c.chunk_id: c for c in nodes}
    records = [rag_record_for(c, vault) for c in nodes]

    if export_json_path:
        with open(export_json_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2)
        print(f"Wrote {len(records)} chunks to {export_json_path}")

    emit_branch_report(vault)
    emit_floater_report(vault)
    emit_outline_stats(vault)

    return records


hierarchical_chunk = chunk_pdf_for_rag


if __name__ == "__main__":
    if len(sys.argv) < 2:
        script = Path(sys.argv[0]).name or "hierarchical_chunking.py"
        print(f"Usage: python {script} <path-to.pdf> [export.json]")
        sys.exit(1)

    pdf_file_path = sys.argv[1]
    export_json_path = (
        sys.argv[2] if len(sys.argv) > 2 else default_export_path(pdf_file_path)
    )

    chunk_pdf_for_rag(pdf_file_path, export_json_path)
