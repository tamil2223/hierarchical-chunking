"""
Pipeline from PDF typography to outline-aware chunks.

This module is the implementation layer: stream text rows from pdfplumber, infer
which rows are titles, walk the document once to attach body copy and parent/child
links, then shape records for vector stores.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Optional

import pdfplumber


@dataclass(frozen=True)
class PipelineThresholds:
    """
    Defaults match many single-column academic PDFs. Replace `THRESHOLDS` if your
    corpus is consistently different (e.g. tight two-column, or 9/10 pt body).

    Suggested ranges (not hard limits):
    - word_cluster_pt: 2.5–4.0 — pdfplumber word merge & our line clustering;
      lower splits lines more aggressively; higher can glue adjacent lines.
    - paragraph_gap_ratio: 0.65–0.90 — vertical gap vs previous line height to
      insert a paragraph break; higher → fewer false breaks, lower → more breaks.
    - heading_font_above_body_pt: 0.35–0.75 — pt above modal body to treat as
      headline size; lower catches subtle headings, higher avoids noisy +0.5 pt.
    - max_title_candidate_chars: 120–200 — lines longer than this skip title
      heuristics (caps/bold/font); stops body paragraphs from becoming headings.
    - orphan_heading_word_count: 5–12 — words used as synthetic title for
      leading para_* chunks before the first real section.
    - parent_snippet_max_chars: 256–800 — parent_context prefix for RAG; raise if
      your embedder context is large and you want more hierarchy signal.
    - max_outline_depth: 2 by default — caps outline tiers for parent/child links and
      stored `level` (1 = top heading, 2 = under that). Deeper headings (e.g. 3.2.1)
      attach to the current top section instead of nesting further. Set to 3+ to
      allow deeper trees. `para_*` chunks stay level 0.
    """

    word_cluster_pt: float = 3.5
    paragraph_gap_ratio: float = 0.7
    heading_font_above_body_pt: float = 0.4
    max_title_candidate_chars: int = 120
    orphan_heading_word_count: int = 5
    parent_snippet_max_chars: int = 256
    max_outline_depth: int = 2


# Override in your app or notebook, e.g. `THRESHOLDS = PipelineThresholds(...)`.
THRESHOLDS: PipelineThresholds = PipelineThresholds()


# --- wire format (JSON keys are fixed for downstream consumers) -----------------


@dataclass
class TextRow:
    """One horizontal band of words on a page, plus lightweight font cues."""

    page_number: int
    text: str
    font_size: float = 0.0
    is_bold: bool = False


@dataclass
class Chunk:
    chunk_id: str
    section_number: str
    heading: str
    content: str
    level: int
    page: int
    parent_id: Optional[str] = None
    children_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "section_number": self.section_number,
            "heading": self.heading,
            "content": self.content,
            "level": self.level,
            "page": self.page,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
        }


# --- outline numbers: regex + nesting depth ------------------------------------

_OUTLINE_NUMBER_RULES: list[tuple[re.Pattern, Callable[[str], int]]] = [
    (
        re.compile(r"^(\d+(?:\.\d+){2,})\s{1,5}(\S.{1,120})$"),
        lambda token: token.count(".") + 1,
    ),
    (
        re.compile(r"^(\d+\.\d+)\s{1,5}(\S.{1,120})$"),
        lambda _t: 2,
    ),
    (
        re.compile(r"^(\d{1,2})\s{1,5}(\S.{1,120})$"),
        lambda _t: 1,
    ),
    (
        re.compile(r"^([A-Z]\.[a-z\d]+)\s{1,5}(\S.{1,120})$"),
        lambda _t: 2,
    ),
    (
        re.compile(r"^([A-Z])\s{1,5}([A-Z]\S.{1,120})$"),
        lambda _t: 1,
    ),
]


def parse_outline_number(line: str) -> Optional[tuple[str, str, int]]:
    """Return (section token, human title, nesting depth) when the row opens a section."""
    line = line.strip()
    for pattern, depth_rule in _OUTLINE_NUMBER_RULES:
        hit = pattern.match(line)
        if hit:
            token = hit.group(1)
            title = hit.group(2).strip()
            return token, title, depth_rule(token)
    return None


def chunk_primary_key(section_token: str) -> str:
    if section_token.startswith("H"):
        return f"heading_{section_token[1:]}"
    return "sec_" + section_token.replace(".", "_")


def _row_from_word_run(page_number: int, words: list) -> TextRow:
    body = " ".join(w["text"] for w in words)
    pts = [w.get("size", 0) for w in words if w.get("size", 0) > 0]
    avg_pt = sum(pts) / len(pts) if pts else 0.0
    bold = any("bold" in (w.get("fontname") or "").lower() for w in words)
    return TextRow(
        page_number=page_number,
        text=body,
        font_size=avg_pt,
        is_bold=bold,
    )


def stream_rows_from_pdf(path: str) -> list[TextRow]:
    """
    Flatten the PDF to a sequence of TextRow. Inserts an empty TextRow when the
    vertical whitespace between baselines looks like a paragraph boundary.
    """
    rows: list[TextRow] = []

    with pdfplumber.open(path) as doc:
        for page_number, page in enumerate(doc.pages, start=1):
            words = page.extract_words(
                x_tolerance=THRESHOLDS.word_cluster_pt,
                y_tolerance=THRESHOLDS.word_cluster_pt,
                extra_attrs=["fontname", "size"],
            )
            if not words:
                continue

            runs: list[tuple[float, list]] = []
            anchor_y = words[0]["top"]
            bucket = [words[0]]
            for w in words[1:]:
                if abs(w["top"] - anchor_y) <= THRESHOLDS.word_cluster_pt:
                    bucket.append(w)
                else:
                    runs.append((anchor_y, bucket))
                    bucket = [w]
                    anchor_y = w["top"]
            runs.append((anchor_y, bucket))

            for idx, (y0, run) in enumerate(runs):
                if idx > 0:
                    prior_y, prior_run = runs[idx - 1]
                    prior_bottom = max(w.get("bottom", prior_y + 12) for w in prior_run)
                    line_height = max(prior_bottom - prior_y, 1)
                    if y0 - prior_bottom > line_height * THRESHOLDS.paragraph_gap_ratio:
                        rows.append(TextRow(page_number=page_number, text=""))

                rows.append(_row_from_word_run(page_number, run))

    return rows


def font_size_ranks(rows: list[TextRow]) -> dict[float, int]:
    """
    Treat the most common rounded size as body copy; anything clearly taller
    becomes outline level 1, 2, … (1 = largest heading on the page).
    """
    samples = [
        round(r.font_size, 1) for r in rows if r.font_size > 0 and r.text.strip()
    ]
    if not samples:
        return {}

    body_pt = Counter(samples).most_common(1)[0][0]
    headline_pts = sorted(
        {s for s in samples if s > body_pt + THRESHOLDS.heading_font_above_body_pt},
        reverse=True,
    )
    return {pt: idx + 1 for idx, pt in enumerate(headline_pts)}


def title_signal_for_row(
    row: TextRow,
    size_rank: dict[float, int],
    mint_synthetic_token: Callable[[], str],
) -> Optional[tuple[str, str, int]]:
    """
    Decide whether this row starts a new chunk as a title. Order is fixed:
    outline number → oversized type → shouty caps → bold stub.
    """
    body = row.text.strip()
    if not body or len(body) > THRESHOLDS.max_title_candidate_chars:
        return None

    numbered = parse_outline_number(body)
    if numbered:
        return numbered

    if size_rank:
        key = round(row.font_size, 1)
        if key in size_rank:
            return mint_synthetic_token(), body, size_rank[key]

    if body.isupper() and " " in body and 4 <= len(body) <= 100:
        return mint_synthetic_token(), body, 1

    if row.is_bold and len(body) <= 80:
        return mint_synthetic_token(), body, 2

    return None


class OutlineDraft:
    """Single-pass mutable workspace: nodes, spill buffers, ancestry stack."""

    def __init__(self) -> None:
        self.nodes: dict[str, Chunk] = {}
        self.spills: dict[str, list[str]] = {}
        self.ancestry: list[tuple[int, str]] = []
        self.cursor: Optional[str] = None
        self._floating_n = 0
        self._synthetic_n = 0

    def mint_synthetic_token(self) -> str:
        self._synthetic_n += 1
        return f"H{self._synthetic_n}"

    def pop_ancestry_until(self, depth: int) -> Optional[str]:
        while self.ancestry and self.ancestry[-1][0] >= depth:
            self.ancestry.pop()
        return self.ancestry[-1][1] if self.ancestry else None

    def open_floating_block(self, page_number: int) -> str:
        self._floating_n += 1
        cid = f"para_{self._floating_n}"
        self.nodes[cid] = Chunk(
            chunk_id=cid,
            section_number=f"P{self._floating_n}",
            heading="",
            content="",
            level=0,
            page=page_number,
        )
        self.spills[cid] = []
        return cid

    def commit_title(
        self,
        section_token: str,
        title: str,
        depth: int,
        page_number: int,
    ) -> str:
        cap = max(1, THRESHOLDS.max_outline_depth)
        depth = max(1, min(int(depth), cap))
        cid = chunk_primary_key(section_token)
        parent = self.pop_ancestry_until(depth)
        self.nodes[cid] = Chunk(
            chunk_id=cid,
            section_number=section_token,
            heading=title,
            content="",
            level=depth,
            page=page_number,
            parent_id=parent,
        )
        self.spills[cid] = []
        if parent and parent in self.nodes:
            self.nodes[parent].children_ids.append(cid)
        self.ancestry.append((depth, cid))
        self.cursor = cid
        return cid

    def touch_body_cursor(self, page_number: int) -> None:
        if self.cursor is None:
            self.cursor = self.open_floating_block(page_number)

    def push_body_tokens(self, fragment: str) -> None:
        assert self.cursor is not None
        buf = self.spills[self.cursor]
        if buf and buf[-1].endswith("- "):
            buf[-1] = buf[-1][:-2] + fragment + " "
        else:
            buf.append(fragment + " ")

    def blank_row_signal(self) -> None:
        if self.cursor is None:
            return
        if self.cursor.startswith("para_"):
            if "".join(self.spills[self.cursor]).strip():
                self.cursor = None
            return
        buf = self.spills[self.cursor]
        if not buf:
            return
        glue = "".join(buf)
        if glue and not glue.endswith("\n\n"):
            self.spills[self.cursor] = [glue.rstrip(" ") + "\n\n"]

    def finalize_copy(self) -> None:
        for node in self.nodes.values():
            node.content = "".join(self.spills.get(node.chunk_id, [])).strip()

    def decorate_floating_titles(self) -> None:
        for node in self.nodes.values():
            if not node.chunk_id.startswith("para_") or not node.content:
                continue
            bits = node.content.split()
            n = THRESHOLDS.orphan_heading_word_count
            node.heading = " ".join(bits[:n]) + ("..." if len(bits) > n else "")

    def discard_empty_floaters(self) -> list[Chunk]:
        return [
            c
            for cid, c in self.nodes.items()
            if not cid.startswith("para_") or c.content
        ]


def materialize_chunks(path: str) -> list[Chunk]:
    rows = stream_rows_from_pdf(path)
    ranks = font_size_ranks(rows)
    draft = OutlineDraft()

    for row in rows:
        signal = title_signal_for_row(row, ranks, draft.mint_synthetic_token)
        if signal:
            token, title, depth = signal
            draft.commit_title(token, title, depth, row.page_number)
            continue

        fragment = row.text.strip()
        if fragment:
            draft.touch_body_cursor(row.page_number)
            draft.push_body_tokens(fragment)
        else:
            draft.blank_row_signal()

    draft.finalize_copy()
    draft.decorate_floating_titles()
    return draft.discard_empty_floaters()


def rag_record_for(node: Chunk, vault: dict[str, Chunk]) -> dict:
    blurb = ""
    if node.parent_id and node.parent_id in vault:
        upstream = vault[node.parent_id]
        lim = THRESHOLDS.parent_snippet_max_chars
        tail = upstream.content[:lim]
        if len(upstream.content) > lim:
            tail += "..."
        blurb = f"[{upstream.section_number}: {upstream.heading}] {tail}"

    return {
        **node.to_dict(),
        "parent_context": blurb,
        "retrieval_text": f"{node.section_number} {node.heading}\n{node.content}",
    }


def _clip(s: str, n: int) -> str:
    return s[:n] + ("..." if len(s) > n else "")


def emit_branch_report(vault: dict[str, Chunk]) -> None:
    roots = [n for n in vault.values() if n.children_ids]
    print(f"\n{'═' * 60}")
    print(f"  Branch view  ({len(roots)} parents)")
    print(f"{'═' * 60}")
    for parent in roots:
        print(f"\n┌─ [{parent.section_number}]  {parent.heading}")
        print(
            f"│  depth {parent.level}  ·  p.{parent.page}  ·  {len(parent.children_ids)} children"
        )
        if parent.content:
            print(f"│  {_clip(parent.content, 200)}")
        print("│")
        for i, child_id in enumerate(parent.children_ids):
            kid = vault.get(child_id)
            if not kid:
                continue
            last = i == len(parent.children_ids) - 1
            arm = "└──" if last else "├──"
            gap = "   " if last else "│  "
            print(f"│  {arm} [{kid.section_number}]  {kid.heading}  (p.{kid.page})")
            if kid.content:
                print(f"│  {gap}   {_clip(kid.content, 150)}")
        print(f"└{'─' * 58}")


def emit_floater_report(vault: dict[str, Chunk]) -> None:
    floaters = [n for n in vault.values() if n.chunk_id.startswith("para_")]
    if not floaters:
        return
    print(f"\n{'═' * 60}")
    print(f"  Preface / floating blocks  ({len(floaters)})")
    print(f"{'═' * 60}")
    for block in floaters:
        print(f"\n  [{block.chunk_id}]  p.{block.page}")
        print(f"  {_clip(block.content, 300)}")
        print(f"  {'─' * 56}")


def emit_outline_stats(vault: dict[str, Chunk]) -> None:
    roots = [n for n in vault.values() if n.children_ids]
    print(f"\n{'═' * 60}")
    print("  Outline stats")
    print(f"{'═' * 60}")
    print(f"  Parents: {len(roots)}\n")
    for i, parent in enumerate(roots, 1):
        k = len(parent.children_ids)
        print(f"  {i}. [{parent.section_number}] {parent.heading}")
        print(f"     → {k} child{'ren' if k != 1 else ''}")
    print(f"{'═' * 60}")


# Stable names for older call sites ------------------------------------------------

PdfLine = TextRow
assemble_chunks = materialize_chunks
build_rag_payload = rag_record_for
print_parent_children = emit_branch_report
print_standalone_paragraphs = emit_floater_report
print_summary = emit_outline_stats

__all__ = [
    "Chunk",
    "PdfLine",
    "PipelineThresholds",
    "TextRow",
    "THRESHOLDS",
    "assemble_chunks",
    "build_rag_payload",
    "materialize_chunks",
    "rag_record_for",
    "stream_rows_from_pdf",
    "emit_branch_report",
    "emit_floater_report",
    "emit_outline_stats",
    "print_parent_children",
    "print_standalone_paragraphs",
    "print_summary",
]
