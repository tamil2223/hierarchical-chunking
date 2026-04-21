from __future__ import annotations

import argparse
import os
import re
import sys
import ssl
import urllib.request
import urllib.error
from typing import List, Sequence, Tuple

Span = Tuple[int, int, str]
SectionSpan = Tuple[int, int, str]


_SENT_RE = re.compile(r"[^.!?]+[.!?]+")


def _clean_label(label: str, *, max_len: int) -> str:
    cleaned = (label or "section").strip().replace(" ", "_")
    return cleaned[:max_len] if max_len > 0 else cleaned


def section_chunks(text: str, *, max_top: int = 12) -> List[SectionSpan]:
    """Split a paper-like document into section spans using numbered headings.

    Finds headings like:
    - "1 Introduction"
    - "3.2 Attention"
    - "3.2.1 Scaled Dot-Product Attention"

    Returns a list of `(start, end, label)` spans over `text`.

    This is heuristic, but works well for many academic PDFs after text extraction.
    """
    if max_top < 1:
        raise ValueError("max_top must be >= 1")

    pattern = re.compile(
        r"(?<=\s)"  # avoid matching at position 0 mid-token
        r"(\d+(?:\.\d+){0,2})"  # 1, 2, 3.1, 3.2.1
        r"\s+([A-Z][a-z][a-zA-Z\-' ]{1,50})"  # Title-ish
    )
    blacklist = re.compile(
        r"^(Figure|Table|Equation|Section|Algorithm|Appendix|Input|Output|"
        r"Residual|Label|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Proceedings)\b"
    )

    def parse_num(s: str) -> Tuple[int, ...]:
        return tuple(int(x) for x in s.split("."))

    candidates: List[Tuple[int, Tuple[int, ...], str]] = []
    for m in pattern.finditer(text):
        num_s, title = m.group(1), m.group(2).strip()
        if blacklist.match(title):
            continue
        num = parse_num(num_s)
        if not (1 <= num[0] <= max_top):
            continue
        candidates.append((m.start(), num, f"{num_s} {title[:45]}"))

    kept: List[Tuple[int, str]] = []
    prev: Tuple[int, ...] = (0,)
    for pos, num, label in candidates:
        if num > prev:
            kept.append((pos, label))
            prev = num

    if not kept:
        return [(0, len(text), "full_doc")]

    spans: List[SectionSpan] = []
    if kept[0][0] > 0:
        spans.append((0, kept[0][0], "preamble"))
    for i, (pos, name) in enumerate(kept):
        end = kept[i + 1][0] if i + 1 < len(kept) else len(text)
        spans.append((pos, end, name))
    return spans


def _sentence_spans_global(text: str, start: int, end: int) -> List[Tuple[int, int]]:
    """Return sentence spans (global offsets) inside [start, end).

    Notes:
    - This is a lightweight regex-based sentence splitter (heuristic).
    - It also captures trailing text without terminal punctuation.
    """
    if start < 0 or end < 0 or start > end:
        raise ValueError(f"Invalid span bounds: start={start}, end={end}")
    if end > len(text):
        raise ValueError(f"Span end out of range: end={end}, len(text)={len(text)}")

    sub = text[start:end]
    spans: List[Tuple[int, int]] = []

    for m in _SENT_RE.finditer(sub):
        s = start + m.start()
        e = start + m.end()
        if s < e and text[s:e].strip():
            spans.append((s, e))

    # Handle trailing text without terminal punctuation.
    tail_s = spans[-1][1] if spans else start
    if tail_s < end and text[tail_s:end].strip():
        spans.append((tail_s, end))

    # Ensure monotonicity and non-empty.
    out: List[Tuple[int, int]] = []
    last = start
    for s, e in spans:
        s = max(s, last)
        if s < e and text[s:e].strip():
            out.append((s, e))
            last = e
    return out


def hierarchical_sentence_chunks(
    text: str,
    section_spans: Sequence[SectionSpan],
    *,
    group_size: int = 10,
    overlap: int = 2,
    label_max_len: int = 30,
) -> List[Span]:
    """Hierarchical chunking: section -> overlapping groups of sentences.

    Inputs:
    - text: full document text
    - section_spans: list of (start, end, section_label) over global `text`
    - group_size: number of sentences per chunk
    - overlap: overlap in sentence count (0 <= overlap < group_size)

    Output:
    - list of (start, end, label) spans over global `text`

    Labels look like:
    - sec_02__chunk_003__3.2.1_Scaled_Dot-Product_Att
    """
    if group_size <= 0:
        raise ValueError("group_size must be >= 1")
    if overlap < 0 or overlap >= group_size:
        raise ValueError("overlap must satisfy 0 <= overlap < group_size")

    spans: List[Span] = []
    step = group_size - overlap

    for sec_i, (sec_s, sec_e, sec_name) in enumerate(section_spans):
        sec_name_clean = _clean_label(str(sec_name), max_len=label_max_len)
        sent_spans = _sentence_spans_global(text, sec_s, sec_e)

        if not sent_spans:
            # Fallback: keep whole section if it's non-empty.
            if sec_s < sec_e and text[sec_s:sec_e].strip():
                spans.append(
                    (sec_s, sec_e, f"sec_{sec_i:02d}__chunk_{0:03d}__{sec_name_clean}")
                )
            continue

        chunk_i = 0
        for i0 in range(0, len(sent_spans), step):
            group = sent_spans[i0 : i0 + group_size]
            if not group:
                continue
            s = group[0][0]
            e = group[-1][1]
            if s < e and text[s:e].strip():
                spans.append(
                    (s, e, f"sec_{sec_i:02d}__chunk_{chunk_i:03d}__{sec_name_clean}")
                )
                chunk_i += 1

    spans = [(s, e, lab) for (s, e, lab) in spans if s < e]
    spans.sort(key=lambda x: (x[0], x[1]))
    return spans


def _load_text_from_pdf(pdf_path: str) -> str:
    try:
        import fitz  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "PyMuPDF is required to read PDFs. Install with: pip install pymupdf"
        ) from e

    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc]
    return re.sub(r"\s+", " ", "\n".join(pages)).strip()


def _download_pdf(url: str, dest_path: str, *, insecure: bool = False) -> None:
    req = urllib.request.Request(
        url,
        headers={
            # Some hosts behave better with a UA.
            "User-Agent": "hierarchical-chunking/0.1 (+https://github.com/)",
        },
    )

    if insecure:
        ctx = ssl._create_unverified_context()  # nosec - explicit user opt-in
    else:
        ctx = None
        # On some macOS Python installs, system certs aren't configured for urllib.
        # If certifi is installed, prefer it as a portable CA bundle.
        try:
            import certifi  # type: ignore

            ctx = ssl.create_default_context(cafile=certifi.where())
        except ModuleNotFoundError:
            ctx = ssl.create_default_context()

    try:
        with urllib.request.urlopen(req, context=ctx) as r, open(dest_path, "wb") as f:
            f.write(r.read())
    except ssl.SSLCertVerificationError as e:
        msg = (
            "SSL certificate verification failed while downloading the PDF.\n\n"
            "Common fixes on macOS:\n"
            "- If you installed Python from python.org, run the bundled certificate installer:\n"
            "  /Applications/Python 3.x/Install Certificates.command\n"
            "- Or install certifi and retry:\n"
            "  pip install certifi\n\n"
            "As a last resort (NOT recommended), you can bypass verification with --insecure.\n"
        )
        raise RuntimeError(msg) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            "Failed to download the PDF URL.\n\n"
            f"URL: {url}\n"
            f"Error: {getattr(e, 'reason', e)}\n\n"
            "If you're behind a corporate proxy/firewall, try downloading the PDF manually and "
            "running with --pdf instead."
        ) from e


def _main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        description="Hierarchical chunking demo (section -> sentence groups)."
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="Path to a local PDF file")
    src.add_argument("--url", help="URL to a PDF (downloaded to --cache)")
    p.add_argument("--cache", default="paper.pdf", help="Download path for --url")
    p.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification for --url downloads (unsafe).",
    )
    p.add_argument("--group-size", type=int, default=10, help="Sentences per chunk")
    p.add_argument("--overlap", type=int, default=2, help="Sentence overlap")
    p.add_argument("--max-top", type=int, default=12, help="Max top-level section number")
    p.add_argument("--preview", type=int, default=5, help="How many chunks to preview")
    args = p.parse_args(argv)

    if args.url:
        if not os.path.exists(args.cache):
            _download_pdf(args.url, args.cache, insecure=args.insecure)
        pdf_path = args.cache
    else:
        pdf_path = args.pdf

    if not pdf_path or not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path!r}")

    text = _load_text_from_pdf(pdf_path)
    secs = section_chunks(text, max_top=args.max_top)
    spans = hierarchical_sentence_chunks(
        text, secs, group_size=args.group_size, overlap=args.overlap
    )

    print(f"PDF: {pdf_path}")
    print(f"Chars: {len(text):,}")
    print(f"Sections: {len(secs)}")
    print(f"Chunks: {len(spans)} (group_size={args.group_size}, overlap={args.overlap})")
    print()

    for s, e, lab in spans[: max(0, args.preview)]:
        snippet = re.sub(r"\s+", " ", text[s:e].strip())[:160]
        print(f"[{s}:{e}] {lab}")
        print(f"  {snippet!r}")
        print()

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv[1:]))

