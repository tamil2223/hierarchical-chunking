# hierarchical-chunking

Tiny demo of **hierarchical chunking for academic PDFs**:

- **Layer 1**: split extracted text into **section spans** using numbered headings (e.g. `1 Introduction`, `3.2 Attention`, `3.2.1 ...`)
- **Layer 2**: within each section, split into **overlapping groups of sentences**

It’s intentionally heuristic (regex-based headings + sentence splitting), but works well for many papers after PDF text extraction.

## Requirements

- Python 3.9+
- PyMuPDF (for PDF text extraction)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pymupdf
```

## Usage (CLI)

Run on a local PDF:

```bash
python3 hierarchical_chunking.py --pdf path/to/paper.pdf
```

Run on a PDF URL (downloaded to `--cache` if missing):

```bash
python3 hierarchical_chunking.py \
  --url "https://arxiv.org/pdf/1706.03762.pdf" \
  --cache paper.pdf
```

If you hit `SSL: CERTIFICATE_VERIFY_FAILED` on macOS, install a CA bundle and retry:

```bash
pip install certifi
```

As a last resort (not recommended), you can bypass certificate verification:

```bash
python3 hierarchical_chunking.py --url "https://..." --cache paper.pdf --insecure
```

Tune chunking:

```bash
python3 hierarchical_chunking.py \
  --pdf path/to/paper.pdf \
  --group-size 10 \
  --overlap 2 \
  --max-top 12 \
  --preview 5
```

### Output format

The script prints a short summary and then previews the first `--preview` chunks:

- `[{start}:{end}] {label}`: character offsets into the extracted text
- a ~160 character snippet from that span

Labels look like:

`sec_02__chunk_003__3.2.1_Scaled_Dot-Product_Att`

## Usage (Python API)

The core functions are:

- `section_chunks(text, max_top=12)` → `[(start, end, label), ...]`
- `hierarchical_sentence_chunks(text, section_spans, group_size=10, overlap=2, label_max_len=30)` → `[(start, end, label), ...]`

Example:

```python
from hierarchical_chunking import section_chunks, hierarchical_sentence_chunks

text = "1 Introduction ... 2 Methods ... (etc)"
secs = section_chunks(text, max_top=12)
spans = hierarchical_sentence_chunks(text, secs, group_size=10, overlap=2)

for s, e, label in spans[:3]:
    print(label, repr(text[s:e][:80]))
```

## Notes / limitations

- **PDF extraction quality matters**: this uses PyMuPDF’s `page.get_text()` and then normalizes whitespace.
- **Sentence splitting is heuristic**: a simple regex (`.!?`) splitter; abbreviations and references may confuse it.
- **Heading detection is heuristic**: it looks for numbered headings and filters out common false-positives (Figure/Table/etc).
