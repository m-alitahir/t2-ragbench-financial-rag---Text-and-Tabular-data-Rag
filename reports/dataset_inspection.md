# Dataset inspection report

This report records the design evidence gathered before implementation.

## Inventory

- Dataset commit: `adf7fe1`
- Dataset content: 2.78 GiB
- Total files: 13,187
- PDF files: 7,353
- Internal PDF pages: 7,662
- PNG files: 3,067
- JSON files: 2,758
- JSONL files: 7
- Invalid PDFs: 0
- Unresolved Git Xet or Git LFS pointers: 0

The PDF set contains 7,047 one-page files, 303 two-page files, and three
three-page files.

## Observed formats

FinQA and ConvFinQA use annual-report pages. Pages contain narrative text,
financial tables, or mixed text-and-table layouts. ConvFinQA reuses 1,806
FinQA source pages byte for byte.

TAT-DQA contains dense tables, footnotes, and some table continuations across
pages. Its JSON files provide ordered blocks, text, word coordinates, and page
relationships. Its PNG files are page images, not additional PDF documents.

The seven JSONL files contain question records and supplied source contexts.
Each row also contains labels. The application reads only an allow-list of
source fields during ingestion.

## Cross-file relationships

- A metadata row links to a PDF through `file_name`.
- Several questions can share one `context_id`.
- TAT-DQA layout JSON and page PNG names share the source PDF stem.
- All ConvFinQA PDFs match FinQA PDF content.
- TAT-DQA has 107 exact-PDF groups that cross dataset splits.
- TAT-DQA includes 35 unlabeled train PDFs as retrieval distractors.

The registry therefore uses a PDF SHA-256 digest as the physical document ID.
It stores dataset path, split, and distractor state as aliases.

## Parsing and chunking decision

The benchmark corpus uses the supplied `context`, `table`, `pre_text`, and
`post_text` fields first. This avoids mixing parser errors with retrieval
errors. It creates one whole-context baseline and smaller table-aware child
chunks. Large prose uses overlapping word windows. Large tables split only at
row boundaries and repeat their header.

The optional raw-PDF corpus remains separate. It prefers TAT-DQA ordered JSON
blocks. It uses PDFplumber for other text and table extraction. It renders only
weak pages, or all pages when the user explicitly requests that mode.

## Visual processing decision

The supplied TAT-DQA page PNGs are the first visual inputs. PDFium can render
pages that have weak extracted text. The visual model creates retrieval
descriptions only. A generated description never replaces source evidence.

## Main risks

- Plain PDF extraction can damage reading order and table structure.
- Multi-row headers, footnotes, units, dashes, and accounting negatives need
  careful preservation.
- Generated descriptions can omit values or invent relationships.
- Duplicate pages across datasets and splits can inflate retrieval scores.
- Question and answer fields share rows with source context fields.
- Document text can contain instruction-like content.

The design addresses these risks with source-field allow-lists, SHA-256
canonicalization, separate description storage, strict test gates, citations,
and passive prompt-injection warnings.
