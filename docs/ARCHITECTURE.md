# Architecture

This project keeps source evidence, generated descriptions, and evaluation labels separate.
That separation prevents accidental benchmark leakage and makes failures easier to diagnose.

## Data flow

```text
T2-RAGBench metadata and PDFs
        |
        v
Canonical registry
  - physical PDF identity: SHA-256
  - aliases: dataset, split, path, and distractor status
  - contexts: source text and structured tables
        |
        v
Source-only chunks ------------------------------+
        |                                         |
        +--> SQLite FTS5 BM25                     |
        |                                         +--> Hybrid retrieval
        +--> Ollama embeddings --> FAISS          |        |
                                                  |        v
Optional descriptions --> separate columns ------+   DeepInfra reranker
                                                           |
                                                           v
                                                Top source evidence
                                                           |
                                                           v
                                             DeepInfra answer model
                                                           |
                                                           v
                                           Answer, calculation, citations
```

## Canonical identity

A file path is not a document identity. The same PDF can appear under several paths.

- `documents.document_id` is the lowercase SHA-256 digest of the PDF bytes.
- `document_aliases` stores every dataset path, split, context link, and distractor flag.
- `contexts` stores the supplied benchmark context separately from the physical PDF.
- `chunks` stores source-derived evidence and an optional generated description.

This design handles all ConvFinQA duplicates and TAT-DQA cross-split duplicates.

## Evidence rules

The answer model receives `source_text`. It does not receive a generated description as evidence.
Descriptions can improve retrieval only when `retrieval.include_descriptions` is enabled.

Every evidence item is labeled as untrusted document data. A passive scanner
adds warning categories for instruction overrides, prompt requests, secret
requests, and command requests. It does not remove legitimate source text.

The following fields must never enter an index:

- `question`
- `program_answer`
- `original_answer`

Ingestion validates keys and text provenance before inserting chunks.

## Model roles

| Role | Model | Location |
|---|---|---|
| Parsed text or table description | `qwen3:8b` | Local Ollama |
| Difficult page or chart description | `qwen3-vl:8b-instruct` | Local Ollama |
| Dense embeddings | `qwen3-embedding:4b` | Local Ollama |
| Candidate reranking | `Qwen/Qwen3-Reranker-4B` | DeepInfra |
| Final answer | `Qwen/Qwen3-235B-A22B-Instruct-2507` | DeepInfra |

Run one local-model stage at a time. The short Ollama keep-alive avoids one
model reload per batch. The runbook includes manual unload commands.

## Retrieval sequence

1. Dataset and split filters restrict candidates before ranking.
2. Browser chat can add one conservative local spelling variant.
3. BM25 searches the original text and any accepted spelling variant.
4. FAISS retrieves semantically related bounded child chunks from the original query.
5. Reciprocal-rank fusion combines ranks without mixing incompatible raw scores.
6. The reranker scores the fused candidates once with the original query.
7. Retrieval evaluation keeps ten parent contexts for true @10 metrics.
8. Answer generation sends only the first eight parent contexts to the answer model.

## Answer generation and arithmetic

Answer prompt version 3 tells the model to identify the exact row, column, year,
company, unit, and scale before choosing values. It defines totals, changes,
percentages, ratios, simple hypothetical increases, and per-unit calculations.
It requires a separate `result_unit` field and explains scale cancellation.

Python evaluates every supplied arithmetic expression. When the displayed scalar
does not agree with that trusted calculator result, the answerer permits one
correction request. Python also checks explicit `per X` wording and scale agreement
between `answer` and `result_unit`. The same bounded correction handles invalid
JSON. It never performs more correction requests than
`generation.correction_attempts`. Every attempt contributes to the saved token
and cost totals. Earlier raw responses remain in `correction_history` for audits.

## Answer evaluation layers

The official score compares `normalized_answer` with `program_answer` using the
configured strict numeric tolerances. It remains unchanged for benchmark
comparability.

The secondary semantic score also considers `original_answer`, explicit
percentage units, ordinary financial rounding, and the locally verified
calculator result. It reports label conflicts instead of silently replacing an
official label.

The optional `qwen3:8b` judge reviews saved disputed development answers only.
It receives the question, gold context, both labels, generated answer, and
calculator result. Its output is diagnostic and never changes either
deterministic score.

The judge is not an accuracy authority. A deterministic `rescore-answers` command
recomputes official and semantic development metrics from saved JSONL records
without retrieval, answer generation, Ollama, or DeepInfra calls.

Complete parent contexts remain available to BM25 and answer expansion. They
are not embedded because some exceed an embedding model's context window.

An optional deterministic router sends identifier-like queries through the
lexical path. It keeps hybrid retrieval for semantic, table, calculation,
visual, and multi-source questions. Evaluation must prove whether routing helps.

## Evaluation boundaries

- Train questions support development and debugging.
- Dev questions select chunking, retrieval, reranking, and prompts.
- Test questions remain sealed until a frozen manifest exists.
- Test labels require a separate explicit unseal option.
- The scorer compares retrieved-context results with oracle-context results.

The freeze manifest hashes configuration, prompts, indexes, source files, exact
local Ollama digests, and configured cloud model IDs.

## Manual chat boundary

The Gradio page calls the same retrieval and answer classes as the `ask`
command. It does not create a second RAG pipeline. Each message uses only the
current question for retrieval and answer generation. Earlier messages remain
visible in the browser but do not become hidden query text.

The chat shows a polished answer by default. Calculations, source paths, pages,
and bounded evidence previews appear in a separate closed Gradio accordion.
Retrieval ranks, cost, latency, and raw diagnostics stay out of the user-facing
answer. Gradio handles one request at a time. The default server address is
`127.0.0.1:7860`, and public sharing is disabled unless the user supplies
`--share` explicitly.

The strict scalar answer remains the benchmark result. In the UI, a
comparison-style question uses a specialized extraction request instead of the
scalar answer request. That request
assigns source values to code-generated evidence IDs. Python verifies each
value against its exact source chunk, computes the direction, absolute change,
average when requested, relative percentage, or percentage-point change, and
renders the table. The model cannot write the final comparison numbers or
reverse the conclusion. Simple answers continue through the scalar answerer.

Clearly unrelated creative requests and requests for prompts or secrets stop
at a narrow local gate. They do not run retrieval or make a paid model request.
An invalid comparison plan falls back to the strict scalar answerer.

The page defaults to `FinQA` and `dev`. Dataset and split filters run before
BM25 ranking, dense candidate selection, fusion, and reranking. `All datasets`
or `All splits` is for general chat, not benchmark measurement. Command-line
evaluation applies each question's dataset and split automatically.

Browser typo expansion uses RapidFuzz and the source-only FTS vocabulary. It
edits only clear unknown words. It protects numbers, years, amounts, short
tickers, and identifier fragments. The original and corrected BM25 lists fuse
before dense fusion, so the paid reranker still runs once. Dense retrieval,
query routing, reranking, and answer generation keep the original question.
Command-line evaluation does not enable typo expansion, so prior development
metrics remain comparable.
FTS document frequency prevents a one-document OCR token from blocking a much
more common one-edit spelling. The original BM25 query always remains active,
so a rare legitimate name is not discarded.

## Raw-PDF comparison mode

`extract-raw-pdfs` writes raw page text and detected tables to separate registry tables.
TAT ordered JSON blocks take precedence over plain PDF text. PDFplumber extracts tables.
PDFium renders weak pages for visual-model fallback. The baseline contexts remain unchanged.
