# Manual runbook

No command in this document has been executed by Codex. Run each command yourself.

Open PowerShell in this project folder before you begin.

## 1. Create the Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The install step adds FAISS. Run it only when you are ready to install the vector dependency.

## 2. Add the DeepInfra key

Open `.env` in VS Code. Paste the key after `DEEPINFRA_API_KEY=`.

Do not add spaces or quotes unless the key itself contains them.

## 3. Confirm Ollama models

```powershell
ollama list
```

The default configuration expects these tags:

```text
qwen3:8b
qwen3-vl:8b-instruct
qwen3-embedding:4b
```

## 4. Inspect configuration

Review `configs/default.yaml`. Confirm the dataset path before running ingestion.

## 5. Build the canonical registry

```powershell
t2rag --config configs/default.yaml build-registry
```

This command hashes PDFs, stores aliases, imports source contexts, and creates safe chunks.
It never adds question or answer fields to the index tables.

## 6. Build and evaluate BM25 first

```powershell
t2rag --config configs/default.yaml build-bm25
t2rag --config configs/default.yaml evaluate-retrieval --split dev --mode bm25
```

Review BM25 results before enabling descriptions or dense retrieval.

## 7. Generate optional descriptions

Text and table descriptions:

```powershell
t2rag --config configs/default.yaml describe --mode text --limit 100
```

Visual fallback for selected difficult pages:

```powershell
t2rag --config configs/default.yaml describe --mode vision --limit 25
```

Descriptions remain separate from source evidence. The default index ignores them.

The two-minute keep-alive avoids reloading Ollama for every request. If memory
is low after a stage, unload its model before you start another local stage:

```powershell
ollama stop qwen3:8b
ollama stop qwen3-vl:8b-instruct
```

For FinQA visual fallback, first build the separate raw-PDF corpus and render weak pages:

```powershell
t2rag --config configs/default.yaml extract-raw-pdfs --render weak
t2rag --config configs/default.yaml describe --mode vision --limit 25
```

Raw extraction does not overwrite supplied benchmark contexts.

## 8. Build the dense index

```powershell
t2rag --config configs/default.yaml build-dense
```

This is the first command that asks Ollama to generate corpus embeddings.
It embeds bounded narrative and table child chunks. Complete parent contexts
remain in SQLite and are loaded only after retrieval.

After it finishes, you can free its memory with:

```powershell
ollama stop qwen3-embedding:4b
```

## 9. Evaluate retrieval variants

```powershell
t2rag --config configs/default.yaml evaluate-retrieval --split dev --mode dense
t2rag --config configs/default.yaml evaluate-retrieval --split dev --mode hybrid
t2rag --config configs/default.yaml evaluate-retrieval --split dev --mode reranked
t2rag --config configs/default.yaml evaluate-retrieval --split dev --mode routed
```

The reranked mode calls the paid DeepInfra reranker.
The routed mode uses transparent rules to prefer BM25 for identifier queries.

## 10. Test answer generation on a small sample

```powershell
t2rag --config configs/default.yaml evaluate-answers --split dev --mode retrieved --limit 20
```

This command calls the paid DeepInfra answer model. Start with a small limit.
Run `--mode oracle` separately when you are ready to measure the answer-model ceiling.

Answer prompt version 3 defines common financial operations, per-unit calculations,
and scale cancellation. It saves the result unit separately. Python checks the
displayed value against both the arithmetic result and the result unit. If JSON,
the value, or the unit is inconsistent, the answerer can make one correction
request. This extra paid request occurs only for the affected answer. The configured
limit is in the `generation` section of `configs/default.yaml`.

The summary contains two separate answer sections:

- `answer` is the official strict `program_answer` metric.
- `semantic_answer` is a deterministic diagnostic for displayed percentages,
  normal rounding, calculator precision, and conflicting labels.

Retrieval metrics use ten results. The answer model still receives only the
eight contexts configured by `retrieval.answer_contexts`.

## 11. Rescore saved development answers for free

Use this after an evaluator change. It reads saved model responses and makes
zero model, retrieval, embedding, reranker, or API requests.

```powershell
python -m t2rag --config configs/default.yaml rescore-answers --dataset finqa --split dev --records work/outputs/answers-finqa-dev.retrieved.jsonl --output work/outputs/rescore-finqa-dev-v1-retrieved.json
```

The output includes new official and semantic aggregates, per-question scores,
and deterministic label-conflict IDs.

## 12. Review disputed answers with local Qwen

The local judge is optional and diagnostic. It does not change official or
semantic deterministic scores. By default, it reviews only records where the
two deterministic checks disagree or the labels conflict.

```powershell
python -m t2rag --config configs/default.yaml judge-answers --dataset finqa --split dev --records work/outputs/answers-finqa-dev.retrieved.jsonl --limit 5
```

The report keeps two independent summaries. `judge_verdict_counts` contains only
the local Qwen verdicts. `deterministic_label_conflicts` contains label conflicts
found by the Python semantic checker, including question IDs.

The semantic checker uses a small absolute tolerance plus relative tolerance.
This accepts normal displayed rounding but rejects an answer that materially
disagrees with its verified calculator result. The judge parser also rejects
contradictory pairs such as `incorrect` with `equivalent: true`.

This command uses local `qwen3:8b` through Ollama. It does not use DeepInfra or
consume paid API credit. Add `--all` only when you deliberately want to review
every saved answer.

## 13. Ask one question

```powershell
t2rag --config configs/default.yaml ask "What was the reported operating income?"
```

The command prints the answer, calculation, citations, and retrieval trace.

## 14. Use the local browser chat

After a source update adds Gradio, refresh the editable installation once:

```powershell
python -m pip install -e ".[dev]"
```

Run the tests before starting the page:

```powershell
python -m pytest
```

Start the local chat server:

```powershell
python -m t2rag --config configs/default.yaml chat-ui
```

The browser should open `http://127.0.0.1:7860`. Press `Ctrl+C` in PowerShell
to stop the server. The server binds only to this computer by default. Do not
use `--share` unless you deliberately want a temporary public Gradio link.

The page offers BM25, dense, hybrid, reranked, and routed retrieval. Reranked
is the default selected mode. Dense modes call local Ollama for each query.
Reranked and routed modes also call the paid DeepInfra reranker. Every generated
financial answer calls the paid DeepInfra answer model. Comparison questions
use a specialized extraction request instead of the normal scalar-answer
request. Simple answers continue through the scalar answerer.

The browser also corrects clear spelling mistakes locally. It searches BM25
with both the original and corrected text, then fuses the results. Dense search,
the paid reranker, and the answer model keep the original question. The page
shows the added search text when it finds a possible correction. Numbers, years,
amounts, short tickers, and identifier fragments are not changed. This feature
uses RapidFuzz and does not require an API key or an index rebuild.

The page defaults to the `FinQA` dataset and `dev` split. Keep these filters for
FinQA development tests. The filters apply before BM25, dense retrieval, fusion,
and reranking. Use `All datasets` or `All splits` only for general chat. Do not
use an all-documents result as a benchmark score.

Each submitted question is independent. The visible chat history helps manual
comparison, but it is not inserted into later retrieval queries. The chat shows
a polished final answer. Comparison questions can show a table, requested
average, absolute change, percentage change, or percentage-point change. Open the closed
**Sources and calculation** section to inspect the page, calculation, and
bounded source preview. Retrieval ranks, latency, and model cost are not shown
in the final answer.

Questions that ask for prompts, keys, or secrets stop locally. Clearly unrelated
story requests also stop locally. These requests do not search the documents or
call a paid model.

Type a question in **Ask a financial question**. Select **Get final answer**.
The selected retrieval mode controls evidence search; it does not disable the
final answer model.

## 15. Bounded development exit plan

1. Rescore saved version 1 answers without model calls.
2. Run one 20-question retrieved-and-oracle evaluation with prompt version 2.
3. Save version 2 under a new output name. Do not overwrite version 1.
4. Compare version 1 and version 2 once.
5. Stop FinQA prompt tuning after that comparison.
6. Evaluate ConvFinQA and TAT-DQA development retrieval.
7. Run one small answer sample for each remaining dataset.
8. Freeze the selected configuration before test evaluation.

Example version 2 output base:

```powershell
python -m t2rag --config configs/default.yaml evaluate-answers --dataset finqa --split dev --mode both --retrieval-mode reranked --limit 20 --output work/outputs/answers-finqa-dev-v2
```

## 16. Freeze the system

```powershell
Copy-Item configs/default.yaml configs/frozen.yaml
```

Open `configs/frozen.yaml`. Set `allow_test_questions: true` and keep
`allow_test_labels: false`. Do not change this file after freezing.

```powershell
t2rag --config configs/frozen.yaml capture-models
t2rag --config configs/frozen.yaml freeze
```

The first command records exact Ollama model digests and cloud model IDs.
The freeze command refuses to continue if this model record is missing.

Do this only after dev settings are final. Save the generated freeze manifest.

## 17. Run final test evaluation

First enable test questions in a copied configuration file. Keep test labels disabled.

Generate test predictions:

```powershell
t2rag --config configs/frozen.yaml evaluate-answers --split test --mode retrieved --predictions-only --frozen-manifest work/outputs/freeze-manifest.json
```

Score saved predictions only after the full system is frozen:

```powershell
t2rag --config configs/frozen.yaml score-predictions --split test --predictions work/outputs/answers-all-test.predictions.jsonl --frozen-manifest work/outputs/freeze-manifest.json --allow-test-labels
```

Never tune settings after viewing test scores.
