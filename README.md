# T2-RAGBench Financial RAG

A local-first Retrieval-Augmented Generation (RAG) application for financial reports in
the [T2-RAGBench dataset](https://huggingface.co/datasets/G4KMU/t2-ragbench).

The application provides:

- SQLite FTS5 BM25 lexical retrieval.
- FAISS dense vector retrieval.
- Local Ollama embeddings.
- Hybrid retrieval with Reciprocal Rank Fusion.
- Optional DeepInfra reranking and answer generation.
- Table-aware financial document chunks.
- Python calculation, unit, and citation checks.
- Local spelling-match support with RapidFuzz.
- Prompt-injection and secret-request guardrails.
- A local Gradio browser interface.

The repository does not include the dataset, generated indexes, saved outputs, or API
keys. A fresh clone must build its own registry and indexes.

## Requirements

Install these tools before you start:

- Python 3.11 or Python 3.12.
- Git.
- Git LFS and Git Xet for the dataset.
- Ollama for local embedding models.
- A DeepInfra account for reranking and generated answers.
- VS Code with a PowerShell terminal on Windows.

An NVIDIA GPU is recommended for Ollama. BM25 retrieval does not need a GPU.

## Python dependencies

The dependency versions are stored in `pyproject.toml`.

### Required packages

| Package | Supported version | Purpose |
|---|---|---|
| `faiss-cpu` | `>=1.9,<2` | Local dense-vector index |
| `gradio` | `>=6,<7` | Browser chat interface |
| `httpx` | `>=0.27,<1` | Ollama and DeepInfra requests |
| `numpy` | `>=1.26,<3` | Vector and numeric operations |
| `pdfplumber` | `>=0.11,<1` | PDF text and table extraction |
| `Pillow` | `>=10,<13` | Image handling |
| `pypdf` | `>=6,<7` | PDF validation and metadata |
| `pypdfium2` | `>=4.30,<5` | PDF page rendering |
| `python-dotenv` | `>=1,<2` | Local environment-file loading |
| `PyYAML` | `>=6,<7` | YAML configuration loading |
| `rapidfuzz` | `>=3.10,<4` | Local spelling-match suggestions |

### Development packages

| Package | Supported version | Purpose |
|---|---|---|
| `pytest` | `>=8,<9` | Automated tests |
| `pytest-cov` | `>=5,<7` | Test coverage |
| `ruff` | `>=0.8,<1` | Static code checks |

### Optional package

| Package | Supported version | Purpose |
|---|---|---|
| `sentence-transformers` | `>=3.3,<6` | Alternative embedding provider |

The default configuration uses Ollama embeddings. It does not require
`sentence-transformers`.

## Models

| Model | Provider | Purpose |
|---|---|---|
| `qwen3-embedding:4b` | Local Ollama | Dense document and query embeddings |
| `qwen3:8b` | Local Ollama | Optional descriptions and diagnostic judge |
| `qwen3-vl:8b-instruct` | Local Ollama | Optional visual page descriptions |
| `Qwen/Qwen3-Reranker-4B` | DeepInfra | Candidate reranking |
| `Qwen/Qwen3-235B-A22B-Instruct-2507` | DeepInfra | Final answers and comparisons |

The embedding model is required for dense, hybrid, and reranked retrieval. The other
local models are optional unless you enable their related features.

## 1. Open the project in VS Code

Open the cloned project folder in VS Code. Open a PowerShell terminal in that folder.

All commands below assume that the terminal is in the project root.

## 2. Check Python

```powershell
python --version
```

Use Python 3.11 or Python 3.12.

## 3. Create the virtual environment

```powershell
python -m venv .venv
```

## 4. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

The prompt should now start with `(.venv)`.

## 5. Update the packaging tools

```powershell
python -m pip install --upgrade pip setuptools wheel
```

## 6. Install the project dependencies

Install the application and development tools:

```powershell
python -m pip install -e ".[dev]"
```

To install the optional Sentence Transformers provider, use:

```powershell
python -m pip install -e ".[dev,sentence-transformers]"
```

You do not need the optional package for the default Ollama configuration.

## 7. Create the environment file

Copy the safe template:

```powershell
Copy-Item .env.example .env
```

Open `.env` and add your DeepInfra key:

```dotenv
DEEPINFRA_API_KEY=your_key_here
```

Never commit `.env`. The `.gitignore` file excludes it.

BM25 retrieval evaluation does not require this key. Reranking, generated answers, and
the current Gradio application require it.

## 8. Download and configure the dataset

Download T2-RAGBench separately from the application repository:

```text
https://huggingface.co/datasets/G4KMU/t2-ragbench
```

The application was developed against dataset commit `adf7fe1`.

Open `configs/default.yaml`. Set `paths.dataset_root` to the downloaded dataset folder.

Example:

```yaml
paths:
  dataset_root: "D:\\New folder (12)\\t2-ragbench"
```

Use your own path. Do not copy the example path unless the dataset is there.

## 9. Start Ollama

Ollama can run as a Windows background service. If it is not running, use:

```powershell
ollama serve
```

Keep that terminal open. Use a second VS Code terminal for the next commands.

Check the installed models:

```powershell
ollama list
```

Install the required embedding model if it is missing:

```powershell
ollama pull qwen3-embedding:4b
```

Install the optional local text model if needed:

```powershell
ollama pull qwen3:8b
```

Install the optional visual model if needed:

```powershell
ollama pull qwen3-vl:8b-instruct
```

Ollama listens on `http://127.0.0.1:11434` by default.

## 10. Run the tests

```powershell
python -m pytest
```

Fix installation or configuration errors before building the indexes.

## 11. Check the project state

```powershell
python -m t2rag --config configs/default.yaml status
```

A new installation should show that the registry and indexes do not exist yet.

## 12. Build the document registry

```powershell
python -m t2rag --config configs/default.yaml build-registry
```

This command:

- Reads the dataset metadata and contexts.
- Deduplicates physical source pages.
- Preserves dataset, split, and source aliases.
- Creates prose and table-aware chunks.
- Excludes questions and gold answers from retrieval text.

## 13. Build the BM25 lexical index

```powershell
python -m t2rag --config configs/default.yaml build-bm25
```

BM25 uses SQLite FTS5. This command does not use Ollama or DeepInfra.

## 14. Test BM25 retrieval

Start with 20 FinQA development questions:

```powershell
python -m t2rag --config configs/default.yaml evaluate-retrieval --dataset finqa --split dev --mode bm25 --limit 20
```

## 15. Build the dense vector index

```powershell
python -m t2rag --config configs/default.yaml build-dense
```

This command sends the document chunks to local `qwen3-embedding:4b`. It saves the FAISS
index and its validation manifest under `work/`.

This step can take time. Successful Ollama requests show `HTTP/1.1 200 OK`.

## 16. Test the other retrieval modes

Dense retrieval:

```powershell
python -m t2rag --config configs/default.yaml evaluate-retrieval --dataset finqa --split dev --mode dense --limit 20
```

Hybrid retrieval:

```powershell
python -m t2rag --config configs/default.yaml evaluate-retrieval --dataset finqa --split dev --mode hybrid --limit 20
```

Reranked retrieval:

```powershell
python -m t2rag --config configs/default.yaml evaluate-retrieval --dataset finqa --split dev --mode reranked --limit 20
```

Reranked retrieval uses the paid DeepInfra reranker. Start with a small limit.

## 17. Test generated answers

Start with five development questions:

```powershell
python -m t2rag --config configs/default.yaml evaluate-answers --dataset finqa --split dev --mode retrieved --retrieval-mode reranked --limit 5
```

This command uses paid reranking and answer-generation requests.

Keep `evaluation.allow_test_labels` set to `false` while you develop and tune the system.

## 18. Ask one question in the terminal

```powershell
python -m t2rag --config configs/default.yaml ask "What was Apple's gross margin in 2004?" --retrieval-mode reranked
```

Replace the example with your own financial question.

## 19. Run the Gradio browser interface

```powershell
python -m t2rag --config configs/default.yaml chat-ui
```

Open this address in a browser:

```text
http://127.0.0.1:7860/
```

Enter one question and select the send control. Press `Ctrl+C` in the terminal to stop
the server.

The server is local by default. Avoid the `--share` option unless you intentionally want
a public link. A public link can expose retrieved evidence and use paid API credit.

## Main YAML settings

Edit `configs/default.yaml` to change the pipeline.

| Setting | Default | Meaning |
|---|---:|---|
| `paths.dataset_root` | Local path | T2-RAGBench dataset location |
| `retrieval.max_words` | 500 | Maximum prose words per chunk |
| `retrieval.overlap_words` | 80 | Repeated words between prose chunks |
| `retrieval.lexical_candidates` | 30 | BM25 candidate count |
| `retrieval.dense_candidates` | 30 | FAISS candidate count |
| `retrieval.fused_candidates` | 30 | Hybrid candidate count |
| `retrieval.rerank_candidates` | 30 | Reranker candidate count |
| `retrieval.answer_contexts` | 8 | Contexts sent to the answer model |
| `retrieval.rrf_k` | 60 | Reciprocal Rank Fusion constant |
| `retrieval.include_descriptions` | `false` | Enable generated retrieval descriptions |
| `generation.correction_attempts` | 1 | Maximum answer correction request |
| `evaluation.allow_test_labels` | `false` | Keep test labels sealed |

Rebuild the registry and indexes after changing the source data or chunk settings.
Rebuild the dense index after changing the embedding model.

Prompt, spelling, and interface changes do not require an index rebuild.

## Retrieval modes

| Mode | BM25 | Local embedding | DeepInfra reranker |
|---|---:|---:|---:|
| `bm25` | Yes | No | No |
| `dense` | No | Yes | No |
| `hybrid` | Yes | Yes | No |
| `reranked` | Yes | Yes | Yes |

Generated answers use DeepInfra in every retrieval mode.

## Generated files

The application stores generated artifacts under `work/`:

```text
work/
  registry.sqlite3
  indexes/
  manifests/
  outputs/
```

Git ignores this folder. It can contain questions, retrieved evidence, model responses,
evaluation labels, and large indexes.

## Project folders

```text
configs/                  YAML configuration
docs/                     Architecture and manual instructions
reports/                  Dataset inspection notes
src/t2rag/ingestion/      Registry, parsing, chunking, descriptions
src/t2rag/retrieval/      BM25, FAISS, embeddings, fusion, reranking
src/t2rag/generation/     Prompts, calculations, answer client
src/t2rag/evaluation/     Metrics, citations, freeze workflow
src/t2rag/safety/         Prompt-injection checks
tests/                    Automated tests
work/                     Generated local artifacts; ignored by Git
```

## Git and secret safety

Never commit these paths:

- `.env`
- `.venv/`
- `work/`
- `tmp/`

Before a GitHub push, confirm that none of them are tracked:

```powershell
git ls-files .env .venv work tmp
```

The command must print nothing. If an API key was ever committed, rotate the key. Removing
the current `.env` file does not remove a secret from Git history.

## Stop and restart

To stop the Gradio or Ollama foreground server, press `Ctrl+C` in its terminal.

After reopening VS Code, activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

You do not need to rebuild the registry or indexes each time. Run the Gradio command or
the single-question command after activation.

