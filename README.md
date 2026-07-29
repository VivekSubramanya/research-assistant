# Research Assistant (arXiv RAG)

A local, arXiv-grounded research assistant with a Streamlit chat UI.

This app:
- Classifies query intent (`default`, `comparison`, `citation_request`, `paper_level_query`)
- Retrieves context from a local vector index (FAISS + SQLite)
- Falls back to live arXiv API results when local retrieval is empty
- Uses a local GGUF model through `llama-cli`
- Returns structured frontend messages (text, table, citations, etc.)

## Current Status

This repository is configured to reproduce the behavior validated in the latest debugging cycle:
- Robust `llama-cli` output extraction (startup/banner noise filtered)
- Paper-ID normalization (`2406.15531v2 -> 2406.15531`)
- Auto-ingestion when a requested paper is missing/incomplete
- Paper-level retrieval strategy:
  - Broad/detailed paper queries use larger context
  - Precise paper queries focus on the top relevant chunks
- Default fallback queries (for example, "top 5 papers") now pass all fallback entries into prompt context (no silent truncation to 2)

## Architecture

```text
User query
  -> Orchestrator (intent + routing)
  -> Retrieval
       - local FAISS/SQLite semantic search, or
       - paper chunk retrieval by arXiv ID, or
       - arXiv API fallback
  -> Prompt builder (intent-aware context policy)
  -> local llama-cli (GGUF)
  -> normalized JSON-like response
  -> Streamlit chat renderer
```

Key files:
- `ResearchAssistant.py`: app entrypoint with Windows-safe pre-imports
- `interface.py`: Streamlit chat UI
- `orchestrator.py`: intent routing + retrieval flow
- `llm.py`: prompt construction + local model invocation + response normalization
- `query_processing.py`: embeddings and FAISS retrieval
- `chunk_and_embed.py`: PDF text extraction + chunking + embedding persistence
- `APIs/arxiv_api.py`: arXiv API utilities

## Prerequisites

- Python 3.11 or 3.12
- Windows (current local CLI integration is set up for `llama-cpp precompiled/llama-cli.exe`)
- A local GGUF model file at:
  - `models/qwen2.5-14b-research-assistant.Q4_K_M.gguf`
- Internet access for arXiv fallback queries and paper downloads

## Setup

### 1) Clone and enter the repo

```powershell
git clone https://github.com/VivekSubramanya/research-assistant.git
cd research-assistant
```

### 2) Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3) Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Ensure required local model assets exist

Required paths used by runtime:
- `llama-cpp precompiled/llama-cli.exe`
- `models/qwen2.5-14b-research-assistant.Q4_K_M.gguf`

If these files are missing, local LLM calls will fail and the app will degrade to context-only fallback behavior.

## Build the Project GGUF (End-to-End)

This section documents the exact pipeline used by this project to produce the runtime model file:

- Base model: `Qwen/Qwen2.5-14B-Instruct`
- Fine-tuning method: QLoRA (LLaMA-Factory)
- Training data source: `training/generate_training_data.py`
- Final runtime artifact:
  - `models/qwen2.5-14b-research-assistant.Q4_K_M.gguf`

### Time and hardware estimates

These are practical ranges, not hard guarantees. Actual duration varies with GPU tier, CPU speed, storage, internet bandwidth, and dataset size.

| Stage | Typical time | Main hardware pressure | Notes |
|---|---:|---|---|
| Generate training data (`training/generate_training_data.py`) | 30 to 180 min | CPU + Ollama model runtime | Depends heavily on `--arxiv-count`, `--rag-count`, and `--intent-count`. |
| QLoRA training (`llamafactory-cli train`) | 4 to 16 hours | GPU VRAM + compute | For Qwen2.5-14B with 4-bit QLoRA, plan for roughly 12 to 16 GB VRAM minimum. |
| Export merged HF model (`llamafactory-cli export`) | 20 to 90 min | CPU RAM + disk I/O | Writes large model files; SSD strongly recommended. |
| Convert HF -> F16 GGUF (`convert_hf_to_gguf.py`) | 20 to 60 min | CPU + RAM + disk I/O | Slower on HDD or memory-constrained machines. |
| Quantize F16 -> Q4_K_M | 15 to 45 min | CPU + RAM | `Q4_K_M` reduces size and improves inference throughput. |

Approximate storage planning:
- Temporary working space during merge/convert can exceed 100 GB.
- Keep at least 150 GB free to avoid mid-pipeline failures.
- Final Q4_K_M artifact is much smaller than the merged F16 model but still multi-GB.

### 1) Generate training datasets for this project

The generator writes LLaMA-Factory-compatible datasets and `dataset_info.json`.

```powershell
# From repo root
.\venv\Scripts\Activate.ps1

# Optional: choose the local Ollama model used to synthesize examples
$env:OLLAMA_MODEL = "qwen2.5:14b"

python .\training\generate_training_data.py `
  --output-dir .\training\data `
  --arxiv-count 1000 `
  --rag-count 1000 `
  --intent-count 500
```

Expected outputs:
- `training/data/arxiv_params.json`
- `training/data/rag_responses.json`
- `training/data/intent_classification.json`
- `training/data/dataset_info.json`

### 2) Prepare LLaMA-Factory workspace

This repo stores training configs under `training/llamafactory/`, while LLaMA-Factory runs in its own repo.

```powershell
# Example path; adjust if your LLaMA-Factory is elsewhere
cd C:\path\to\LLaMA-Factory

pip install -e ".[torch,metrics]"
```

Copy this project's generated data into LLaMA-Factory `data/`:

```powershell
Copy-Item -Path "C:\Users\pc\OneDrive\Desktop\ResearchAssistant\training\data\*" -Destination ".\data\" -Recurse -Force
```

Copy this project's training YAML files into your LLaMA-Factory root (or reference them by absolute path):

```powershell
Copy-Item -Path "C:\Users\pc\OneDrive\Desktop\ResearchAssistant\training\llamafactory\train_qwen2.5_14b_lora.yaml" -Destination ".\"
Copy-Item -Path "C:\Users\pc\OneDrive\Desktop\ResearchAssistant\training\llamafactory\export_gguf.yaml" -Destination ".\"
```

### 3) Train the adapter (QLoRA)

This uses the exact config in `training/llamafactory/train_qwen2.5_14b_lora.yaml`:
- `model_name_or_path: Qwen/Qwen2.5-14B-Instruct`
- datasets: `arxiv_params,rag_responses,intent_classification`
- output adapter dir: `./saves/qwen2.5-14b-lora-research-assistant`

```powershell
cd C:\path\to\LLaMA-Factory
llamafactory-cli train .\train_qwen2.5_14b_lora.yaml
```

### 4) Export merged model (HF format)

This uses the exact config in `training/llamafactory/export_gguf.yaml`.

```powershell
cd C:\path\to\LLaMA-Factory
llamafactory-cli export .\export_gguf.yaml
```

Expected merged model output:
- `C:\path\to\LLaMA-Factory\models\qwen2.5-14b-research-assistant-merged`

### 5) Convert merged HF model to F16 GGUF

Use `llama.cpp` converter:

```powershell
cd C:\path\to\llama.cpp
python .\convert_hf_to_gguf.py `
  "C:\path\to\LLaMA-Factory\models\qwen2.5-14b-research-assistant-merged" `
  --outfile "C:\Users\pc\OneDrive\Desktop\ResearchAssistant\models\qwen2.5-14b-research-assistant.f16.gguf" `
  --outtype F16
```

### 6) Quantize to Q4_K_M GGUF used by this app

Option A (project script, current repo path assumptions):

```powershell
cd C:\Users\pc\OneDrive\Desktop\ResearchAssistant
python .\quantize_gguf.py
```

Option B (llama.cpp quantizer):

```powershell
cd C:\path\to\llama.cpp
.\build\bin\Release\llama-quantize.exe `
  "C:\Users\pc\OneDrive\Desktop\ResearchAssistant\models\qwen2.5-14b-research-assistant.f16.gguf" `
  "C:\Users\pc\OneDrive\Desktop\ResearchAssistant\models\qwen2.5-14b-research-assistant.Q4_K_M.gguf" `
  Q4_K_M
```

### 7) Point runtime to the produced GGUF

Runtime default path in `llm.py` already expects:
- `models/qwen2.5-14b-research-assistant.Q4_K_M.gguf`

Also ensure the CLI binary exists:
- `llama-cpp precompiled/llama-cli.exe`

Then run:

```powershell
streamlit run ResearchAssistant.py
```

### 8) Quick verification after GGUF build

In the app, run these prompts:
- `Tell me about paper 2406.15531`
- `What are the top 5 papers for computer science?`

And verify logs include successful `stage=llm_call output={'status': 'ok' ...}` entries.

## Run the app

```powershell
streamlit run ResearchAssistant.py
```

Open the local URL shown by Streamlit (typically `http://localhost:8501`).

## Reproducibility Checklist (Current Behavior)

Use these prompts to validate the latest fixes:

1. Top-N fallback context coverage
- Prompt: `What are the top 5 papers for computer science?`
- Expected:
  - `stage=arxiv_fallback_complete` shows `chunk_count: 5`
  - Answer should reflect multiple distinct papers from fallback context

2. Versioned paper ID normalization
- Prompt: `Tell me about paper 2406.15531v2`
- Expected:
  - Routed as `paper_level_query`
  - Paper lookup normalizes to `2406.15531`

3. Paper-level precise query behavior
- Prompt: `What dataset is used in paper 2406.15532?`
- Expected:
  - Focused answer; if dataset is absent in context, explicitly says so

4. Paper-level detailed behavior
- Prompt: `Tell me about paper 2406.15532 in detail`
- Expected:
  - More comprehensive summary when context allows

## Data and Storage

Runtime-generated local artifacts:
- `research.db` (SQLite)
- `faiss_index.bin` (vector index)
- `papers/` (downloaded PDFs)
- `logs.txt`

These are intentionally ignored by git in `.gitignore`.

## Running tests

```powershell
pip install -r requirements-test.txt
pytest -q
```

Focused regression run used during recent fixes:

```powershell
pytest tests/test_units.py -k "fallback_papers_for_top_n_query or falls_back_to_chunk_summary_when_llm_is_unavailable"
```

## Configuration knobs

Environment variables used in `llm.py`:
- `RESEARCH_ASSISTANT_CTX_SIZE` (default: `8192`)
- `RESEARCH_ASSISTANT_LLM_TIMEOUT_SECONDS` (default: `180`)
- `RESEARCH_ASSISTANT_JSON_TOKENS` (default: `256`)
- `RESEARCH_ASSISTANT_ANSWER_TOKENS` (default: `768`)

Example:

```powershell
$env:RESEARCH_ASSISTANT_CTX_SIZE = "8192"
$env:RESEARCH_ASSISTANT_LLM_TIMEOUT_SECONDS = "240"
streamlit run ResearchAssistant.py
```

## Training assets

Training and export helpers are available under `training/` and `training/llamafactory/`.
See `training/llamafactory/README.md` for LoRA training/export workflow.

## Notes and limitations

- The app is designed to be arXiv-context grounded. It should not invent facts beyond provided context.
- arXiv API can return temporary `500` errors; fallback path handles this and returns a degraded message.
- Current local CLI integration assumes Windows pathing and the included `llama-cli.exe` location.

## License

No license file is currently included. 
