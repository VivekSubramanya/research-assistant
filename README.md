# Research Assistant for arXiv

A local, arXiv-grounded assistant for finding, reading, and comparing research papers through a conversational interface.

The project is built around a simple idea: researchers should be able to ask direct questions about papers without repeatedly searching, downloading, reading, and cross-checking each paper by hand. Answers are generated from retrieved arXiv material, and the assistant is instructed to say when the available context is not sufficient.

## Problem Statement

Academic research is often less straightforward than the final bibliography makes it look. A useful paper may be dozens of pages long, terminology can vary between fields, and comparison usually means opening several PDFs and checking methods, datasets, assumptions, and results one at a time.

This creates a few recurring problems:

- **Reading is expensive in time.** A researcher may inspect many papers before learning that only a few are relevant.
- **Semantic comparison is difficult.** Search tools return papers, but rarely make it easy to compare what those papers claim or how their methods differ.
- **Relevant work can be obscured.** Keyword choices, unfamiliar terminology, and incomplete searches can leave meaningful papers outside the initial approach.
- **Evidence becomes fragmented.** Notes, PDFs, browser tabs, and citations are often spread across different tools, making factual verification slower.

The goal is not to replace careful reading or scholarly judgment. It is to reduce the mechanical work around discovery and first-pass analysis so that more time can be spent evaluating the research itself.

## Solution Overview

Research Assistant works across arXiv and lets a user ask natural-language questions about papers, topics, methods, datasets, and citations. It can:

- discover papers through arXiv;
- answer focused questions about a paper using its retrieved text;
- summarize a paper at different levels of detail;
- compare papers or approaches in a structured format;
- retrieve likely sources for a topic or citation request;
- build a local index so previously ingested material can be searched semantically;
- keep inference local through a quantized GGUF model.

The assistant is intentionally grounded: its prompt restricts the model to supplied arXiv context. When that context does not support an answer, the expected behavior is to state the limitation rather than fill the gap with outside knowledge.

## Architecture

### Top-down flow

```text
User question
    |
    v
Streamlit chat interface
    |
    v
Intent and query routing
    |-- paper-level question
    |-- comparison
    |-- citation request
    `-- general research question
    |
    v
Context retrieval
    |-- local FAISS semantic search
    |-- SQLite paper/chunk lookup
    `-- live arXiv API fallback
    |
    v
Intent-aware prompt construction
    |
    v
Fine-tuned Qwen2.5-14B GGUF through llama.cpp
    |
    v
Normalized answer, table, list, or citations
```

### Main components

| Layer | Implementation | Why it is used |
|---|---|---|
| Interface | Streamlit | Provides an interactive UI without requiring a separate frontend service. |
| Orchestration | Python routing and intent heuristics | Keeps paper lookup, comparison, citation, and general search flows explicit and testable. |
| Paper discovery | arXiv API through `requests` and `feedparser` | Uses arXiv as the primary source instead of mixing unverified web content into the research context. |
| PDF extraction | PyMuPDF with pypdf fallback | PyMuPDF is fast for normal PDFs; pypdf provides a second extraction path when needed. |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Produces compact 384-dimensional embeddings quickly on CPU, which suits a local desktop workflow. |
| Vector search | FAISS | Provides fast similarity search over paper chunks without an external vector database. |
| Metadata storage | SQLite | Keeps papers, chunks, and FAISS mappings in a portable local database. |
| Response model | Fine-tuned Qwen2.5-14B-Instruct | Balances response quality, structured-output reliability, local deployment, and manageable quantization. |
| Inference | llama.cpp and GGUF | Runs quantized models locally on consumer hardware without a mandatory hosted-inference bill. |

### Why Qwen2.5-14B-Instruct?

Qwen2.5-14B-Instruct sits in a useful middle ground for this project. It is large enough to follow multi-part research instructions, produce structured JSON-like responses, and synthesize several retrieved passages, while still being practical to quantize and run locally. Its open weights also make domain-specific QLoRA fine-tuning and GGUF distribution possible.

This repository fine-tunes the model on three task families:

- arXiv query-parameter generation;
- grounded research-assistant responses;
- intent classification.

The runtime artifact uses `Q4_K_M` quantization. This reduces memory and storage requirements while retaining more capability than very small local models. The tradeoff is that a 14B model is still demanding on older CPUs and low-memory systems.

### Model alternatives

The best alternative depends on where the assistant will run and what it can cost.

| Situation | Practical alternative | Tradeoff |
|---|---|---|
| Lower-memory laptop or faster CPU inference | Qwen2.5-7B-Instruct in GGUF | Faster and cheaper to run, but comparisons and strict structured output may be less reliable. Evaluate or fine-tune it on this project's datasets before replacement. |
| Very constrained local device | A 3B-class instruct model | Easier to run, but more likely to miss details across long or multi-paper context. |
| Workstation with more RAM or VRAM | Qwen2.5-32B-Instruct or a comparable larger open model | Better synthesis may be possible, but model size, inference time, and training cost rise considerably. |
| Minimal local hardware | A hosted model API such as OpenAI, Anthropic, or Gemini | Removes local inference requirements and may improve reasoning, but adds recurring cost, network dependence, and different privacy considerations. This requires adapting the current llama.cpp call layer. |

For embedding retrieval, `all-MiniLM-L6-v2` favors speed and small local indexes. Larger E5 or BGE embedding models may improve retrieval quality in some domains, but use more compute and require rebuilding the FAISS index because their vector dimensions differ.

### Repository map

| Path | Responsibility |
|---|---|
| `ResearchAssistant.py` | Windows-safe application entry point. |
| `interface.py` | Streamlit chat interface and session transcript. |
| `orchestrator.py` | Intent routing, retrieval selection, ingestion, and fallback flow. |
| `llm.py` | Query expansion, prompt construction, llama.cpp invocation, and response normalization. |
| `query_processing.py` | Query embeddings and FAISS retrieval. |
| `chunk_and_embed.py` | PDF extraction, semantic chunking, embeddings, and index persistence. |
| `data_ingestion.py` | arXiv metadata lookup, PDF download, and database initialization. |
| `APIs/arxiv_api.py` | arXiv query, retry, parsing, and download utilities. |
| `training/` | Dataset generation, evaluation, QLoRA configuration, export, and inspection tools. |
| `tests/` | Unit, contract, edge-case, and integration tests. |

## Requirements

### Runtime

- Windows 10 or 11. The current CLI path expects `llama-cli.exe`.
- Python 3.11 or 3.12.
- Internet access for package installation, live arXiv searches, and PDF downloads.
- A local `llama-cli.exe` build compatible with the GGUF model.
- Enough free memory for a 14B Q4 model. About 16 GB of system RAM is a practical starting point; more memory or supported GPU offload will improve the experience.
- Several gigabytes of free storage for the model, downloaded PDFs, database, and vector index.

### Building the model yourself

The full training path is substantially more demanding than runtime inference:

- A CUDA-capable GPU is strongly recommended.
- The provided 4-bit QLoRA configuration is intended for roughly 12-16 GB of VRAM, although exact use depends on the software stack and hardware.
- Keep approximately 150 GB free during merge and conversion. Intermediate Hugging Face and F16 files are much larger than the final quantized GGUF.
- An SSD is strongly recommended.

## Setup

There are two setup paths:

1. **Use the provided GGUF**, which is the quickest way to run the project.
2. **Build your own GGUF**, which reproduces the training and export pipeline.

Both paths require the repository, Python dependencies, and a llama.cpp CLI binary.

### Common setup

#### 1. Clone the repository

```powershell
git clone https://github.com/VivekSubramanya/research-assistant.git
cd research-assistant
```

#### 2. Create and activate a virtual environment

```powershell
python -m venv venv
& ./venv/Scripts/Activate.ps1
```

If PowerShell blocks activation, allow scripts for the current process and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ./venv/Scripts/Activate.ps1
```

#### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

#### 4. Provide llama.cpp

The runtime expects this executable:

```text
llama-cpp precompiled/llama-cli.exe
```

Build or obtain a Windows llama.cpp release, create the `llama-cpp precompiled` directory in the project root, and place `llama-cli.exe` there. The binary directory is excluded from Git because compiled artifacts are large and platform-specific.

### Option A: Use the provided GGUF

1. Download **only the GGUF model** from the [project Google Drive folder](https://drive.google.com/drive/folders/1oPKMnV7Je9fQbhud6a9KCsJQHagApGVZ?usp=drive_link).
2. Create a `models` directory in the project root if it does not exist.
3. Place the downloaded file at this exact path:

```text
models/qwen2.5-14b-research-assistant.Q4_K_M.gguf
```

The download contains the model artifact only. Install Python packages and provide `llama-cli.exe` through the common setup steps above.

Your runtime layout should include:

```text
research-assistant/
|-- llama-cpp precompiled/
|   `-- llama-cli.exe
|-- models/
|   `-- qwen2.5-14b-research-assistant.Q4_K_M.gguf
|-- ResearchAssistant.py
`-- requirements.txt
```

### Option B: Build your own GGUF

The project model is based on `Qwen/Qwen2.5-14B-Instruct` and trained with 4-bit QLoRA through LLaMA-Factory.

#### 1. Generate training datasets

The generator uses a local Ollama model to create and validate three datasets. Make sure Ollama is running and the selected generation model is available.

```powershell
$env:OLLAMA_MODEL = "qwen2.5:14b"

python ./training/generate_training_data.py `
  --output-dir ./training/data `
  --arxiv-count 1000 `
  --rag-count 1000 `
  --intent-count 500
```

Generated outputs:

```text
training/data/arxiv_params.json
training/data/rag_responses.json
training/data/intent_classification.json
training/data/dataset_info.json
```

Generation commonly takes 30-180 minutes depending on the model, hardware, and sample counts.

#### 2. Install LLaMA-Factory

Use a separate clone or environment:

```powershell
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
python -m pip install -e ".[torch,metrics]"
```

Copy this project's generated datasets into LLaMA-Factory:

```powershell
Copy-Item -Path "C:/path/to/research-assistant/training/data/*" `
  -Destination "./data/" -Recurse -Force
```

Copy the provided training and export configurations:

```powershell
Copy-Item "C:/path/to/research-assistant/training/llamafactory/train_qwen2.5_14b_lora.yaml" "./"
Copy-Item "C:/path/to/research-assistant/training/llamafactory/export_gguf.yaml" "./"
```

#### 3. Train the QLoRA adapter

```powershell
llamafactory-cli train ./train_qwen2.5_14b_lora.yaml
```

The checked-in configuration uses:

- 4-bit bitsandbytes quantization;
- LoRA rank 32 and alpha 64;
- two training epochs;
- a 5% validation split;
- output at `./saves/qwen2.5-14b-lora-research-assistant`.

Training typically takes several hours. Monitor VRAM use and the generated loss plot rather than relying on a fixed estimate.

#### 4. Merge the adapter

```powershell
llamafactory-cli export ./export_gguf.yaml
```

The merged Hugging Face model is written to:

```text
./models/qwen2.5-14b-research-assistant-merged
```

#### 5. Convert the merged model to F16 GGUF

From an up-to-date llama.cpp clone:

```powershell
python ./convert_hf_to_gguf.py `
  "C:/path/to/LLaMA-Factory/models/qwen2.5-14b-research-assistant-merged" `
  --outfile "C:/path/to/research-assistant/models/qwen2.5-14b-research-assistant.f16.gguf" `
  --outtype F16
```

#### 6. Quantize to Q4_K_M

Using the llama.cpp quantizer:

```powershell
./build/bin/Release/llama-quantize.exe `
  "C:/path/to/research-assistant/models/qwen2.5-14b-research-assistant.f16.gguf" `
  "C:/path/to/research-assistant/models/qwen2.5-14b-research-assistant.Q4_K_M.gguf" `
  Q4_K_M
```

The repository also includes `quantize_gguf.py`, but its paths assume local `llama.cpp/gguf-py` tooling inside this workspace. The llama.cpp quantizer above is the more portable route.

#### 7. Confirm runtime artifacts

Before launching the app, verify both paths exist:

```text
llama-cpp precompiled/llama-cli.exe
models/qwen2.5-14b-research-assistant.Q4_K_M.gguf
```

## How to Use

### Start the application

From the activated virtual environment:

```powershell
streamlit run ResearchAssistant.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`.

### Ask research questions

| Goal | Example |
|---|---|
| Discover papers | `What are the top 5 papers on retrieval-augmented generation?` |
| Summarize by ID | `Tell me about paper 2406.15531.` |
| Request detail | `Tell me about paper 2406.15531 in detail.` |
| Inspect a specific fact | `What dataset is used in paper 2406.15531?` |
| Compare approaches | `Compare retrieval-augmented generation and long-context language models.` |
| Request sources | `Give me sources on graph neural networks for molecular property prediction.` |

Paper IDs with version suffixes are accepted. For example, `2406.15531v2` is normalized to `2406.15531` for local storage and retrieval.

Each prompt is currently handled independently. The visible transcript is useful for reference, but follow-up questions should repeat the paper ID or subject instead of relying on conversational memory.

### Understand first-run behavior

- A direct paper-ID request can trigger PDF download, extraction, chunking, and indexing before the answer is generated.
- A general query searches the local FAISS index first.
- When local retrieval is empty, the assistant queries arXiv and constructs context from returned titles and summaries.
- Local artifacts accumulate as the application is used, so later searches can reuse ingested papers.

### View logs

Recent logs are available in the interface, and the full `logs.txt` file can be downloaded from the sidebar. Logs record intent routing, retrieval, fallback, model calls, and completion timing.

## Configuration

| Variable | Default | Purpose |
|---|---:|---|
| `RESEARCH_ASSISTANT_CTX_SIZE` | `8192` | llama.cpp context window. |
| `RESEARCH_ASSISTANT_LLM_TIMEOUT_SECONDS` | `180` | Maximum model-call duration. |
| `RESEARCH_ASSISTANT_JSON_TOKENS` | `256` | Output budget for structured helper calls. |
| `RESEARCH_ASSISTANT_ANSWER_TOKENS` | `768` | Output budget for research answers. |

Example:

```powershell
$env:RESEARCH_ASSISTANT_CTX_SIZE = "8192"
$env:RESEARCH_ASSISTANT_LLM_TIMEOUT_SECONDS = "240"
streamlit run ResearchAssistant.py
```

Larger context and output budgets consume more memory and increase response time.

## Data and Storage

| Path | Contents |
|---|---|
| `research.db` | SQLite paper metadata, chunks, queries, and FAISS mappings. |
| `faiss_index.bin` | Persisted semantic-search index. |
| `papers/` | PDFs downloaded automatically for direct paper-ID questions. |
| `pdfs/` | PDFs downloaded through the standalone ingestion workflow. |
| `logs.txt` | Runtime logs. |

Models, databases, indexes, PDFs, logs, and compiled binaries are excluded from Git.

## Tests

```powershell
python -m pip install -r requirements-test.txt
python -m pytest tests -q
```

The suite covers unit behavior, response contracts, edge cases, and the ingestion-to-answer integration path.

## Limitations

- This is a research aid, not a substitute for reading the original paper or verifying a claim before publication.
- Grounding reduces hallucination risk but does not eliminate extraction, retrieval, or interpretation errors.
- arXiv does not include every journal, conference, dataset, or proprietary source.
- Scanned PDFs and unusual layouts may extract poorly.
- Semantic retrieval quality depends on papers already ingested and the embedding model's representation of the query.
- The runtime is Windows-oriented and expects a local `llama-cli.exe` at a fixed project path.
- The reasoning layer is stateless between prompts; it does not yet resolve references such as "the second paper" from earlier messages.

## Responsible Use

Treat generated answers as a starting point for investigation. Open the cited arXiv paper, confirm important claims in the source text, and apply the citation and attribution standards required by your institution or workplace.

## License

No license file is currently included.
