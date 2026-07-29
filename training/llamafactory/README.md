# LLaMA-Factory Training & Export Guide

This folder contains configs for fine-tuning **Qwen2.5-14B-Instruct** on the synthetic datasets produced by `training/generate_training_data.py`.

## Assumptions

- LLaMA-Factory is installed in a separate environment or directory.
- The datasets are in `training/data/` relative to this repository.
- You have a GPU with enough VRAM. Qwen2.5-14B QLoRA training needs ~12–16 GB VRAM.

## 1. Copy dataset_info.json and datasets into LLaMA-Factory

From your LLaMA-Factory directory:

```powershell
# Windows example
Copy-Item -Path "C:\Users\pc\OneDrive\Desktop\ResearchAssistant\training\data\*" -Destination "data\" -Recurse -Force
```

Or create a symlink if you prefer to keep the data in this repo:

```powershell
New-Item -ItemType Junction -Path "data\research_assistant" -Target "C:\Users\pc\OneDrive\Desktop\ResearchAssistant\training\data"
```

If you use a symlink, update `dataset_info.json` entries to point to the new paths, or keep the files directly in `data/`.

## 2. Install LLaMA-Factory (if not already done)

```powershell
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

## 3. Download the base model

LLaMA-Factory will download `Qwen/Qwen2.5-14B-Instruct` automatically on first run if you have internet access and `model_name_or_path: Qwen/Qwen2.5-14B-Instruct`.

To use a local copy, change `model_name_or_path` in both YAMLs to the absolute path.

## 4. Train

```powershell
llamafactory-cli train train_qwen2.5_14b_lora.yaml
```

Training will:
- Use LoRA rank 32, alpha 64.
- Train for 3 epochs with a 5% validation split.
- Save checkpoints to `./saves/qwen2.5-14b-lora-research-assistant`.

Monitor VRAM usage. If you run out of memory:
- Increase `gradient_accumulation_steps` and reduce `per_device_train_batch_size` (already 1).
- Reduce `lora_rank` to 16 and `lora_alpha` to 32.
- Set `quantization_bit: 4` in the training YAML if not already active through QLoRA.

## 5. Merge adapter into base model

```powershell
llamafactory-cli export export_gguf.yaml
```

This produces a merged Hugging Face model at:

```text
./models/qwen2.5-14b-research-assistant-merged
```

## 6. Convert merged model to GGUF for Ollama

Use llama.cpp's conversion script. If you do not have llama.cpp:

```powershell
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
pip install -r requirements.txt
```

Then convert:

```powershell
python convert_hf_to_gguf.py `
  "C:\Users\pc\OneDrive\Desktop\ResearchAssistant\training\llamafactory\models\qwen2.5-14b-research-assistant-merged" `
  --outfile qwen2.5-14b-research-assistant.Q4_K_M.gguf `
  --outtype Q4_K_M
```

## 7. Import into Ollama

Use the provided `Modelfile`. Edit the `FROM` path to point to your GGUF file.

```powershell
ollama create qwen2.5-research-assistant -f Modelfile
ollama run qwen2.5-research-assistant
```

## 8. Use the fine-tuned model in your app

Set the environment variable in the same terminal before launching the Streamlit app:

```powershell
$env:OLLAMA_MODEL = "qwen2.5-research-assistant"
python ResearchAssistant.py
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Out of memory during training | Lower `lora_rank`/`lora_alpha`, enable `quantization_bit: 4`, or use a smaller model like Qwen2.5-7B-Instruct. |
| Dataset not found | Ensure `dataset_info.json` and the three `.json` files are inside LLaMA-Factory's `data/` folder. |
| GGUF conversion fails | Make sure llama.cpp is up to date and you installed its Python requirements. |
| Model outputs are not JSON | Increase JSON examples in the RAG dataset, or add a stricter system prompt in `llm.py`. |
