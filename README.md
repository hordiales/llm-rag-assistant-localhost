Summary
===============================

llm-rag-assistant is a fully local, retrieval-augmented chatbot powered by llama-cpp-python. It answers questions in Spanish using your own Q&A dataset: FAISS + multilingual sentence-transformers retrieve relevant context, and a local instruction-tuned LLM (default: Gemma 3 1B Instruct, GGUF) generates the response.

> Looking for the Spanish version? See `README_es.md`.

## 🚀 Features

- 🔍 Semantic search with multilingual sentence-transformers
- 🧠 Local LLM inference with llama-cpp-python (CPU friendly GGUF models)
- 💻 Runs on standard laptops/desktops — no GPU or CUDA required
- 🔒 100% offline, no API keys or external services
- 🗂️ Works with any JSON Q&A dataset

Quick Start (RAG Console Bot)
===============================

This repository ships a console-based RAG chatbot that runs entirely offline.

Requirements
------------
1. Python 3.9+
2. Install dependencies (recommend `llama-cpp-python >= 0.3.2` for Gemma 3 support):
   ```bash
   pip install "llama-cpp-python>=0.3.2" faiss-cpu sentence-transformers
   ```
   On macOS you can fall back to conda if compilation fails:
   ```bash
   conda install -c conda-forge llama-cpp-python
   pip install faiss-cpu sentence-transformers
   ```
3. Download a GGUF model and place it under `../models/`:
   - **Gemma 3 1B Instruct (recommended)**
     ```bash
     wget https://huggingface.co/google/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf \
       -O gemma-3-1b-it.Q4_K_M.gguf
     ```
   - **Mistral-7B-Instruct**
     ```bash
     wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf \
       -O mistral-7b-instruct.Q4_K_M.gguf
     ```

   > ℹ️ Hugging Face may require you to sign in and accept the license before downloading. If you hit a 403 error, open the model page, accept the terms, and rerun the command.

   Always verify file integrity by comparing the `sha256` hash against the value published by the model provider:
   ```bash
   sha256 gemma-3-1b-it.Q4_K_M.gguf
   sha256 mistral-7b-instruct.Q4_K_M.gguf
   ```

   - Gemma 3: Gemma license → https://ai.google.dev/gemma
   - Mistral 7B: Apache 2.0 → https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1

   Recommended steps before downloading Gemma:
   1. Sign in with your Hugging Face account and **accept the license** in the *Files and versions* tab of `google/gemma-3-1b-it-GGUF`.
   2. Log into the CLI: `huggingface-cli login` (or use an access token).
   3. Run the `wget`/`huggingface-cli download` commands above.
   4. Confirm `llama-cpp-python >= 0.3.2`; older releases throw “unknown model architecture: 'gemma3'”.

   Transformers backend (optional):
   ```bash
   huggingface-cli download google/gemma-3-1b-it \
     --local-dir ../models/gemma-3-1b-it-transformers \
     --local-dir-use-symlinks False
   ```
   > Requires license acceptance and `huggingface-cli login`.

4. Build your Q&A dataset in `qa_dataset.json`:
   ```json
   [
     {
       "pregunta": "¿Cuál es el horario de atención?",
       "respuesta": "Nuestro horario es de lunes a viernes de 9 a 18 y sábados de 9 a 14."
     },
     {
       "pregunta": "¿Cómo puedo contactar con soporte técnico?",
       "respuesta": "Puedes escribir a soporte@empresa.com o llamar al 900-123-456."
     }
   ]
   ```
5. Configure `config.yaml`:
   ```yaml
   models:
     embeddings:
       model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
     generation:
       llama_cpp_model_path: "../models/gemma-3-1b-it.Q4_K_M.gguf"
       max_tokens: 256
   ```

Project Structure
-----------------
- `prepare_embeddings.py` → builds `dataset_index.faiss` and `qa.json`
- `chatbot_rag_local.py` → console chatbot using llama-cpp
- `chatbot_rag_local_transformers.py` → transformers-based alternative
- `qa_dataset.json` → user knowledge base (input)

Usage
-----
1. `python prepare_embeddings.py`
2. `python chatbot_rag_local.py`
3. Chat with your knowledge base in Spanish :)

Alternative without llama.cpp (transformers)
-------------------------------------------
1. `pip install torch transformers accelerate`
2. Download Hugging Face weights to `../models/gemma-3-1b-it-transformers`
3. Update `config.yaml` → `transformers` section (path/device/dtype)
4. `python chatbot_rag_local_transformers.py`

> Needs ample RAM (12 GB+ recommended) or a GPU for smooth inference.

Models available under `../models`
----------------------------------
| Model | Strengths | Considerations |
|-------|-----------|----------------|
| `gemma-3-1b-it.Q4_K_M.gguf` | ✅ 1B parameters, Q4_K_M quantization (~2.1 GB). Fast startup on CPU/MPS.<br>✅ Tuned for Spanish/science; low hallucination rate.<br>✅ Works great on Apple Silicon and AVX2-only CPUs. | ℹ️ Must accept Gemma license and use `llama-cpp-python` ≥0.3.2.<br>ℹ️ Smaller model may need more explicit prompts for long answers. |
| `mistral-7b-instruct.Q4_K_M.gguf` | ✅ 7B parameters, robust with generic prompts.<br>✅ Widely battle-tested. | ⚠️ ~4.1 GB on disk, slower on CPU.<br>⚠️ Higher RAM usage (~7–8 GB with long contexts). |
| `qwen2.5-1.5b-instruct-q2_k.gguf` | ✅ Ultra-light (<1 GB), ideal for tight hardware budgets.<br>✅ Good multilingual coverage. | ⚠️ Aggressive Q2 quantization → lower fidelity.<br>⚠️ Needs carefully structured prompts. |

**Recommendation**: Gemma 3 1B Instruct offers the best balance for this project—fast, accurate in Spanish, and resource-friendly. Keep Mistral as a backup if you need longer answers and have extra RAM.

Evaluation & Metrics
====================

Available scripts
-----------------
- `model_evaluation.py` → generates answers (default) or runs BERTScore (`python model_evaluation.py bertscore`)
- `calculate_metrics_from_json.py` → recomputes BERTScore, ROUGE, BLEU, cosine similarity from an existing JSON
- `real_rag_evaluation.py` → end-to-end evaluation of the live RAG pipeline

Sample workflow
---------------
```bash
# 1. Generate answers
python model_evaluation.py

# 2. Compute BERTScore on existing results
python model_evaluation.py bertscore
```

Outputs:
- `evaluation_results.json` – question/ground-truth/generated triples
- `bertscore_results.json` – BERTScore stats and per-sample metrics

Metric interpretation
---------------------
- **BERTScore F1**
  - > 0.85 → Excellent
  - 0.70–0.85 → Good
  - 0.50–0.70 → Needs improvement
  - < 0.50 → Problematic
- **ROUGE-1/2/L** → Unigram/bigram/longest-sequence overlap
- **BLEU-4** → 4-gram precision (typical range 0.2–0.6 for natural text)

Configuration knobs (in the scripts)
------------------------------------
```python
n_samples = 15        # Number of Q&A pairs to evaluate
random.seed(42)       # Reproducibility
lang = "es"           # Language for BERTScore
```

Example output
--------------
```
📊 BERTScore (semantic similarity):
   Precision: 0.7724 ± 0.0879
   Recall:    0.8905 ± 0.0591
   F1-Score:  0.8265 ± 0.0732

📝 ROUGE:
   ROUGE-1:   0.5064 ± 0.2007
   ROUGE-2:   0.4026 ± 0.2220
   ROUGE-L:   0.4760 ± 0.2138
```

Hardware recommendations
------------------------
- Minimum 8 GB RAM (16 GB preferred)
- ~5 GB free disk space for models and indexes
