Resumen
===============================

llm-rag-assistant es un chatbot RAG totalmente local basado en llama-cpp-python. Responde preguntas en español usando tu propio dataset de Q&A: FAISS + sentence-transformers multilingües recuperan el contexto relevante y un LLM tipo instruct (por defecto: Gemma 3 1B Instruct en formato GGUF) genera la respuesta.

Versiones probadas
-------------------
| Componente | Versión |
|------------|---------|
| Python | 3.10.14 |
| llama-cpp-python | 0.3.5 |
| faiss-cpu | 1.7.4 |
| sentence-transformers | 2.7.0 |
| torch (CPU) | 2.2.0 |
| transformers | 4.36.2 |
| accelerate | 0.26.0 |
| scikit-learn | 1.4.2 |
| numpy | 1.26.4 |
| scipy | 1.11.4 |
| bert-score | 0.3.13 |
| rouge-score | 0.1.2 |
| nltk | 3.8.1 |

> Todas las dependencias Python están fijas en `requirements.txt` para garantizar reproducibilidad.

## 🚀 Características

- 🔍 Búsqueda semántica con sentence-transformers multilingües
- 🧠 Inferencia local con llama-cpp-python (modelos GGUF optimizados para CPU)
- 💻 Funciona en notebooks o desktops sin GPU ni CUDA
- 🔒 100% offline, sin claves de API ni servicios externos
- 🗂️ Compatible con cualquier dataset de Q&A en JSON

Inicio rápido (bot RAG en consola)
===================================

Este repositorio incluye un chatbot RAG de consola que corre completamente offline.

Requisitos
----------
1. Python 3.9+
2. Instala dependencias (recomendado `llama-cpp-python >= 0.3.5` para soporte Gemma 3):
   ```bash
   pip install "llama-cpp-python>=0.3.2" faiss-cpu sentence-transformers
   ```
   En macOS puedes usar conda si el build falla:
   ```bash
   conda install -c conda-forge llama-cpp-python
   pip install faiss-cpu sentence-transformers
   ```
3. Descarga un modelo GGUF y guárdalo en `../models/`:
   - **Gemma 3 1B Instruct (recomendado)**
     ```bash
     wget https://huggingface.co/google/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_K_M.gguf \
       -O gemma-3-1b-it.Q4_K_M.gguf
     ```
   - **Mistral-7B-Instruct**
     ```bash
     wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf \
       -O mistral-7b-instruct.Q4_K_M.gguf
     ```

   > ℹ️ Hugging Face puede exigir iniciar sesión y aceptar la licencia antes de habilitar la descarga. Si aparece error 403, entra al repositorio, acepta los términos y vuelve a ejecutar el comando.

   Verifica la integridad comparando el `sha256` con el publicado por el proveedor:
   ```bash
   sha256 gemma-3-1b-it.Q4_K_M.gguf
   sha256 mistral-7b-instruct.Q4_K_M.gguf
   ```

   - Gemma 3: licencia Gemma → https://ai.google.dev/gemma
   - Mistral 7B: Apache 2.0 → https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1

   Pasos recomendados antes de descargar Gemma:
   1. Inicia sesión en Hugging Face y **acepta la licencia** en la pestaña *Files and versions* de `google/gemma-3-1b-it-GGUF`.
   2. Haz login en la CLI: `huggingface-cli login` (o token de acceso).
   3. Ejecuta los comandos `wget`/`huggingface-cli download` anteriores.
   4. Confirma que tienes `llama-cpp-python >= 0.3.5`; versiones previas lanzan “unknown model architecture: 'gemma3'”.

   Backend `transformers` (opcional):
   ```bash
   huggingface-cli download google/gemma-3-1b-it \
     --local-dir ../models/gemma-3-1b-it-transformers \
     --local-dir-use-symlinks False
   ```
   > Requiere aceptar la licencia y autenticarse con `huggingface-cli login`.

4. Construye tu dataset `qa_dataset.json`:
   ```json
   [
     {
       "pregunta": "¿Cuál es el horario de atención?",
       "respuesta": "Nuestro horario es de lunes a viernes de 9 a 18 y sábados de 9 a 14."
     },
     {
       "pregunta": "¿Cómo contacto a soporte?",
       "respuesta": "Puedes escribir a soporte@empresa.com o llamar al 900-123-456."
     }
   ]
   ```
5. Configura `config.yaml`:
   ```yaml
   models:
     embeddings:
       model_name: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
     generation:
       llama_cpp_model_path: "../models/gemma-3-1b-it.Q4_K_M.gguf"
       max_tokens: 256
   ```

Estructura del proyecto
-----------------------
- `prepare_embeddings.py` → genera `dataset_index.faiss` y `qa.json`
- `chatbot_rag_local.py` → chatbot de consola con llama-cpp
- `chatbot_rag_local_transformers.py` → alternativa basada en transformers
- `qa_dataset.json` → base de conocimiento

Uso
----
1. `python prepare_embeddings.py`
2. `python chatbot_rag_local.py`
3. Chatea en español con tu base de conocimiento :)

Docker
------
Construye la imagen desde la raíz del repositorio:
```bash
docker build -t llm-rag-assistant .
```

Ejecuta el bot de consola montando tu carpeta de modelos GGUF para que el contenedor resuelva `../models/...`:
```bash
docker run --rm -it \
  -v $(pwd)/../models:/models \
  llm-rag-assistant
```

También puedes sobreescribir el comando por defecto para lanzar otros flujos, por ejemplo:
```bash
docker run --rm -it \
  -v $(pwd)/../models:/models \
  llm-rag-assistant \
  python model_evaluation.py bertscore
```

Alternativa sin llama.cpp (transformers)
---------------------------------------
1. `pip install torch transformers accelerate`
2. Descarga pesos HF en `../models/gemma-3-1b-it-transformers`
3. Ajusta sección `transformers` en `config.yaml`
4. `python chatbot_rag_local_transformers.py`

> Requiere RAM suficiente (≥12 GB recomendado) o GPU.

Modelos en `../models`
----------------------
| Modelo | Ventajas | Consideraciones |
|--------|----------|----------------|
| `gemma-3-1b-it.Q4_K_M.gguf` | ✅ 1 B parámetros, cuantización Q4_K_M (~2.1 GB). Arranque rápido en CPU/MPS.<br>✅ Afinado para español/ciencias; baja alucinación.<br>✅ Excelente en Apple Silicon y CPUs con AVX2. | ℹ️ Necesita aceptar licencia Gemma y `llama-cpp-python` ≥0.3.2.<br>ℹ️ Al ser más pequeño, conviene usar prompts claros para respuestas largas. |
| `mistral-7b-instruct.Q4_K_M.gguf` | ✅ 7 B parámetros, robusto con prompts genéricos.<br>✅ Amplia adopción en la comunidad. | ⚠️ ~4.1 GB en disco, más lento en CPU pura.<br>⚠️ Mayor uso de RAM (~7–8 GB con contexto largo). |
| `qwen2.5-1.5b-instruct-q2_k.gguf` | ✅ Ultra liviano (<1 GB) ideal para hardware limitado.<br>✅ Buen soporte multilingüe. | ⚠️ Cuantización Q2 → menor fidelidad semántica.<br>⚠️ Requiere prompts estructurados. |

**Recomendación**: Gemma 3 1B Instruct es el balance ideal: rápido, preciso en español y amigable con los recursos. Mistral queda como respaldo si necesitas respuestas más extensas y tienes RAM extra.

Evaluación y métricas
=====================

Scripts disponibles
-------------------
- `model_evaluation.py` → genera respuestas o calcula BERTScore (`python model_evaluation.py bertscore`)
- `calculate_metrics_from_json.py` → recalcula BERTScore, ROUGE, BLEU y similitud coseno desde un JSON existente
- `real_rag_evaluation.py` → evalúa el pipeline RAG en vivo

Ejemplo de uso
--------------
```bash
# 1. Generar respuestas
python model_evaluation.py

# 2. Calcular BERTScore sobre resultados existentes
python model_evaluation.py bertscore
```

Salidas:
- `evaluation_results.json` – pregunta/respuesta dataset vs. respuesta generada
- `bertscore_results.json` – estadísticas y métricas por muestra

Interpretación
--------------
- **BERTScore F1**
  - > 0.85 → Excelente
  - 0.70–0.85 → Bueno
  - 0.50–0.70 → Mejorable
  - < 0.50 → Problemático
- **ROUGE-1/2/L** → Coincidencia de unigramas/bigramas/secuencias largas
- **BLEU-4** → Precisión de 4-gramas (0.2–0.6 típico en lenguaje natural)

Parámetros ajustables (en los scripts)
--------------------------------------
```python
n_samples = 15
random.seed(42)
lang = "es"
```

Ejemplo de salida
-----------------
```
📊 BERTScore:
   Precision: 0.7724 ± 0.0879
   Recall:    0.8905 ± 0.0591
   F1-Score:  0.8265 ± 0.0732

📝 ROUGE:
   ROUGE-1:   0.5064 ± 0.2007
   ROUGE-2:   0.4026 ± 0.2220
   ROUGE-L:   0.4760 ± 0.2138
```

Recomendaciones de hardware
---------------------------
- Mínimo 8 GB RAM (16 GB ideal)
- ~5 GB libres para modelos + índices
