graph TD
  A["Archivo JSON (Dataset local)"] --> B["Modelo de Embeddings (sentence-transformers)"]
  B --> C["Embeddings vectoriales"]
  C --> D["Índice Vectorial (FAISS)"]
  E["Consulta del Usuario"] --> D
  D --> F["Contexto Recuperado (Top-k)"]
  F --> G["LLM local cuantizado (llama-cpp + GGUF)"]
  G --> H["Respuesta Generada"]

  style A fill:#f9f,stroke:#333,stroke-width:1px
  style B fill:#bbf,stroke:#333,stroke-width:1px
  style D fill:#bfb,stroke:#333,stroke-width:1px
  style G fill:#ffd,stroke:#333,stroke-width:1px

