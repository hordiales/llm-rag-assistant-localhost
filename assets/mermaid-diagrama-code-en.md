graph TD
  A["JSON File (Local Dataset)"] --> B["Embeddings Model (sentence-transformers)"]
  B --> C["Vector Embeddings"]
  C --> D["Vector Index (FAISS)"]
  E["User Query"] --> D
  D --> F["Retrieved Context (Top-k)"]
  F --> G["Local Quantized LLM (llama-cpp + GGUF)"]
  G --> H["Generated Response"]

  style A fill:#f9f,stroke:#333,stroke-width:1px
  style B fill:#bbf,stroke:#333,stroke-width:1px
  style D fill:#bfb,stroke:#333,stroke-width:1px
  style G fill:#fff,stroke:#333,stroke-width:1px

