import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

with open("qa_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
questions = [item["pregunta"] for item in data]
answers = [item["respuesta"] for item in data]
embeddings = embedder.encode(questions, convert_to_numpy=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, "dataset_index.faiss")
with open("qa.json", "w", encoding="utf-8") as f:
    json.dump({"questions": questions, "answers": answers}, f, ensure_ascii=False, indent=2)
