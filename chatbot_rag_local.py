from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import faiss, json, numpy as np, yaml
from pathlib import Path

index = faiss.read_index("dataset_index.faiss")
with open("qa.json", "r", encoding="utf-8") as f:
    db = json.load(f)

with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)
embedder_model_name = config['models']['embeddings']['model_name']
llama_model_path = config['models']['generation']['llama_cpp_model_path']
max_tokens = config['models']['generation']['max_tokens']

embedder = SentenceTransformer(embedder_model_name)
llm = Llama(model_path=llama_model_path, n_ctx=2048)

def buscar_contexto(pregunta):
    emb = embedder.encode([pregunta])
    _, I = index.search(np.array(emb).astype(np.float32), 1)
    return db["questions"][I[0][0]], db["answers"][I[0][0]]

print(
    "Modelos cargados → "
    f"Embeddings: {embedder_model_name} | "
    f"LLM: {Path(llama_model_path).name if llama_model_path else llama_model_path}"
)
print("""🤖 Bienvenido al chatbot con RAG local 
Escribí tu pregunta (o 'salir'):
""")
while True:
    user_input = input("Tú: ")
    if user_input.lower() in ["salir", "exit"]: break
    pregunta_similar, respuesta_contexto = buscar_contexto(user_input)

    prompt = (
        "Responde en español usando únicamente la información del contexto.\n"
        "No repitas estas instrucciones ni el contexto; si falta información, dilo y pide más datos.\n\n"
        f"Contexto:\n- Pregunta base: {pregunta_similar}\n- Respuesta asociada: {respuesta_contexto}\n\n"
        f"Pregunta: {user_input}\n"
        "Respuesta:"
    )

    output = llm(prompt, max_tokens=max_tokens)
    print("\nBot:", output["choices"][0]["text"].strip(), "\n")
