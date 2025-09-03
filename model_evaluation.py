#!/usr/bin/env python3
"""
BERTScore Evaluation Script for RAG System
Compares dataset responses with model-generated responses
"""

# por default genera respuestas con el sistema RAG
# para calcular BERTScore, ejecutar: python bertscore_evaluation.py bertscore

import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import yaml
from llama_cpp import Llama
import sys
import os

def load_dataset():
    """Load the Q&A dataset"""
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_config():
    """Load system configuration"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_rag_system():
    """Initialize the RAG system components"""
    print("Inicializando sistema RAG...")
    
    # Load config
    config = load_config()
    embedder_model_name = config['models']['embeddings']['model_name']
    llama_model_path = config['models']['generation']['llama_cpp_model_path']
    max_tokens = config['models']['generation']['max_tokens']
    
    # Load embedder
    embedder = SentenceTransformer(embedder_model_name)
    
    # Load FAISS index
    index = faiss.read_index("dataset_index.faiss")
    
    # Load Q&A database
    with open("qa.json", "r", encoding="utf-8") as f:
        db = json.load(f)
    
    # Load LLM
    llm = Llama(model_path=llama_model_path, n_ctx=2048)
    
    return embedder, index, db, llm, max_tokens

def buscar_contexto(pregunta, embedder, index, db):
    """Search for similar context in the dataset"""
    emb = embedder.encode([pregunta])
    _, I = index.search(np.array(emb).astype(np.float32), 1)
    return db["questions"][I[0][0]], db["answers"][I[0][0]]

def generate_response(user_input, embedder, index, db, llm, max_tokens):
    """Generate response using the RAG system"""
    pregunta_similar, respuesta_contexto = buscar_contexto(user_input, embedder, index, db)
    
    prompt = f"""[INST] Eres un asistente. Un usuario pregunta: "{user_input}".
Basándote en este conocimiento previo:
Pregunta previa: "{pregunta_similar}"
Respuesta: "{respuesta_contexto}"
Responde en español de forma clara y precisa. [/INST]"""
    
    output = llm(prompt, max_tokens=max_tokens)
    return output["choices"][0]["text"].strip()

def select_random_samples(dataset, n_samples=10):
    """Select random samples from the dataset"""
    random.seed(42)  # For reproducibility
    samples = random.sample(dataset, min(n_samples, len(dataset)))
    return samples

def main():
    print("=== Evaluación BERTScore del Sistema RAG ===\n")
    
    # Check if required files exist
    required_files = ["dataset_index.faiss", "qa.json", "config.yaml"]
    for file in required_files:
        if not os.path.exists(file):
            print(f"Error: Archivo requerido '{file}' no encontrado.")
            print("Por favor ejecuta 'python prepare_embeddings.py' primero.")
            sys.exit(1)
    
    # Load dataset
    print("Cargando dataset...")
    dataset = load_dataset()
    
    # Select random samples
    print("Seleccionando muestras aleatorias...")
    samples = select_random_samples(dataset, n_samples=10)
    
    print(f"Seleccionadas {len(samples)} muestras para evaluación:\n")
    for i, sample in enumerate(samples, 1):
        print(f"{i}. {sample['pregunta']}")
    print()
    
    # Setup RAG system
    try:
        embedder, index, db, llm, max_tokens = setup_rag_system()
    except Exception as e:
        print(f"Error inicializando sistema RAG: {e}")
        sys.exit(1)
    
    # Generate responses
    print("Generando respuestas con el modelo...")
    results = []
    
    for i, sample in enumerate(samples, 1):
        question = sample['pregunta']
        reference_answer = sample['respuesta']
        
        print(f"Procesando {i}/{len(samples)}: {question[:50]}...")
        
        try:
            generated_answer = generate_response(question, embedder, index, db, llm, max_tokens)
            
            results.append({
                'pregunta': question,
                'respuesta_dataset': reference_answer,
                'respuesta_generada': generated_answer
            })
            
        except Exception as e:
            print(f"Error generando respuesta para pregunta {i}: {e}")
            results.append({
                'pregunta': question,
                'respuesta_dataset': reference_answer,
                'respuesta_generada': f"ERROR: {str(e)}"
            })
    
    # Save results for BERTScore calculation
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nResultados guardados en 'evaluation_results.json'")
    print("Para calcular BERTScore, instala: pip install bert-score")
    print("Luego ejecuta la función de cálculo de BERTScore: python model_evaluation.py bertscore")
    print("Para resultados generales, ejecuta: python comprehensive_evaluation.py")
    
    # Display sample results
    print("\n=== Muestra de Resultados ===")
    for i, result in enumerate(results[:3], 1):
        print(f"\n--- Ejemplo {i} ---")
        print(f"Pregunta: {result['pregunta']}")
        print(f"Dataset: {result['respuesta_dataset']}")
        print(f"Generada: {result['respuesta_generada']}")

def calculate_bertscore():
    """Calculate BERTScore metrics"""
    try:
        from bert_score import score
    except ImportError:
        print("Error: bert-score no está instalado.")
        print("Instala con: pip install bert-score")
        return
    
    # Load results
    with open("evaluation_results.json", "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # Extract responses
    references = [r['respuesta_dataset'] for r in results if not r['respuesta_generada'].startswith('ERROR')]
    candidates = [r['respuesta_generada'] for r in results if not r['respuesta_generada'].startswith('ERROR')]
    
    if not references or not candidates:
        print("No hay respuestas válidas para evaluar.")
        return
    
    print(f"Calculando BERTScore para {len(references)} muestras...")
    
    # Calculate BERTScore
    P, R, F1 = score(candidates, references, lang="es", verbose=True)
    
    # Calculate statistics
    stats = {
        'precision': {
            'mean': P.mean().item(),
            'std': P.std().item(),
            'min': P.min().item(),
            'max': P.max().item()
        },
        'recall': {
            'mean': R.mean().item(),
            'std': R.std().item(),
            'min': R.min().item(),
            'max': R.max().item()
        },
        'f1': {
            'mean': F1.mean().item(),
            'std': F1.std().item(),
            'min': F1.min().item(),
            'max': F1.max().item()
        }
    }
    
    # Save detailed results
    detailed_results = []
    for i, (result, p, r, f1) in enumerate(zip(results, P.tolist(), R.tolist(), F1.tolist())):
        if not result['respuesta_generada'].startswith('ERROR'):
            detailed_results.append({
                **result,
                'bertscore_precision': p,
                'bertscore_recall': r,
                'bertscore_f1': f1
            })
    
    with open("bertscore_results.json", "w", encoding="utf-8") as f:
        json.dump({
            'statistics': stats,
            'detailed_results': detailed_results
        }, f, ensure_ascii=False, indent=2)
    
    print("\n=== Resultados BERTScore ===")
    print(f"Precision: {stats['precision']['mean']:.4f} ± {stats['precision']['std']:.4f}")
    print(f"Recall:    {stats['recall']['mean']:.4f} ± {stats['recall']['std']:.4f}")
    print(f"F1-Score:  {stats['f1']['mean']:.4f} ± {stats['f1']['std']:.4f}")
    
    print(f"\nResultados detallados guardados en 'bertscore_results.json'")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bertscore":
        calculate_bertscore()
    else:
        main()