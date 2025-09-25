#!/usr/bin/env python3
"""
Comparative RAG System Evaluation Script
Evaluates 3 different models (Gemma 3, Mistral 7B, Qwen 2.5) with BERTScore, ROUGE, BLEU, and Cosine Similarity metrics
"""

import json
import random
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import yaml
import sys
import os
from typing import List, Dict, Tuple
from llama_cpp import Llama
from pathlib import Path
import time
from datetime import datetime

# Model configurations
MODELS = {
    "gemma-3-1b": {
        "path": "../models/gemma-3-1b-it-q4_k_m.gguf",
        "name": "Gemma 3 1B Instruct",
        "short_name": "Gemma3"
    },
    "mistral-7b": {
        "path": "../models/mistral-7b-instruct.Q4_K_M.gguf",
        "name": "Mistral 7B Instruct",
        "short_name": "Mistral7B"
    },
    "qwen-2.5-1.5b": {
        "path": "../models/qwen2.5-1.5b-instruct-q2_k.gguf",
        "name": "Qwen 2.5 1.5B Instruct",
        "short_name": "Qwen2.5"
    }
}

def load_config():
    """Load system configuration"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_rag_components():
    """Initialize RAG components (embedder, index, database)"""
    print("🔧 Inicializando componentes RAG...")

    config = load_config()
    embedder_model_name = config['models']['embeddings']['model_name']

    print(f"📚 Cargando modelo de embeddings: {embedder_model_name}")
    embedder = SentenceTransformer(embedder_model_name)

    print(f"🔍 Cargando índice FAISS...")
    index = faiss.read_index("dataset_index.faiss")

    print(f"📖 Cargando base de datos Q&A...")
    with open("qa.json", "r", encoding="utf-8") as f:
        db = json.load(f)

    return embedder, index, db

def load_model(model_path: str, model_name: str) -> Llama:
    """Load a specific LLM model"""
    print(f"🧠 Cargando {model_name}: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ Modelo no encontrado: {model_path}")
        return None

    try:
        llm = Llama(model_path=model_path, n_ctx=2048, verbose=False)
        print(f"✅ {model_name} cargado correctamente")
        return llm
    except Exception as e:
        print(f"❌ Error cargando {model_name}: {e}")
        return None

def buscar_contexto(pregunta, embedder, index, db):
    """Search for similar context in the dataset"""
    emb = embedder.encode([pregunta])
    _, I = index.search(np.array(emb).astype(np.float32), 1)
    return db["questions"][I[0][0]], db["answers"][I[0][0]]

def generate_response(user_input: str, llm: Llama, embedder, index, db, max_tokens: int = 256) -> str:
    """Generate response using RAG system with specific model"""
    if llm is None:
        return "[ERROR] Modelo no disponible"

    try:
        pregunta_similar, respuesta_contexto = buscar_contexto(user_input, embedder, index, db)

        prompt = (
            "Responde en español usando únicamente la información del contexto.\n"
            "No repitas estas instrucciones ni el contexto; si falta información, dilo y pide más datos.\n\n"
            f"Contexto:\n- Pregunta base: {pregunta_similar}\n- Respuesta asociada: {respuesta_contexto}\n\n"
            f"Pregunta: {user_input}\n"
            "Respuesta:"
        )

        output = llm(prompt, max_tokens=max_tokens)
        return output["choices"][0]["text"].strip()

    except Exception as e:
        return f"[ERROR] {str(e)}"

def load_test_dataset():
    """Load test questions from qa.json"""
    print("📚 Cargando dataset de prueba...")

    # Try to load the dataset
    if os.path.exists("qa_dataset.json"):
        with open("qa_dataset.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        # Use qa.json as fallback
        with open("qa.json", "r", encoding="utf-8") as f:
            db = json.load(f)
            dataset = [
                {"pregunta": q, "respuesta": a}
                for q, a in zip(db["questions"], db["answers"])
            ]

    print(f"   Dataset: {len(dataset)} muestras disponibles")
    return dataset

def select_test_samples(dataset, n_samples=20):
    """Select random samples for evaluation"""
    random.seed(42)  # For reproducibility
    samples = random.sample(dataset, min(n_samples, len(dataset)))
    print(f"🎲 Seleccionadas {len(samples)} muestras para evaluación")
    return samples

def evaluate_models(samples: List[Dict], embedder, index, db) -> Dict:
    """Evaluate all models on the test samples"""
    results = {
        "evaluation_info": {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(samples),
            "models_tested": list(MODELS.keys())
        },
        "model_results": {},
        "detailed_responses": []
    }

    # Load all models
    loaded_models = {}
    for model_id, model_info in MODELS.items():
        model = load_model(model_info["path"], model_info["name"])
        loaded_models[model_id] = model

        if model is None:
            print(f"⚠️  Saltando evaluación de {model_info['name']} - modelo no disponible")

    available_models = {k: v for k, v in loaded_models.items() if v is not None}

    if not available_models:
        print("❌ Ningún modelo disponible para evaluación")
        return results

    print(f"\n🚀 Evaluando {len(available_models)} modelos en {len(samples)} muestras...")

    # Generate responses for each sample with each model
    for i, sample in enumerate(samples, 1):
        question = sample['pregunta']
        reference = sample['respuesta']

        print(f"\n[{i:2d}/{len(samples)}] Procesando: {question[:60]}...")

        sample_results = {
            'id': i-1,
            'question': question,
            'reference_answer': reference,
            'model_responses': {}
        }

        # Get response from each available model
        for model_id, llm in available_models.items():
            model_name = MODELS[model_id]["short_name"]
            print(f"  → {model_name}...", end=" ")

            start_time = time.time()
            response = generate_response(question, llm, embedder, index, db)
            duration = time.time() - start_time

            sample_results['model_responses'][model_id] = {
                'response': response,
                'duration_seconds': duration,
                'status': 'success' if not response.startswith('[ERROR]') else 'error'
            }

            print(f"({duration:.1f}s)")

        results["detailed_responses"].append(sample_results)

    # Calculate metrics for each model
    for model_id in available_models.keys():
        print(f"\n📊 Calculando métricas para {MODELS[model_id]['name']}...")

        # Extract successful responses
        model_responses = []
        references = []

        for sample in results["detailed_responses"]:
            if model_id in sample['model_responses']:
                response_data = sample['model_responses'][model_id]
                if response_data['status'] == 'success':
                    model_responses.append(response_data['response'])
                    references.append(sample['reference_answer'])

        if not model_responses:
            print(f"❌ No hay respuestas válidas para {MODELS[model_id]['name']}")
            continue

        # Calculate metrics
        metrics = calculate_all_metrics(references, model_responses, model_id)
        results["model_results"][model_id] = {
            "model_info": MODELS[model_id],
            "successful_responses": len(model_responses),
            "metrics": metrics
        }

    return results

def calculate_all_metrics(references: List[str], candidates: List[str], model_name: str) -> Dict:
    """Calculate all evaluation metrics"""
    metrics = {}

    print(f"  🔍 BERTScore...", end=" ")
    try:
        bert_metrics = calculate_bertscore_metrics(references, candidates)
        metrics.update(bert_metrics)
        print("✓")
    except Exception as e:
        print(f"✗ ({e})")

    print(f"  📋 ROUGE...", end=" ")
    try:
        rouge_metrics = calculate_rouge_metrics(references, candidates)
        metrics.update(rouge_metrics)
        print("✓")
    except Exception as e:
        print(f"✗ ({e})")

    print(f"  🔤 BLEU...", end=" ")
    try:
        bleu_metrics = calculate_bleu_metrics(references, candidates)
        metrics.update(bleu_metrics)
        print("✓")
    except Exception as e:
        print(f"✗ ({e})")

    print(f"  🎯 Cosine Similarity...", end=" ")
    try:
        sim_metrics = calculate_similarity_metrics(references, candidates)
        metrics.update(sim_metrics)
        print("✓")
    except Exception as e:
        print(f"✗ ({e})")

    return metrics

def calculate_bertscore_metrics(references: List[str], candidates: List[str]) -> Dict:
    """Calculate BERTScore metrics"""
    try:
        from bert_score import score
    except ImportError:
        os.system("pip install bert-score")
        from bert_score import score

    P, R, F1 = score(candidates, references, lang="es", verbose=False)

    return {
        'bertscore': {
            'precision': {
                'mean': float(P.mean().item()),
                'std': float(P.std().item()),
                'scores': [float(x) for x in P.tolist()]
            },
            'recall': {
                'mean': float(R.mean().item()),
                'std': float(R.std().item()),
                'scores': [float(x) for x in R.tolist()]
            },
            'f1': {
                'mean': float(F1.mean().item()),
                'std': float(F1.std().item()),
                'scores': [float(x) for x in F1.tolist()]
            }
        }
    }

def calculate_rouge_metrics(references: List[str], candidates: List[str]) -> Dict:
    """Calculate ROUGE metrics"""
    try:
        from rouge_score import rouge_scorer
    except ImportError:
        os.system("pip install rouge-score")
        from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for ref, cand in zip(references, candidates):
        scores = scorer.score(ref, cand)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    return {
        'rouge': {
            'rouge1': {
                'mean': float(np.mean(rouge1_scores)),
                'std': float(np.std(rouge1_scores)),
                'scores': [float(x) for x in rouge1_scores]
            },
            'rouge2': {
                'mean': float(np.mean(rouge2_scores)),
                'std': float(np.std(rouge2_scores)),
                'scores': [float(x) for x in rouge2_scores]
            },
            'rougeL': {
                'mean': float(np.mean(rougeL_scores)),
                'std': float(np.std(rougeL_scores)),
                'scores': [float(x) for x in rougeL_scores]
            }
        }
    }

def calculate_bleu_metrics(references: List[str], candidates: List[str]) -> Dict:
    """Calculate BLEU metrics"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import word_tokenize
    except ImportError:
        os.system("pip install nltk")
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import word_tokenize

    smoothing = SmoothingFunction()

    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []

    for ref, cand in zip(references, candidates):
        ref_tokens = word_tokenize(ref.lower())
        cand_tokens = word_tokenize(cand.lower())

        bleu1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0),
                             smoothing_function=smoothing.method1)
        bleu2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0),
                             smoothing_function=smoothing.method1)
        bleu3 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.33, 0.33, 0.33, 0),
                             smoothing_function=smoothing.method1)
        bleu4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                             smoothing_function=smoothing.method1)

        bleu1_scores.append(bleu1)
        bleu2_scores.append(bleu2)
        bleu3_scores.append(bleu3)
        bleu4_scores.append(bleu4)

    return {
        'bleu': {
            'bleu1': {
                'mean': float(np.mean(bleu1_scores)),
                'std': float(np.std(bleu1_scores)),
                'scores': [float(x) for x in bleu1_scores]
            },
            'bleu2': {
                'mean': float(np.mean(bleu2_scores)),
                'std': float(np.std(bleu2_scores)),
                'scores': [float(x) for x in bleu2_scores]
            },
            'bleu3': {
                'mean': float(np.mean(bleu3_scores)),
                'std': float(np.std(bleu3_scores)),
                'scores': [float(x) for x in bleu3_scores]
            },
            'bleu4': {
                'mean': float(np.mean(bleu4_scores)),
                'std': float(np.std(bleu4_scores)),
                'scores': [float(x) for x in bleu4_scores]
            }
        }
    }

def calculate_similarity_metrics(references: List[str], candidates: List[str]) -> Dict:
    """Calculate cosine similarity metrics"""
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Get embeddings
    ref_embeddings = embedder.encode(references)
    cand_embeddings = embedder.encode(candidates)

    # Calculate cosine similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = []
    for ref_emb, cand_emb in zip(ref_embeddings, cand_embeddings):
        sim = cosine_similarity([ref_emb], [cand_emb])[0][0]
        similarities.append(sim)

    return {
        'similarity': {
            'cosine': {
                'mean': float(np.mean(similarities)),
                'std': float(np.std(similarities)),
                'scores': [float(x) for x in similarities]
            }
        }
    }

def create_comparison_report(results: Dict) -> None:
    """Create and save comparison report"""
    print("\n" + "="*80)
    print("📊 REPORTE COMPARATIVO DE MODELOS RAG")
    print("="*80)

    # Create comparison table
    comparison_data = []

    for model_id, model_result in results["model_results"].items():
        model_info = model_result["model_info"]
        metrics = model_result["metrics"]

        row = {
            'Modelo': model_info["short_name"],
            'Nombre Completo': model_info["name"],
            'Respuestas Exitosas': model_result["successful_responses"]
        }

        # Add metric means
        if 'bertscore' in metrics:
            row['BERTScore F1'] = f"{metrics['bertscore']['f1']['mean']:.4f}"
            row['BERTScore Precision'] = f"{metrics['bertscore']['precision']['mean']:.4f}"
            row['BERTScore Recall'] = f"{metrics['bertscore']['recall']['mean']:.4f}"

        if 'rouge' in metrics:
            row['ROUGE-1'] = f"{metrics['rouge']['rouge1']['mean']:.4f}"
            row['ROUGE-2'] = f"{metrics['rouge']['rouge2']['mean']:.4f}"
            row['ROUGE-L'] = f"{metrics['rouge']['rougeL']['mean']:.4f}"

        if 'bleu' in metrics:
            row['BLEU-1'] = f"{metrics['bleu']['bleu1']['mean']:.4f}"
            row['BLEU-2'] = f"{metrics['bleu']['bleu2']['mean']:.4f}"
            row['BLEU-4'] = f"{metrics['bleu']['bleu4']['mean']:.4f}"

        if 'similarity' in metrics:
            row['Cosine Similarity'] = f"{metrics['similarity']['cosine']['mean']:.4f}"

        comparison_data.append(row)

    # Create DataFrame and display
    df = pd.DataFrame(comparison_data)
    print("\n📋 TABLA COMPARATIVA:")
    print(df.to_string(index=False))

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"comparative_rag_evaluation_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Save CSV summary
    csv_file = f"comparative_rag_summary_{timestamp}.csv"
    df.to_csv(csv_file, index=False, encoding='utf-8')

    print(f"\n✅ Resultados detallados guardados en: {output_file}")
    print(f"✅ Resumen CSV guardado en: {csv_file}")

    # Show best performing model for each metric
    print(f"\n🏆 MEJORES MODELOS POR MÉTRICA:")

    metric_columns = [col for col in df.columns if col not in ['Modelo', 'Nombre Completo', 'Respuestas Exitosas']]

    for metric in metric_columns:
        if metric in df.columns:
            # Convert to numeric for comparison
            df_numeric = df.copy()
            df_numeric[metric] = pd.to_numeric(df_numeric[metric], errors='coerce')
            best_idx = df_numeric[metric].idxmax()
            best_model = df.loc[best_idx, 'Modelo']
            best_score = df.loc[best_idx, metric]
            print(f"   {metric:20}: {best_model} ({best_score})")

def main():
    print("=== 🚀 EVALUACIÓN COMPARATIVA DE MODELOS RAG ===")
    print("Modelos: Gemma 3 1B, Mistral 7B, Qwen 2.5 1.5B")
    print("Métricas: BERTScore, ROUGE, BLEU, Cosine Similarity\n")

    # Check required files
    required_files = ["dataset_index.faiss", "qa.json", "config.yaml"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("❌ Archivos requeridos faltantes:")
        for file in missing_files:
            print(f"   - {file}")
        sys.exit(1)

    # Setup RAG components
    try:
        embedder, index, db = setup_rag_components()
    except Exception as e:
        print(f"❌ Error configurando componentes RAG: {e}")
        sys.exit(1)

    # Load test dataset
    try:
        dataset = load_test_dataset()
        samples = select_test_samples(dataset, n_samples=20)
    except Exception as e:
        print(f"❌ Error cargando dataset: {e}")
        sys.exit(1)

    # Run comparative evaluation
    print(f"\n🎯 Iniciando evaluación comparativa...")
    results = evaluate_models(samples, embedder, index, db)

    if not results["model_results"]:
        print("❌ No se pudieron evaluar modelos")
        sys.exit(1)

    # Generate comparison report
    create_comparison_report(results)

    print(f"\n🎉 Evaluación comparativa completada!")

if __name__ == "__main__":
    main()