#!/usr/bin/env python3
"""
Real RAG System Evaluation Script
Evaluates actual RAG responses using BERTScore, ROUGE, BLEU, and Cosine Similarity metrics
"""

import json
import random
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import yaml
import sys
import os
from typing import List, Dict
from llama_cpp import Llama

def load_dataset():
    """Load the Q&A dataset"""
    with open("qa_dataset.json", "r", encoding="utf-8") as f:
        return json.load(f)

def load_config():
    """Load system configuration"""
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_rag_system():
    """Initialize the real RAG system components"""
    print("🔧 Inicializando sistema RAG real...")
    
    # Load config
    config = load_config()
    embedder_model_name = config['models']['embeddings']['model_name']
    llama_model_path = config['models']['generation']['llama_cpp_model_path']
    max_tokens = config['models']['generation']['max_tokens']
    
    print(f"📚 Cargando modelo de embeddings: {embedder_model_name}")
    embedder = SentenceTransformer(embedder_model_name)
    
    print(f"🔍 Cargando índice FAISS...")
    index = faiss.read_index("dataset_index.faiss")
    
    print(f"📖 Cargando base de datos Q&A...")
    with open("qa.json", "r", encoding="utf-8") as f:
        db = json.load(f)
    
    print(f"🧠 Cargando modelo LLM: {llama_model_path}")
    try:
        llm = Llama(model_path=llama_model_path, n_ctx=2048, verbose=False)
        print("✅ Sistema RAG inicializado correctamente")
    except Exception as e:
        print(f"❌ Error cargando LLM: {e}")
        print("💡 Nota: Para evaluación real, asegúrate de que el modelo esté disponible")
        return embedder, index, db, None, max_tokens
    
    return embedder, index, db, llm, max_tokens

def buscar_contexto(pregunta, embedder, index, db):
    """Search for similar context in the dataset"""
    emb = embedder.encode([pregunta])
    _, I = index.search(np.array(emb).astype(np.float32), 1)
    return db["questions"][I[0][0]], db["answers"][I[0][0]]

def generate_real_response(user_input, embedder, index, db, llm, max_tokens):
    """Generate response using the real RAG system"""
    if llm is None:
        # Fallback para cuando el LLM no está disponible
        print(f"⚠️  LLM no disponible, usando respuesta de fallback para: {user_input[:50]}...")
        _, respuesta_contexto = buscar_contexto(user_input, embedder, index, db)
        return f"[RESPUESTA SIMULADA] {respuesta_contexto}"
    
    # Usar el sistema RAG real
    pregunta_similar, respuesta_contexto = buscar_contexto(user_input, embedder, index, db)
    
    prompt = (
        "Responde en español usando únicamente la información del contexto.\n"
        "No repitas estas instrucciones ni el contexto; si falta información, dilo y pide más datos.\n\n"
        f"Contexto:\n- Pregunta base: {pregunta_similar}\n- Respuesta asociada: {respuesta_contexto}\n\n"
        f"Pregunta: {user_input}\n"
        "Respuesta:"
    )
    
    try:
        output = llm(prompt, max_tokens=max_tokens)
        return output["choices"][0]["text"].strip()
    except Exception as e:
        print(f"⚠️  Error generando respuesta para '{user_input[:30]}...': {e}")
        return f"[ERROR EN GENERACIÓN] {respuesta_contexto}"

def select_random_samples(dataset, n_samples=15):
    """Select random samples from the dataset"""
    random.seed(42)  # For reproducibility
    samples = random.sample(dataset, min(n_samples, len(dataset)))
    return samples

def generate_rag_responses(samples, embedder, index, db, llm, max_tokens):
    """Generate responses using the real RAG system"""
    results = []
    total_samples = len(samples)
    
    print(f"🤖 Generando respuestas reales del sistema RAG...")
    print(f"📊 Procesando {total_samples} muestras...")
    
    for i, sample in enumerate(samples, 1):
        question = sample['pregunta']
        reference_answer = sample['respuesta']
        
        print(f"[{i:2d}/{total_samples}] Procesando: {question[:60]}...")
        
        try:
            generated_answer = generate_real_response(question, embedder, index, db, llm, max_tokens)
            
            # Obtener el contexto usado
            pregunta_similar, contexto_usado = buscar_contexto(question, embedder, index, db)
            
            results.append({
                'id': i-1,
                'pregunta': question,
                'respuesta_dataset': reference_answer,
                'respuesta_generada': generated_answer,
                'contexto_usado': contexto_usado,
                'pregunta_similar': pregunta_similar,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"❌ Error procesando pregunta {i}: {e}")
            results.append({
                'id': i-1,
                'pregunta': question,
                'respuesta_dataset': reference_answer,
                'respuesta_generada': f"[ERROR] {str(e)}",
                'contexto_usado': "",
                'pregunta_similar': "",
                'status': 'error'
            })
    
    successful_results = [r for r in results if r['status'] == 'success']
    error_results = [r for r in results if r['status'] == 'error']
    
    print(f"✅ Completadas: {len(successful_results)}/{total_samples}")
    if error_results:
        print(f"❌ Errores: {len(error_results)}/{total_samples}")
    
    return results

def calculate_bertscore_metrics(references: List[str], candidates: List[str]) -> Dict:
    """Calculate BERTScore metrics"""
    try:
        from bert_score import score
    except ImportError:
        print("📦 Instalando bert-score...")
        os.system("pip install bert-score")
        from bert_score import score
    
    print("📊 Calculando BERTScore...")
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
        print("📦 Instalando rouge-score...")
        os.system("pip install rouge-score")
        from rouge_score import rouge_scorer
    
    print("📊 Calculando ROUGE...")
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
        print("📦 Instalando nltk...")
        os.system("pip install nltk")
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        import nltk
        nltk.download('punkt', quiet=True)
        from nltk.tokenize import word_tokenize
    
    print("📊 Calculando BLEU...")
    smoothing = SmoothingFunction()
    
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    
    for ref, cand in zip(references, candidates):
        ref_tokens = word_tokenize(ref.lower())
        cand_tokens = word_tokenize(cand.lower())
        
        # BLEU scores with different n-gram weights
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
    """Calculate additional similarity metrics"""
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    
    print("📊 Calculando métricas de similitud...")
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Get embeddings
    ref_embeddings = embedder.encode(references)
    cand_embeddings = embedder.encode(candidates)
    
    # Calculate cosine similarities
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

def print_results_summary(metrics: Dict, mode: str = "REAL"):
    """Print a summary of all metrics"""
    print("\n" + "="*60)
    print(f"EVALUACIÓN {mode} - RESUMEN DE MÉTRICAS")
    print("="*60)
    
    # BERTScore
    if 'bertscore' in metrics:
        bert = metrics['bertscore']
        print(f"\n📊 BERTScore:")
        print(f"   Precision: {bert['precision']['mean']:.4f} ± {bert['precision']['std']:.4f}")
        print(f"   Recall:    {bert['recall']['mean']:.4f} ± {bert['recall']['std']:.4f}")
        print(f"   F1-Score:  {bert['f1']['mean']:.4f} ± {bert['f1']['std']:.4f}")
    
    # ROUGE
    if 'rouge' in metrics:
        rouge = metrics['rouge']
        print(f"\n📋 ROUGE:")
        print(f"   ROUGE-1:   {rouge['rouge1']['mean']:.4f} ± {rouge['rouge1']['std']:.4f}")
        print(f"   ROUGE-2:   {rouge['rouge2']['mean']:.4f} ± {rouge['rouge2']['std']:.4f}")
        print(f"   ROUGE-L:   {rouge['rougeL']['mean']:.4f} ± {rouge['rougeL']['std']:.4f}")
    
    # BLEU
    if 'bleu' in metrics:
        bleu = metrics['bleu']
        print(f"\n🔤 BLEU:")
        print(f"   BLEU-1:    {bleu['bleu1']['mean']:.4f} ± {bleu['bleu1']['std']:.4f}")
        print(f"   BLEU-2:    {bleu['bleu2']['mean']:.4f} ± {bleu['bleu2']['std']:.4f}")
        print(f"   BLEU-3:    {bleu['bleu3']['mean']:.4f} ± {bleu['bleu3']['std']:.4f}")
        print(f"   BLEU-4:    {bleu['bleu4']['mean']:.4f} ± {bleu['bleu4']['std']:.4f}")
    
    # Similarity
    if 'similarity' in metrics:
        sim = metrics['similarity']
        print(f"\n🎯 Similitud Coseno:")
        print(f"   Coseno:    {sim['cosine']['mean']:.4f} ± {sim['cosine']['std']:.4f}")

def check_prerequisites():
    """Check if all required files exist"""
    required_files = [
        "qa_dataset.json",
        "dataset_index.faiss", 
        "qa.json",
        "config.yaml"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Archivos requeridos faltantes:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Para generar archivos faltantes:")
        if "dataset_index.faiss" in missing_files or "qa.json" in missing_files:
            print("   python prepare_embeddings.py")
        return False
    
    return True

def main():
    print("=== 🚀 EVALUACIÓN REAL DEL SISTEMA RAG ===")
    print("Métricas: BERTScore, ROUGE, BLEU, Similitud Coseno")
    print("Usando: Gemma 3 1B Instruct + RAG real\n")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ No se pueden ejecutar las evaluaciones sin los archivos requeridos")
        sys.exit(1)
    
    # Load dataset
    print("📚 Cargando dataset...")
    dataset = load_dataset()
    print(f"   Dataset: {len(dataset)} muestras disponibles")
    
    # Select samples
    print("\n🎲 Seleccionando muestras aleatorias...")
    samples = select_random_samples(dataset, n_samples=15)
    print(f"   Seleccionadas: {len(samples)} muestras")
    
    # Show selected questions
    print("\n📝 Preguntas seleccionadas:")
    for i, sample in enumerate(samples, 1):
        print(f"   {i:2d}. {sample['pregunta']}")
    
    # Setup RAG system
    try:
        embedder, index, db, llm, max_tokens = setup_rag_system()
        is_real_llm = llm is not None
        
        if not is_real_llm:
            print("\n⚠️  MODO FALLBACK: LLM no disponible, usando respuestas de contexto")
            mode = "FALLBACK"
        else:
            print(f"\n✅ MODO REAL: Usando LLM con max_tokens={max_tokens}")
            mode = "REAL"
            
    except Exception as e:
        print(f"\n❌ Error configurando sistema RAG: {e}")
        sys.exit(1)
    
    # Generate responses
    print("\n" + "="*50)
    results = generate_rag_responses(samples, embedder, index, db, llm, max_tokens)
    
    # Filter successful results for evaluation
    successful_results = [r for r in results if r['status'] == 'success' and not r['respuesta_generada'].startswith('[ERROR')]
    
    if not successful_results:
        print("\n❌ No hay respuestas válidas para evaluar")
        sys.exit(1)
    
    print(f"\n📊 Evaluando {len(successful_results)} respuestas exitosas...")
    
    # Extract text for evaluation
    references = [r['respuesta_dataset'] for r in successful_results]
    candidates = [r['respuesta_generada'] for r in successful_results]
    
    # Calculate all metrics
    all_metrics = {}
    
    print("\n🔄 Calculando métricas...")
    
    # BERTScore
    try:
        bert_metrics = calculate_bertscore_metrics(references, candidates)
        all_metrics.update(bert_metrics)
    except Exception as e:
        print(f"❌ Error calculando BERTScore: {e}")
    
    # ROUGE
    try:
        rouge_metrics = calculate_rouge_metrics(references, candidates)
        all_metrics.update(rouge_metrics)
    except Exception as e:
        print(f"❌ Error calculando ROUGE: {e}")
    
    # BLEU
    try:
        bleu_metrics = calculate_bleu_metrics(references, candidates)
        all_metrics.update(bleu_metrics)
    except Exception as e:
        print(f"❌ Error calculando BLEU: {e}")
    
    # Similarity
    try:
        sim_metrics = calculate_similarity_metrics(references, candidates)
        all_metrics.update(sim_metrics)
    except Exception as e:
        print(f"❌ Error calculando similitud: {e}")
    
    # Combine results with individual scores
    detailed_results = []
    for i, result in enumerate(successful_results):
        detailed_result = result.copy()
        
        # Add individual metric scores
        if 'bertscore' in all_metrics:
            detailed_result.update({
                'bertscore_precision': all_metrics['bertscore']['precision']['scores'][i],
                'bertscore_recall': all_metrics['bertscore']['recall']['scores'][i],
                'bertscore_f1': all_metrics['bertscore']['f1']['scores'][i]
            })
        
        if 'rouge' in all_metrics:
            detailed_result.update({
                'rouge1': all_metrics['rouge']['rouge1']['scores'][i],
                'rouge2': all_metrics['rouge']['rouge2']['scores'][i],
                'rougeL': all_metrics['rouge']['rougeL']['scores'][i]
            })
        
        if 'bleu' in all_metrics:
            detailed_result.update({
                'bleu1': all_metrics['bleu']['bleu1']['scores'][i],
                'bleu2': all_metrics['bleu']['bleu2']['scores'][i],
                'bleu3': all_metrics['bleu']['bleu3']['scores'][i],
                'bleu4': all_metrics['bleu']['bleu4']['scores'][i]
            })
        
        if 'similarity' in all_metrics:
            detailed_result.update({
                'cosine_similarity': all_metrics['similarity']['cosine']['scores'][i]
            })
        
        detailed_results.append(detailed_result)
    
    # Save results
    final_results = {
        'evaluation_info': {
            'mode': mode,
            'is_real_llm': is_real_llm,
            'total_samples': len(samples),
            'successful_samples': len(successful_results),
            'failed_samples': len(samples) - len(successful_results)
        },
        'summary_metrics': all_metrics,
        'detailed_results': detailed_results,
        'failed_results': [r for r in results if r['status'] == 'error']
    }
    
    output_file = f"real_rag_evaluation_results_{mode.lower()}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print_results_summary(all_metrics, mode)
    
    # Show examples
    print(f"\n📝 EJEMPLOS DE EVALUACIÓN ({mode}):")
    for i, result in enumerate(detailed_results[:3], 1):
        print(f"\n--- Ejemplo {i} ---")
        print(f"Pregunta:     {result['pregunta'][:70]}...")
        print(f"Dataset:      {result['respuesta_dataset'][:70]}...")
        print(f"RAG {mode}:     {result['respuesta_generada'][:70]}...")
        
        if 'bertscore_f1' in result:
            print(f"BERTScore F1: {result['bertscore_f1']:.3f}")
        if 'rouge1' in result:
            print(f"ROUGE-1:      {result['rouge1']:.3f}")
        if 'bleu4' in result:
            print(f"BLEU-4:       {result['bleu4']:.3f}")
        if 'cosine_similarity' in result:
            print(f"Coseno:       {result['cosine_similarity']:.3f}")
    
    print(f"\n✅ Resultados completos guardados en: {output_file}")
    
    if not is_real_llm:
        print(f"\n💡 NOTA: Esta evaluación usa respuestas de fallback.")
        print(f"   Para evaluación real, instala llama-cpp-python y el modelo Gemma 3 1B Instruct")
        print(f"   Comando: conda install -c conda-forge llama-cpp-python")

if __name__ == "__main__":
    main()
