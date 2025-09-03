#!/usr/bin/env python3
"""
Calculate comprehensive metrics from existing evaluation results
Uses data from evaluation_results.json or any similar JSON file
"""

import json
import numpy as np
import sys
import os
from typing import List, Dict

def load_evaluation_results(filename="evaluation_results.json"):
    """Load evaluation results from JSON file"""
    if not os.path.exists(filename):
        print(f"❌ Archivo {filename} no encontrado")
        print("💡 Ejecuta primero: python bertscore_evaluation.py")
        return None
    
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_bertscore_from_data(references: List[str], candidates: List[str]) -> Dict:
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

def calculate_rouge_from_data(references: List[str], candidates: List[str]) -> Dict:
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
        'rouge1': {
            'mean': float(np.mean(rouge1_scores)),
            'std': float(np.std(rouge1_scores)),
            'scores': rouge1_scores
        },
        'rouge2': {
            'mean': float(np.mean(rouge2_scores)),
            'std': float(np.std(rouge2_scores)),
            'scores': rouge2_scores
        },
        'rougeL': {
            'mean': float(np.mean(rougeL_scores)),
            'std': float(np.std(rougeL_scores)),
            'scores': rougeL_scores
        }
    }

def calculate_bleu_from_data(references: List[str], candidates: List[str]) -> Dict:
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
        'bleu1': {
            'mean': float(np.mean(bleu1_scores)),
            'std': float(np.std(bleu1_scores)),
            'scores': bleu1_scores
        },
        'bleu2': {
            'mean': float(np.mean(bleu2_scores)),
            'std': float(np.std(bleu2_scores)),
            'scores': bleu2_scores
        },
        'bleu3': {
            'mean': float(np.mean(bleu3_scores)),
            'std': float(np.std(bleu3_scores)),
            'scores': bleu3_scores
        },
        'bleu4': {
            'mean': float(np.mean(bleu4_scores)),
            'std': float(np.std(bleu4_scores)),
            'scores': bleu4_scores
        }
    }

def calculate_cosine_similarity_from_data(references: List[str], candidates: List[str]) -> Dict:
    """Calculate cosine similarity"""
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("📦 Instalando dependencias...")
        os.system("pip install sentence-transformers scikit-learn")
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    
    print("📊 Calculando similitud coseno...")
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    
    # Get embeddings
    ref_embeddings = embedder.encode(references)
    cand_embeddings = embedder.encode(candidates)
    
    # Calculate cosine similarities
    similarities = []
    for ref_emb, cand_emb in zip(ref_embeddings, cand_embeddings):
        sim = cosine_similarity([ref_emb], [cand_emb])[0][0]
        similarities.append(float(sim))
    
    return {
        'cosine': {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'scores': similarities
        }
    }

def print_comprehensive_summary(metrics: Dict):
    """Print comprehensive metrics summary"""
    print("\n" + "="*70)
    print("📊 EVALUACIÓN COMPREHENSIVA - TODAS LAS MÉTRICAS")
    print("="*70)
    
    # BERTScore
    if 'bertscore' in metrics:
        bert = metrics['bertscore']
        print(f"\n🤖 BERTScore (Similitud Semántica):")
        print(f"   Precision: {bert['precision']['mean']:.4f} ± {bert['precision']['std']:.4f}")
        print(f"   Recall:    {bert['recall']['mean']:.4f} ± {bert['recall']['std']:.4f}")
        print(f"   F1-Score:  {bert['f1']['mean']:.4f} ± {bert['f1']['std']:.4f}")
        
        # Interpretación BERTScore
        f1_mean = bert['f1']['mean']
        if f1_mean >= 0.9:
            bert_quality = "🌟 EXCELENTE"
        elif f1_mean >= 0.8:
            bert_quality = "⭐ MUY BUENO"
        elif f1_mean >= 0.7:
            bert_quality = "✅ BUENO"
        else:
            bert_quality = "⚠️  MEJORABLE"
        print(f"   Calidad:   {bert_quality}")
    
    # ROUGE
    if 'rouge' in metrics:
        rouge = metrics['rouge']
        print(f"\n📝 ROUGE (N-gramas y Secuencias):")
        print(f"   ROUGE-1:   {rouge['rouge1']['mean']:.4f} ± {rouge['rouge1']['std']:.4f}")
        print(f"   ROUGE-2:   {rouge['rouge2']['mean']:.4f} ± {rouge['rouge2']['std']:.4f}")
        print(f"   ROUGE-L:   {rouge['rougeL']['mean']:.4f} ± {rouge['rougeL']['std']:.4f}")
    
    # BLEU
    if 'bleu' in metrics:
        bleu = metrics['bleu']
        print(f"\n🔤 BLEU (Precisión N-gramas):")
        print(f"   BLEU-1:    {bleu['bleu1']['mean']:.4f} ± {bleu['bleu1']['std']:.4f}")
        print(f"   BLEU-2:    {bleu['bleu2']['mean']:.4f} ± {bleu['bleu2']['std']:.4f}")
        print(f"   BLEU-3:    {bleu['bleu3']['mean']:.4f} ± {bleu['bleu3']['std']:.4f}")
        print(f"   BLEU-4:    {bleu['bleu4']['mean']:.4f} ± {bleu['bleu4']['std']:.4f}")
    
    # Cosine Similarity
    if 'similarity' in metrics:
        sim = metrics['similarity']
        print(f"\n🎯 Similitud Vectorial:")
        print(f"   Coseno:    {sim['cosine']['mean']:.4f} ± {sim['cosine']['std']:.4f}")
    
    # Overall assessment
    print(f"\n📈 EVALUACIÓN GENERAL:")
    if 'bertscore' in metrics and 'rouge' in metrics and 'bleu' in metrics:
        bert_f1 = metrics['bertscore']['f1']['mean']
        rouge1 = metrics['rouge']['rouge1']['mean']
        bleu4 = metrics['bleu']['bleu4']['mean']
        
        overall_score = (bert_f1 + rouge1 + bleu4) / 3
        
        if overall_score >= 0.85:
            overall_quality = "🏆 SISTEMA EXCELENTE"
        elif overall_score >= 0.75:
            overall_quality = "🥈 SISTEMA MUY BUENO"
        elif overall_score >= 0.65:
            overall_quality = "🥉 SISTEMA BUENO"
        else:
            overall_quality = "⚠️  SISTEMA MEJORABLE"
        
        print(f"   Score Global: {overall_score:.4f}")
        print(f"   Calidad:      {overall_quality}")

def main():
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = "evaluation_results.json"
    
    print(f"=== 📊 CÁLCULO DE MÉTRICAS DESDE {input_file.upper()} ===\n")
    
    # Load existing results
    print(f"📂 Cargando resultados desde: {input_file}")
    results = load_evaluation_results(input_file)
    
    if not results:
        sys.exit(1)
    
    print(f"✅ Cargados {len(results)} pares pregunta-respuesta")
    
    # Extract references and candidates
    references = []
    candidates = []
    
    for result in results:
        if 'respuesta_dataset' in result and 'respuesta_generada' in result:
            # Skip error cases
            if not result['respuesta_generada'].startswith('ERROR'):
                references.append(result['respuesta_dataset'])
                candidates.append(result['respuesta_generada'])
    
    print(f"📊 Evaluando {len(references)} pares válidos")
    
    if not references:
        print("❌ No hay datos válidos para evaluar")
        sys.exit(1)
    
    # Calculate all metrics
    all_metrics = {}
    
    print("\n🔄 Calculando todas las métricas...")
    
    # BERTScore
    try:
        bert_metrics = calculate_bertscore_from_data(references, candidates)
        all_metrics['bertscore'] = bert_metrics
    except Exception as e:
        print(f"❌ Error calculando BERTScore: {e}")
    
    # ROUGE
    try:
        rouge_metrics = calculate_rouge_from_data(references, candidates)
        all_metrics['rouge'] = rouge_metrics
    except Exception as e:
        print(f"❌ Error calculando ROUGE: {e}")
    
    # BLEU
    try:
        bleu_metrics = calculate_bleu_from_data(references, candidates)
        all_metrics['bleu'] = bleu_metrics
    except Exception as e:
        print(f"❌ Error calculando BLEU: {e}")
    
    # Cosine Similarity
    try:
        sim_metrics = calculate_cosine_similarity_from_data(references, candidates)
        all_metrics['similarity'] = sim_metrics
    except Exception as e:
        print(f"❌ Error calculando similitud: {e}")
    
    # Create detailed results
    detailed_results = []
    for i, result in enumerate([r for r in results if not r['respuesta_generada'].startswith('ERROR')]):
        detailed_result = result.copy()
        
        # Add individual scores
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
    
    # Save comprehensive results
    output_file = input_file.replace('.json', '_comprehensive_metrics.json')
    comprehensive_results = {
        'source_file': input_file,
        'summary_metrics': all_metrics,
        'detailed_results': detailed_results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(comprehensive_results, f, ensure_ascii=False, indent=2)
    
    # Print results
    print_comprehensive_summary(all_metrics)
    
    # Show examples
    print(f"\n📝 EJEMPLOS DETALLADOS:")
    for i, result in enumerate(detailed_results[:3], 1):
        print(f"\n--- Ejemplo {i} ---")
        print(f"Pregunta: {result['pregunta'][:50]}...")
        print(f"Dataset:  {result['respuesta_dataset'][:60]}...")
        print(f"Generada: {result['respuesta_generada'][:60]}...")
        
        if 'bertscore_f1' in result:
            print(f"BERTScore F1: {result['bertscore_f1']:.3f}")
        if 'rouge1' in result:
            print(f"ROUGE-1:      {result['rouge1']:.3f}")
        if 'bleu4' in result:
            print(f"BLEU-4:       {result['bleu4']:.3f}")
        if 'cosine_similarity' in result:
            print(f"Coseno:       {result['cosine_similarity']:.3f}")
    
    print(f"\n✅ Resultados completos guardados en: {output_file}")

if __name__ == "__main__":
    main()