from retrieval import get_retriever, get_response 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from sklearn.metrics import ndcg_score
from rank_bm25 import BM25Okapi
import evaluate
import json
import os
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util

def evaluate():
    with open("evaluation_data.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    retriever = get_retriever()  
    smooth = SmoothingFunction().method1

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    total_ndcg, total_mrr = 0, 0
    bleu_scores, rouge1_scores, rouge2_scores, rougeL_scores, semantic_scores = [], [], [], [], []

    for example in data:
        query = example["query"]
        relevant_snippets = example["relevant_doc_snippets"]
        reference = example["reference_answer"]

        # 1. Evaluate Retrieval (NDCG, MRR)
        docs = retriever.invoke(query)
        scores = [1 if any(rel in doc.page_content for rel in relevant_snippets) else 0 for doc in docs]
        relevance = [scores]
        ideal = [[1]*len(scores)]
        total_ndcg += ndcg_score(ideal, relevance)
        try:
            total_mrr += 1 / (scores.index(1) + 1)
        except ValueError:
            pass

        # 2. Evaluate Response (BLEU, ROUGE-1, ROUGE-2, ROUGE-L, Semantic)
        generated = get_response(query)
        bleu = sentence_bleu([reference.split()], generated.split(), smoothing_function=smooth)
        rouge_scores = scorer.score(reference, generated)

        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
        bleu_scores.append(bleu)

        # Semantic similarity
        embedding_ref = model.encode(reference, convert_to_tensor=True)
        embedding_gen = model.encode(generated, convert_to_tensor=True)
        sim_score = util.pytorch_cos_sim(embedding_ref, embedding_gen).item()
        semantic_scores.append(sim_score)

    num = len(data)
    print(f"NDCG: {total_ndcg / num:.3f}")
    print(f"MRR: {total_mrr / num:.3f}")
    print(f"BLEU: {sum(bleu_scores) / num:.3f}")
    print(f"ROUGE-1: {sum(rouge1_scores) / num:.3f}")
    print(f"ROUGE-2: {sum(rouge2_scores) / num:.3f}")
    print(f"ROUGE-L: {sum(rougeL_scores) / num:.3f}")
    print(f"Semantic Similarity: {sum(semantic_scores) / num:.3f}")

if __name__ == "__main__":
    evaluate()
