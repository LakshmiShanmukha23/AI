# File: rag_eval.py
"""
Extended RAG pipeline:
1. Builds FAISS index from documents.
2. Allows adding new documents interactively.
3. Performs top-k retrieval.
4. Passes context to generator with prompt engineering.
5. Benchmarks latency and memory usage.

Example:
python rag_eval.py --embed_model all-MiniLM-L6-v2 --gen_model google/flan-t5-base --k 4
python rag_eval.py --interactive
"""
import argparse
import time
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import gpu_memory_used

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def add_to_index(index, embeddings):
    # append embeddings to an existing index
    index.add(embeddings)

def retrieve(index, query_emb, docs, k=4):
    D, I = index.search(query_emb, k)
    retrieved = [docs[i] for i in I[0]]
    return retrieved, D

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embed_model', type=str, default='all-MiniLM-L6-v2')
    parser.add_argument('--gen_model', type=str, default='google/flan-t5-base')
    parser.add_argument('--k', type=int, default=4)
    parser.add_argument('--interactive', action='store_true', help='Enable interactive Q&A loop')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Starter document set (replace/extend with your corpus)
    docs = [
        "Python is a programming language that lets you work quickly.",
        "Transformers makes it easy to use pretrained models for NLP tasks.",
        "FAISS enables efficient similarity search in dense vector spaces.",
        "DistilBERT is a distilled version of BERT for lightweight use.",
        "LoRA allows parameter-efficient fine-tuning of LLMs.",
    ]

    print("Encoding documents...")
    embedder = SentenceTransformer(args.embed_model)
    embeddings = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=True)
    index = build_faiss_index(embeddings)

    tokenizer = AutoTokenizer.from_pretrained(args.gen_model)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(args.gen_model).to(args.device)

    def answer_query(query):
        start_time = time.time()
        q_emb = embedder.encode([query], convert_to_numpy=True)
        retrieved, distances = retrieve(index, q_emb, docs, k=args.k)
        context = "\n".join(retrieved)
        prompt = f"Use the following context to answer the question concisely.\nContext:\n{context}\nQuestion: {query}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(args.device)
        with torch.inference_mode():
            out = gen_model.generate(**inputs, max_length=128)
        ans = tokenizer.decode(out[0], skip_special_tokens=True)
        latency = time.time() - start_time
        mem = gpu_memory_used() if torch.cuda.is_available() else 0.0
        return ans, latency, mem, retrieved, distances

    if args.interactive:
        print("Entering interactive RAG Q&A mode. Commands: 'add' to add doc, 'exit' to quit.")
        while True:
            q = input("Enter query: ").strip()
            if q.lower() in ['exit', 'quit']:
                break
            if q.lower() == 'add':
                new_doc = input("Enter new document text: ").strip()
                docs.append(new_doc)
                new_emb = embedder.encode([new_doc], convert_to_numpy=True)
                add_to_index(index, new_emb)
                print("Document added to index. Current doc count:", len(docs))
                continue
            ans, latency, mem, retrieved, distances = answer_query(q)
            print(f"\nAnswer: {ans}\nLatency: {latency:.3f}s | GPU memory: {mem:.2f} GB")
            print("Retrieved docs and distances:")
            for r, d in zip(retrieved, distances[0]):
                print(f" - [{d:.4f}] {r}")
            print("\n---\n")
    else:
        query = "How do I efficiently fine-tune a large model?"
        ans, latency, mem, retrieved, distances = answer_query(query)
        print(f"Query: {query}\nAnswer: {ans}\nLatency: {latency:.3f}s | GPU memory: {mem:.2f} GB")
        print("Retrieved docs:")
        for r, d in zip(retrieved, distances[0]):
            print(f" - [{d:.4f}] {r}")

if __name__ == '__main__':
    main()
