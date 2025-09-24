# File: adaptive_selector.py
"""
A lightweight rule-based selector that recommends an adaptation strategy based on resource and task inputs.
Example:
python adaptive_selector.py --memory_gb 8.0 --latency_ms 300 --data_size 100 --knowledge
"""
import argparse

def select_strategy(memory_gb, latency_ms, data_size, knowledge_heavy=False, required_accuracy=0.8):
    # memory_gb: available GPU memory in GB (per GPU)
    # latency_ms: target per-sample latency budget
    # data_size: number of labelled samples available

    # low-memory fast path
    if memory_gb < 8 and latency_ms < 300:
        return {'strategy': 'prompting', 'notes': 'Use small model (distilbert/flan-small), FP16 and few-shot prompting.'}

    # data-rich supervised
    if data_size >= 1000 and memory_gb >= 12:
        return {'strategy': 'fine-tuning', 'notes': 'Fine-tune with LoRA or FP16 training. Use early stopping.'}

    # knowledge-heavy tasks
    if knowledge_heavy and latency_ms >= 500:
        return {'strategy': 'RAG', 'notes': 'Build FAISS index and use retrieval-augmented generation.'}

    # default conservative
    if memory_gb >= 8:
        return {'strategy': 'prompting_or_lora', 'notes': 'Consider LoRA for small compute fine-tuning; else prompting.'}

    return {'strategy': 'prompting', 'notes': 'Fallback to prompting.'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory_gb', type=float, default=8.0)
    parser.add_argument('--latency_ms', type=int, default=300)
    parser.add_argument('--data_size', type=int, default=100)
    parser.add_argument('--knowledge', action='store_true')
    args = parser.parse_args()

    rec = select_strategy(args.memory_gb, args.latency_ms, args.data_size, args.knowledge)
    print('Recommendation:')
    for k,v in rec.items():
        print(k, ':', v)

if __name__ == '__main__':
    main()
