# strategy_recommender.py

import argparse

def recommend_strategy(task, hardware, memory_gb, latency_ms, dataset_size=None):
    recommendation = {}

    # ------------------------
    # Step 1: Model Size
    # ------------------------
    if hardware == "cpu" or memory_gb < 8:
        recommendation["model"] = "distilbert-base-uncased" if task != "summarization" else "flan-t5-small"
    elif 8 <= memory_gb <= 16:
        recommendation["model"] = "bert-base-uncased" if task != "summarization" else "flan-t5-base"
    else:
        recommendation["model"] = "bert-large-uncased" if task != "summarization" else "flan-t5-xl"

    # ------------------------
    # Step 2: Adaptation Strategy
    # ------------------------
    if hardware == "cpu" and (dataset_size is None or dataset_size < 5000):
        recommendation["strategy"] = "Prompting" if task != "qa" else "RAG"
    elif memory_gb < 8:
        recommendation["strategy"] = "LoRA Fine-tuning with Quantization"
    elif dataset_size and dataset_size > 50000:
        recommendation["strategy"] = "LoRA Fine-tuning"
    elif task in ["qa", "chatbot"]:
        recommendation["strategy"] = "RAG"
    else:
        recommendation["strategy"] = "Prompting"

    # ------------------------
    # Step 3: Quantization
    # ------------------------
    if memory_gb < 8:
        recommendation["quantization"] = "INT8"
    elif memory_gb <= 16:
        recommendation["quantization"] = "FP16"
    else:
        recommendation["quantization"] = "Full Precision"

    # ------------------------
    # Step 4: Latency-based Adjustments
    # ------------------------
    if latency_ms < 100:
        recommendation["latency_mode"] = "Low-latency optimization (smaller model, batch=1, FP16/INT8)"
    else:
        recommendation["latency_mode"] = "Standard inference"

    return recommendation


def main():
    parser = argparse.ArgumentParser(description="Adaptive Strategy Recommender for Foundation Models")
    parser.add_argument("--task", type=str, required=True, help="Task type: classification, qa, summarization, chatbot")
    parser.add_argument("--hardware", type=str, required=True, help="Hardware: cpu or gpu")
    parser.add_argument("--memory_gb", type=float, required=True, help="Available memory in GB")
    parser.add_argument("--latency_ms", type=int, required=True, help="Latency requirement in ms")
    parser.add_argument("--dataset_size", type=int, required=False, help="Optional: size of dataset")

    args = parser.parse_args()

    rec = recommend_strategy(
        task=args.task.lower(),
        hardware=args.hardware.lower(),
        memory_gb=args.memory_gb,
        latency_ms=args.latency_ms,
        dataset_size=args.dataset_size,
    )

    print("\n=== Adaptive Strategy Recommendation ===")
    print(f"✅ Task Type: {args.task}")
    print(f"✅ Hardware: {args.hardware} ({args.memory_gb} GB)")
    print(f"✅ Latency Requirement: {args.latency_ms} ms")
    if args.dataset_size:
        print(f"✅ Dataset Size: {args.dataset_size}")
    print("----------------------------------------")
    for k, v in rec.items():
        print(f"🔹 {k.capitalize()}: {v}")


if __name__ == "__main__":
    main()
