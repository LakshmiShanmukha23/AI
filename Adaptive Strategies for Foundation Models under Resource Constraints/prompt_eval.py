# File: prompt_eval.py
"""
Evaluate prompting performance (zero-shot / few-shot) for a seq2seq or causal model by generating answers with templates.
Example:
python prompt_eval.py --model google/flan-t5-base --dataset glue --subset sst2 --n_examples 20
"""
import argparse
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import measure_latency
import torch

def generate_prompt(example=None, template=None):
    if example is None:
        return template
    return template.format(input=example)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='google/flan-t5-base')
    parser.add_argument('--dataset', type=str, default='glue')
    parser.add_argument('--subset', type=str, default='sst2')
    parser.add_argument('--n_examples', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(args.device)
    dataset = load_dataset(args.dataset, args.subset, split='validation')

    template = "Classify the sentiment of the following sentence. Return POSITIVE or NEGATIVE.\nInput: {input}\nAnswer:"

    def infer_single_text(text):
        prompt = generate_prompt(text, template)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True).to(args.device)
        with torch.inference_mode():
            out = model.generate(**inputs, max_length=50)
        ans = tokenizer.decode(out[0], skip_special_tokens=True)
        return ans

    # measure latency for one sample
    lat = measure_latency(lambda: infer_single_text(dataset[0]['sentence']), n_runs=10)
    print(f"Avg latency (s) per sample (estimate): {lat:.4f}")

    # run through a few examples and print predictions
    for i in range(min(args.n_examples, len(dataset))):
        text = dataset[i].get('sentence') or dataset[i].get('sentence1') or str(dataset[i])
        pred = infer_single_text(text)
        print(i, pred)

if __name__ == '__main__':
    main()
