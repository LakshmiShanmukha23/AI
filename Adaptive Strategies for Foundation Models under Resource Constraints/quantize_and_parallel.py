# File: quantize_and_parallel.py
"""
Demonstrates simple ways to load models with FP16 or 8-bit (via bitsandbytes) and run on multiple GPUs.
Note: For serious multi-GPU training, use `accelerate` and its config.
Example:
python quantize_and_parallel.py --model google/flan-t5-base --fp16
"""
import argparse
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def load_model(model_name, use_fp16=False, use_8bit=False, device_map=None):
    kwargs = {}
    if use_8bit:
        # Requires bitsandbytes and transformers with 8-bit support
        kwargs['load_in_8bit'] = True
    if use_fp16:
        kwargs['torch_dtype'] = torch.float16
    if device_map:
        kwargs['device_map'] = device_map
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='google/flan-t5-base')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--int8', action='store_true')
    args = parser.parse_args()

    device_map = None
    if torch.cuda.device_count() > 1:
        # simple model parallel example: split layers across GPUs automatically (transformers supports "auto")
        device_map = 'auto'

    model = load_model(args.model, use_fp16=args.fp16, use_8bit=args.int8, device_map=device_map)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print('Model loaded. Device map:', device_map)

    # simple inference
    text = "Summarize: Transformers simplify NLP tasks."
    inputs = tokenizer(text, return_tensors='pt')
    # move inputs to model device (first parameter)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_length=50)
    print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == '__main__':
    main()
