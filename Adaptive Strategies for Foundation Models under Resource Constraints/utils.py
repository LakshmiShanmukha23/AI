# File: utils.py
import time
import psutil
import torch
from transformers import AutoTokenizer

def measure_latency(func, *args, n_runs=20, **kwargs):
    # warmup
    for _ in range(5):
        func(*args, **kwargs)
    times = []
    for _ in range(n_runs):
        t0 = time.time()
        func(*args, **kwargs)
        times.append(time.time()-t0)
    return sum(times)/len(times)

def gpu_memory_used(device=0):
    if not torch.cuda.is_available():
        return 0.0
    # returns GB of max memory allocated on device
    return torch.cuda.max_memory_allocated(device) / (1024**3)

def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)
