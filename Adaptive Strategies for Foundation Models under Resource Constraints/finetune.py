# File: finetune.py
"""
Simple supervised fine-tuning script using Hugging Face Trainer for text classification.
Example usage:
python finetune.py --model distilbert-base-uncased --dataset glue --subset sst2 --epochs 3 --output_dir out
"""
import argparse
import os
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='distilbert-base-uncased')
    parser.add_argument('--dataset', type=str, default='glue')
    parser.add_argument('--subset', type=str, default='sst2')
    parser.add_argument('--output_dir', type=str, default='out')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.subset)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def preprocess(ex):
        # support GLUE-style datasets (sst2: 'sentence', mrpc: 'sentence1'/'sentence2')
        if 'sentence' in ex:
            return tokenizer(ex['sentence'], truncation=True, padding='max_length', max_length=128)
        if 'sentence1' in ex and 'sentence2' in ex:
            return tokenizer(ex['sentence1'], ex['sentence2'], truncation=True, padding='max_length', max_length=128)
        # fallback: take first text column
        first_text = None
        for k in ex.keys():
            if isinstance(ex[k], str):
                first_text = ex[k]
                break
        return tokenizer(first_text, truncation=True, padding='max_length', max_length=128)

    # map preprocess
    dataset = dataset.map(lambda x: preprocess(x), batched=False)

    # derive num_labels
    try:
        num_labels = dataset['train'].features['label'].num_classes
    except Exception:
        # fallback
        labels = dataset['train']['label']
        num_labels = len(set(labels))

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        fp16=args.fp16,
        logging_dir=os.path.join(args.output_dir, 'logs')
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'] if 'validation' in dataset else dataset['test'],
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()

if __name__ == '__main__':
    main()
