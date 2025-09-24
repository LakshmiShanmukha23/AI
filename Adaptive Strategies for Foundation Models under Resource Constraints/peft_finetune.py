# File: peft_finetune.py
"""
Parameter-Efficient Fine-Tuning (LoRA) example using the `peft` library.
Example usage:
python peft_finetune.py --model distilbert-base-uncased --dataset glue --subset sst2 --output_dir peft_out --epochs 3
"""
import argparse
from datasets import load_dataset
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    parser.add_argument('--output_dir', type=str, default='peft_out')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, args.subset)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def preprocess(ex):
        text = ex.get('sentence', ex.get('sentence1'))
        return tokenizer(text, truncation=True, padding='max_length', max_length=128)

    dataset = dataset.map(lambda x: preprocess(x), batched=False)

    num_labels = dataset['train'].features['label'].num_classes if hasattr(dataset['train'].features['label'], 'num_classes') else len(set(dataset['train']['label']))
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)

    # Prepare LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=['q', 'v'] if 'bert' in args.model or 'distilbert' in args.model else None,
        lora_dropout=0.1,
        bias='none',
        task_type='SEQ_CLS'
    )

    # Optionally prepare model for k-bit training if using 8-bit quantization (requires bitsandbytes)
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        # prepare_model_for_kbit_training is optional; continue if not available
        pass

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy='epoch',
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_dir=args.output_dir + '/logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'] if 'validation' in dataset else dataset['test'],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

if __name__ == '__main__':
    main()
