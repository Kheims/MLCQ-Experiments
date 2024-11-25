import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import json
import logging
from sklearn.metrics import classification_report, precision_recall_fscore_support
from tqdm import tqdm
import time

logging.basicConfig(
    filename='codebert_log.txt',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class CodeSmellDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
            #data = json.load(f)
            #self.data = data[:2000]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Map smells to numeric labels
        self.smell_to_label = {
            "blob": 0,
            "feature envy": 1,
            "long method": 2,
            "data class": 3,
            "no smell": 4
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        code = item['code_snippet']
        label = item['smell'] if item['severity'] != "none" else "no smell"
        label = self.smell_to_label[label]

        inputs = self.tokenizer(
            code,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}
        inputs['labels'] = torch.tensor(label)
        
        return inputs

    def get_smell_repartition(self):
        self.smell_count = {}
        for data in self.data:
            smell = data['smell'] if data['severity'] != 'none' else 'no smell'
            self.smell_count[smell] = self.smell_count.get(smell, 0) + 1
        
        for smell, instances in self.smell_count.items():
            print(f"{smell} has {instances} occurrences")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    metrics = {
        "precision_blob": precision[0],
        "recall_blob": recall[0],
        "f1_blob": f1[0],
        "precision_feature_envy": precision[1],
        "recall_feature_envy": recall[1],
        "f1_feature_envy": f1[1],
        "precision_long_method": precision[2],
        "recall_long_method": recall[2],
        "f1_long_method": f1[2],
        "precision_data_class": precision[3],
        "recall_data_class": recall[3],
        "f1_data_class": f1[3],
        "precision_no_smell": precision[4],
        "recall_no_smell": recall[4],
        "f1_no_smell": f1[4],
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1
    }
    
    return metrics
def main():
    tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')
    model = RobertaForSequenceClassification.from_pretrained('microsoft/codebert-base', num_labels=5)

    dataset = CodeSmellDataset("MLCQCodeSmellSamples.json", tokenizer)
    dataset.get_smell_repartition()
    
    smelly_indices = [i for i, item in enumerate(dataset) if item['labels'] != 4]
    smelly_dataset = Subset(dataset, smelly_indices)

    oversampled_smelly_dataset = Subset(dataset, smelly_indices * 3)
    balanced_dataset = ConcatDataset([dataset, oversampled_smelly_dataset])
    
    train_size = int(0.8 * len(balanced_dataset))
    val_size = len(balanced_dataset) - train_size
    train_dataset, val_dataset = random_split(balanced_dataset, [train_size, val_size])

    # Define Trainer arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=2,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    logging.info("\nStarting training...")
    logging.info(f"Number of Epochs: {training_args.num_train_epochs}")
    logging.info(f"Batch Size: {training_args.per_device_train_batch_size}")
    logging.info(f"Warmup steps : {training_args.warmup_steps}")
    logging.info(f"Weight decay : {training_args.weight_decay}")

    trainer.train()

    logging.info("Evaluating model...")
    results = trainer.evaluate()
    logging.info(f"Evaluation results: {results}")
    logging.info("Training and evaluation completed.")

    model.save_pretrained('./best_model')

if __name__ == "__main__":
    main()
