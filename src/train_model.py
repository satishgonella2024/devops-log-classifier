from datasets import Dataset
import evaluate
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import login

# Load Hugging Face Token
import os
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
login(HUGGINGFACE_TOKEN)

# Load Dataset
df = pd.read_csv("data/kubernetes_synthetic_logs.csv")
label_mapping = {"ERROR": 0, "WARNING": 1, "INFO": 2}
df["label"] = df["log_level"].map(label_mapping)


train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["log_text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)


train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length")

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Model and Metrics
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=predictions, references=labels)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
    hub_model_id="satishgonella/devops-log-classifier",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    compute_metrics=compute_metrics,
)

# Train and Push
trainer.train()
trainer.push_to_hub()

# Save and push the tokenizer to the Hub
tokenizer.save_pretrained(training_args.output_dir)
tokenizer.push_to_hub("satishgonella/devops-log-classifier")

# Push the model to the Hub
trainer.push_to_hub()

