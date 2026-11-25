"""
Advanced model: Fine-tune DistilBERT for Red Flag classification.
More accurate than the baseline, especially on nuanced phrasing.

Usage:
    python src/train_bert.py

Requirements: torch, transformers, datasets
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import load_and_split

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 128
BATCH_SIZE   = 16
EPOCHS       = 5
LR           = 2e-5
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH    = "models/distilbert_redflag"

print(f"🖥️  Using device: {DEVICE}")

# ─── GENERATE DATASET IF MISSING ─────────────────────────────────────────────
if not os.path.exists("data/dataset.csv"):
    print("⚙️  Dataset not found. Generating...")
    from data.generate_dataset import build_dataset
    build_dataset()

# ─── LOAD & SPLIT DATA ───────────────────────────────────────────────────────
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()

# ─── TOKENIZER ───────────────────────────────────────────────────────────────
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)


class RedFlagDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


train_ds = RedFlagDataset(X_train, y_train, tokenizer, MAX_LEN)
val_ds   = RedFlagDataset(X_val,   y_val,   tokenizer, MAX_LEN)
test_ds  = RedFlagDataset(X_test,  y_test,  tokenizer, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

# ─── MODEL ────────────────────────────────────────────────────────────────────
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps,
)


# ─── TRAINING LOOP ───────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in tqdm(loader, desc="Training"):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


def eval_epoch(model, loader):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    return total_loss / len(loader), accuracy_score(all_labels, all_preds), all_preds, all_labels


# ─── RUN TRAINING ────────────────────────────────────────────────────────────
print(f"\n🏋️  Fine-tuning {MODEL_NAME} for {EPOCHS} epochs...\n")
best_val_acc = 0

for epoch in range(1, EPOCHS + 1):
    print(f"\n── Epoch {epoch}/{EPOCHS} ──")
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
    val_loss, val_acc, _, _ = eval_epoch(model, val_loader)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        os.makedirs(SAVE_PATH, exist_ok=True)
        model.save_pretrained(SAVE_PATH)
        tokenizer.save_pretrained(SAVE_PATH)
        print(f"  ✅ Best model saved → {SAVE_PATH}")

# ─── TEST EVALUATION ─────────────────────────────────────────────────────────
print("\n📈 Final Test Evaluation:")
_, test_acc, test_preds, test_labels = eval_epoch(model, test_loader)
print(classification_report(test_labels, test_preds, target_names=["Green Flag ✅", "Red Flag 🚩"]))
print(f"Test Accuracy: {test_acc:.4f}")
