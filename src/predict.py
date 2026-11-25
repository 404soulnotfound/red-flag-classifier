"""
Inference helper — load model and predict on new texts.
Works with either the baseline or DistilBERT model.
"""

import os
import sys
import joblib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import clean_text

BERT_PATH     = "models/distilbert_redflag"
BASELINE_PATH = "models/baseline_model.joblib"

RED_FLAG_CATEGORIES = {
    "gaslighting": ["imagining", "overreacting", "never said", "misremember", "dramatic"],
    "love_bombing": ["soulmate", "never felt", "my everything", "marry you", "perfect"],
    "controlling": ["location", "who is that", "delete", "let me see your phone", "where are you"],
    "guilt_tripping": ["after everything", "i sacrifice", "i guess i don't matter", "you owe me"],
    "manipulation": ["if you loved me", "you'd be lost", "i'll hurt myself", "no one else will"],
    "disrespect": ["stupid", "shut up", "pathetic", "dumb", "you're a joke"],
}


def guess_category(text: str) -> str:
    """Heuristic category guesser for red flag texts."""
    text_lower = text.lower()
    for category, keywords in RED_FLAG_CATEGORIES.items():
        if any(kw in text_lower for kw in keywords):
            return category.replace("_", " ").title()
    return "General Red Flag"


class BaselinePredictor:
    def __init__(self):
        self.model = joblib.load(BASELINE_PATH)

    def predict(self, text: str):
        cleaned = clean_text(text)
        prob    = self.model.predict_proba([cleaned])[0]
        label   = int(np.argmax(prob))
        return label, float(prob[1])  # (label, red_flag_prob)


class BERTPredictor:
    def __init__(self):
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        self.torch     = torch
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_PATH)
        self.model     = DistilBertForSequenceClassification.from_pretrained(BERT_PATH).to(self.device)
        self.model.eval()

    def predict(self, text: str):
        cleaned = clean_text(text)
        enc     = self.tokenizer(
            cleaned, return_tensors="pt",
            max_length=128, padding="max_length", truncation=True
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with self.torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = self.torch.softmax(logits, dim=1).cpu().numpy()[0]
        label = int(np.argmax(probs))
        return label, float(probs[1])


def load_predictor(use_bert: bool = False):
    if use_bert and os.path.exists(BERT_PATH):
        print("🤖 Loading DistilBERT predictor...")
        return BERTPredictor()
    elif os.path.exists(BASELINE_PATH):
        print("📦 Loading baseline predictor...")
        return BaselinePredictor()
    else:
        raise FileNotFoundError("No trained model found. Run train_baseline.py or train_bert.py first.")


def format_result(text: str, label: int, prob: float) -> dict:
    if label == 1:
        category = guess_category(text)
        return {
            "verdict":    "🚩 RED FLAG",
            "category":   category,
            "confidence": f"{prob * 100:.1f}%",
            "emoji":      "🚩",
        }
    else:
        return {
            "verdict":    "✅ GREEN FLAG",
            "category":   "Healthy Communication",
            "confidence": f"{(1 - prob) * 100:.1f}%",
            "emoji":      "✅",
        }
