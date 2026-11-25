"""
Baseline model: TF-IDF + Logistic Regression
Fast to train, good interpretability, great starting point.

Usage:
    python src/train_baseline.py
"""

import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.preprocess import load_and_split

# ─── GENERATE DATASET IF MISSING ─────────────────────────────────────────────
if not os.path.exists("data/dataset.csv"):
    print("⚙️  Dataset not found. Generating...")
    from data.generate_dataset import build_dataset
    build_dataset()

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
X_train, X_val, X_test, y_train, y_val, y_test = load_and_split()

# ─── BUILD PIPELINE ──────────────────────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),     # unigrams + bigrams
        max_features=10_000,
        sublinear_tf=True,
    )),
    ("clf", ComplementNB(
        alpha=0.5,
    )),
])

# ─── TRAIN ────────────────────────────────────────────────────────────────────
print("\n🏋️  Training TF-IDF + ComplementNB...")
pipeline.fit(X_train, y_train)

# ─── EVALUATE ────────────────────────────────────────────────────────────────
print("\n📈 Validation Results:")
val_preds = pipeline.predict(X_val)
print(classification_report(y_val, val_preds, target_names=["Green Flag ✅", "Red Flag 🚩"]))

print("\n📈 Test Results:")
test_preds = pipeline.predict(X_test)
test_probs = pipeline.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, test_preds)
auc = roc_auc_score(y_test, test_probs)
print(classification_report(y_test, test_preds, target_names=["Green Flag ✅", "Red Flag 🚩"]))
print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {auc:.4f}")

# ─── CONFUSION MATRIX ─────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
cm = confusion_matrix(y_test, test_preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Reds",
    xticklabels=["Green Flag", "Red Flag"],
    yticklabels=["Green Flag", "Red Flag"],
    ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix — Baseline Model")
plt.tight_layout()
plt.savefig("models/baseline_confusion_matrix.png", dpi=150)
plt.close()
print("\n📊 Confusion matrix saved → models/baseline_confusion_matrix.png")

# ─── TOP FEATURES ─────────────────────────────────────────────────────────────
tfidf = pipeline.named_steps["tfidf"]
clf   = pipeline.named_steps["clf"]
feature_names = tfidf.get_feature_names_out()
coefs = clf.feature_log_prob_[1] - clf.feature_log_prob_[0]

top_red   = np.argsort(coefs)[-15:][::-1]
top_green = np.argsort(coefs)[:15]

print("\n🚩 Top Red Flag words/phrases:")
for i in top_red:
    print(f"  {feature_names[i]:<30} coef={coefs[i]:.3f}")

print("\n✅ Top Green Flag words/phrases:")
for i in top_green:
    print(f"  {feature_names[i]:<30} coef={coefs[i]:.3f}")

# ─── SAVE MODEL ───────────────────────────────────────────────────────────────
model_path = "models/baseline_model.joblib"
joblib.dump(pipeline, model_path)
print(f"\n💾 Model saved → {model_path}")
