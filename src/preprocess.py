"""
Preprocessing utilities for the Red Flag Text Classifier.
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s',.!?]", "", text)   # keep common punctuation
    text = re.sub(r"\s+", " ", text)
    return text


def load_and_split(csv_path: str = "data/dataset.csv", test_size: float = 0.2, val_size: float = 0.1):
    """
    Load dataset CSV and return train / val / test splits.
    Returns: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].apply(clean_text)

    X = df["text"].tolist()
    y = df["label"].tolist()

    # First split off test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # Then split validation from training
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=42, stratify=y_train_val
    )

    print(f"📊 Split sizes → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def get_category_map():
    return {
        "gaslighting": 0,
        "love_bombing": 1,
        "controlling": 2,
        "guilt_tripping": 3,
        "manipulation": 4,
        "disrespect": 5,
        "green_flag": 6,
    }
