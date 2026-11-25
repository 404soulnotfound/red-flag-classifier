# 🚩 Red Flag Text Classifier

A machine learning project that classifies text messages as **Red Flag 🚩** or **Green Flag ✅** — detecting manipulative, controlling, or toxic communication patterns.

---

## 🗂 Project Structure

```
red-flag-classifier/
├── data/
│   ├── generate_dataset.py     # Synthetic dataset generator
│   └── dataset.csv             # Generated training data
├── models/
│   ├── baseline_model.joblib           # Saved baseline model
│   ├── baseline_confusion_matrix.png   # Evaluation plot
│   └── distilbert_redflag/             # Saved BERT model (optional)
├── src/
│   ├── preprocess.py           # Text cleaning & data splitting
│   ├── train_baseline.py       # TF-IDF + ComplementNB
│   ├── train_bert.py           # DistilBERT fine-tuning
│   └── predict.py              # Inference utilities
├── app.py                      # Web UI (Gradio or built-in HTTP server)
├── run.py                      # One-click full pipeline runner
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run everything (generate data → train → launch UI)
```bash
python run.py
```

### 3. Or step by step:
```bash
# Generate dataset
python data/generate_dataset.py

# Train baseline model (fast, ~seconds)
python src/train_baseline.py

# (Optional) Fine-tune DistilBERT (slower, more accurate)
python src/train_bert.py

# Launch web app
python app.py
```
Then open **http://localhost:7860** in your browser.

---

## 🔴 Red Flag Categories

| Category | Description | Example |
|----------|-------------|---------|
| **Gaslighting** | Making someone question their reality | *"You're imagining things, that never happened."* |
| **Love Bombing** | Overwhelming affection used to manipulate | *"I want to marry you after 3 days of knowing you."* |
| **Controlling** | Monitoring, restricting behaviour | *"Who is that guy? Delete that photo now."* |
| **Guilt Tripping** | Making someone feel responsible for your emotions | *"After everything I've done for you..."* |
| **Manipulation** | Coercive tactics | *"If you loved me, you wouldn't go out."* |
| **Disrespect** | Belittling or demeaning language | *"You're so stupid, how do you not understand?"* |

---

## 🤖 Models

### Baseline — TF-IDF + ComplementNB
- Fast to train (seconds)
- Handles class imbalance well and shows which words drive the prediction
- Achieves ~83.3% test accuracy / ~0.88 ROC-AUC on the synthetic dataset

### Advanced — DistilBERT Fine-tuned
- Pre-trained transformer, understands context
- Higher accuracy on nuanced or ambiguous texts
- Requires PyTorch + ~5 min to fine-tune on CPU (faster on GPU)

---


## 📦 Requirements

- Python 3.9+
- See `requirements.txt` for packages (`app.py` also includes a zero-dependency built-in web UI fallback if Gradio is not installed)

---

## ⚠️ Disclaimer

This project is trained on synthetic text examples for educational and portfolio purposes only. It is not a substitute for professional psychological or relationship advice.
