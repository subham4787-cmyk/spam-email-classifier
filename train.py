"""Train and compare spam classifiers, then save the best one.

Usage:  python src/train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from preprocess import clean_text

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "spam_ham_dataset.csv"
DATA_URL = ("https://raw.githubusercontent.com/nikhil1102/Spam-Classifier/"
            "main/spam_ham_dataset.csv")
SEED = 42


def load_data() -> pd.DataFrame:
    """Load the dataset, downloading and caching it on first use."""
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        pd.read_csv(DATA_URL).to_csv(DATA_PATH, index=False)
    return pd.read_csv(DATA_PATH)[["label", "text"]]


def main() -> None:
    df = load_data()
    print(f"Loaded {df.shape[0]} emails\n{df['label'].value_counts()}\n")

    df["clean_text"] = df["text"].apply(clean_text)
    y = df["label"].map({"ham": 0, "spam": 1})
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], y, test_size=0.2, random_state=SEED, stratify=y
    )

    classifiers = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Linear SVM": LinearSVC(),
    }

    metrics, pipelines = {}, {}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, clf) in zip(axes, classifiers.items()):
        pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=3000)),
                         ("clf", clf)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        metrics[name] = {
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred),
            "recall": recall_score(y_test, pred),
            "f1": f1_score(y_test, pred),
        }
        pipelines[name] = pipe
        ConfusionMatrixDisplay(confusion_matrix(y_test, pred),
                               display_labels=["ham", "spam"]).plot(
            ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(f"{name}\nF1 = {metrics[name]['f1']:.3f}")
        print(f"{name:20s}", {k: round(v, 4) for k, v in metrics[name].items()})

    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(results / "confusion_matrices.png", dpi=150)
    (results / "metrics.json").write_text(json.dumps(metrics, indent=2))

    best = max(metrics, key=lambda n: metrics[n]["f1"])
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(pipelines[best], ROOT / "models" / "spam_model.joblib")
    print(f"\nBest model: {best} -> models/spam_model.joblib")


if __name__ == "__main__":
    main()
