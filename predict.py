"""Classify an email as SPAM or HAM with the trained model.

Usage:
    python src/predict.py "Subject: You won a free prize, click here!"
    python src/predict.py            # interactive mode, type 'exit' to quit
"""
import sys
from pathlib import Path

import joblib

from preprocess import clean_text

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "spam_model.joblib"


def load_model():
    if not MODEL_PATH.exists():
        sys.exit("No trained model found. Run:  python src/train.py")
    return joblib.load(MODEL_PATH)


def predict_email(model, text: str) -> str:
    return "SPAM" if model.predict([clean_text(text)])[0] == 1 else "HAM"


def main() -> None:
    model = load_model()
    if len(sys.argv) > 1:
        print(predict_email(model, " ".join(sys.argv[1:])))
        return
    print("Email Spam Classifier - type 'exit' to quit")
    while True:
        text = input("\nEnter an email/message: ")
        if text.strip().lower() == "exit":
            break
        print(f"Prediction: {predict_email(model, text)}")


if __name__ == "__main__":
    main()
