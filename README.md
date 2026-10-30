# Spam Email Classifier

A machine-learning project that classifies emails as **spam** or **ham** (not spam) using TF-IDF features and three classical classifiers: Multinomial Naive Bayes, Logistic Regression, and Linear SVM.

## Dataset
5,171 emails from the Enron spam/ham dataset (3,672 ham, 1,499 spam), downloaded automatically on first run from
[nikhil1102/Spam-Classifier](https://github.com/nikhil1102/Spam-Classifier) and cached in `data/`.

## Pipeline
1. Load data and explore class balance and email length
2. Clean text: lowercase, remove email addresses, URLs, punctuation and numbers, remove stopwords, Porter stemming
3. Stratified 80/20 train/test split (`random_state=42`)
4. TF-IDF vectorisation (top 3,000 features), fitted on the training set only
5. Train and compare the three models
6. Evaluate with accuracy, precision, recall, F1 and confusion matrices

## Results (test set, 1,035 emails)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Naive Bayes | 0.948 | 0.887 | 0.940 | 0.913 |
| Logistic Regression | 0.978 | 0.954 | 0.970 | 0.962 |
| **Linear SVM** | **0.981** | **0.961** | **0.973** | **0.967** |

![Confusion matrices](results/confusion_matrices.png)

## Project structure
```
notebooks/spam_classifier.ipynb   step-by-step walkthrough with outputs
src/preprocess.py                 text cleaning
src/train.py                      train, evaluate, save best model
src/predict.py                    classify new emails (CLI / interactive)
results/                          metrics.json and confusion matrix plot
requirements.txt
```

## Quick start
```bash
pip install -r requirements.txt
python src/train.py                       # trains, writes results/ and models/spam_model.joblib
python src/predict.py "Subject: You won a free prize, click here to claim now!!!"   # -> SPAM
python src/predict.py                     # interactive mode
```
NLTK stopwords are downloaded automatically on first run.

## Limitations
- The ham emails are all from Enron employees, so the model partly learns Enron-specific vocabulary (`enron`, `meter`, `deal`, ...). It performs well on this dataset but may generalise poorly to ordinary personal or modern email, and very short messages can be misclassified.
- The dataset contains ~178 duplicate emails. Removing them changes Linear SVM accuracy only slightly (98.1% to 97.7%).

## Ideas for improvement
Cross-validation and hyperparameter tuning, n-gram features, a more diverse training set, and a small web demo (Streamlit / Flask).
