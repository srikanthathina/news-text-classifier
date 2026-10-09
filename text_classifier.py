"""
News Article Text Classifier (NLP + Machine Learning)
Author: Srikanth Athini

Classifies news posts into 20 topics using TF-IDF features and compares
three ML models: Naive Bayes, Logistic Regression and Linear SVM.

Run in Google Colab or locally:  python text_classifier.py
Requires: pip install scikit-learn matplotlib joblib
"""

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# 1. Load data (headers/footers/quotes removed so the model learns from the text itself)
REMOVE = ("headers", "footers", "quotes")
train = fetch_20newsgroups(subset="train", remove=REMOVE, random_state=42)
test = fetch_20newsgroups(subset="test", remove=REMOVE, random_state=42)
print(f"Train documents: {len(train.data)} | Test documents: {len(test.data)}")
print(f"Number of classes: {len(train.target_names)}")


# 2. Build one pipeline per model (same TF-IDF features for a fair comparison)
def make_pipeline(model):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                  max_features=50000, sublinear_tf=True)),
        ("clf", model),
    ])


models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
}

# 3. Train and evaluate every model on the held-out test set
results = {}
for name, model in models.items():
    pipe = make_pipeline(model)
    pipe.fit(train.data, train.target)
    preds = pipe.predict(test.data)
    acc = accuracy_score(test.target, preds)
    f1 = f1_score(test.target, preds, average="macro")
    results[name] = (acc, f1, pipe, preds)
    print(f"{name:20s} accuracy = {acc:.4f} | macro F1 = {f1:.4f}")

# 4. Pick the best model by macro F1 and report details
best_name = max(results, key=lambda k: results[k][1])
best_acc, best_f1, best_pipe, best_preds = results[best_name]
print(f"\nBest model: {best_name} (accuracy {best_acc:.4f}, macro F1 {best_f1:.4f})\n")
print(classification_report(test.target, best_preds, target_names=train.target_names))

# 5. 5-fold cross-validation on the training data to check the result is stable
cv_scores = cross_val_score(make_pipeline(models[best_name]), train.data, train.target,
                            cv=5, scoring="accuracy")
print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 6. Save a confusion matrix image for the README / GitHub
fig, ax = plt.subplots(figsize=(11, 11))
ConfusionMatrixDisplay.from_predictions(test.target, best_preds,
                                        display_labels=train.target_names,
                                        xticks_rotation=90, ax=ax, colorbar=False)
ax.set_title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.png")

# 7. Save the trained model and try a prediction
joblib.dump(best_pipe, "text_classifier.joblib")
sample = ["The new graphics card renders games at higher frame rates than the old one."]
print("Sample prediction:", train.target_names[best_pipe.predict(sample)[0]])
