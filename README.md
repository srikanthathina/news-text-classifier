# News Article Text Classifier (NLP + Machine Learning)

A text classification project that sorts news posts into 20 topics using TF-IDF features and compares three machine learning models: Multinomial Naive Bayes, Logistic Regression and Linear SVM.

## Tech Stack
Python, Scikit-learn, TF-IDF, NLP, Matplotlib, Joblib, Google Colab

## Dataset
20 Newsgroups (built into Scikit-learn): 11,314 training and 7,532 test documents across 20 topics. Headers, footers and quoted replies are removed so the model learns from the text itself.

## Approach
1. Convert text to TF-IDF features (unigrams and bigrams, English stop words removed)
2. Train three models with the same features for a fair comparison
3. Evaluate on the held-out test set using accuracy and macro F1-score
4. Check stability with 5-fold cross-validation
5. Save the best model and a confusion matrix

## Results

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Naive Bayes | 0.7011 | 0.6792 |
| Logistic Regression | 0.6958 | 0.6807 |
| Linear SVM | 0.6945 | 0.6835 |

Best model by macro F1: Linear SVM. 5-fold cross-validation accuracy: 0.7629 (+/- 0.0046).

Note: the test accuracy is lower than the cross-validation accuracy because this dataset's train/test split is by date, so the test set is slightly different data and harder to predict. The weakest classes were talk.religion.misc, alt.atheism and talk.politics.misc, which overlap heavily in vocabulary with related topics.

## How to Run
```
pip install scikit-learn matplotlib joblib
python text_classifier.py
```

## Files
- `text_classifier.py` - training and evaluation script
- `confusion_matrix.png` - confusion matrix of the best model
