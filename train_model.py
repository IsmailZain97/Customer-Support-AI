import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
)

from database import fetch_all_tickets


# ─── Artefact file paths ──────────────────────────────────────────────────────
MODEL_PATH      = "sentiment_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
METRICS_PATH    = "model_metrics.pkl"
CM_IMAGE_PATH   = "confusion_matrix.png"


# ─── Training ─────────────────────────────────────────────────────────────────
def train_sentiment_model() -> dict:
    print("Fetching data from PostgreSQL ...")
    df = fetch_all_tickets()

    if df is None or df.empty:
        raise RuntimeError("No data found. Run data_generator.py first.")

  
    df = df[df["rating"] != 3].copy()
    df["label"] = (df["rating"] >= 4).astype(int)   # 1 = POSITIVE, 0 = NEGATIVE

    print(f"Training set: {len(df)} rows  (after dropping neutral rating=3 rows)")
    print(f"Class split — NEGATIVE: {(df['label']==0).sum()}  |  POSITIVE: {(df['label']==1).sum()}")

    X = df["issue_description"]
    y = df["label"]

    # ── Train / Test Split (80 / 20, stratified) ──────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Vectorisation ──────────────────────────────────────────────────────────
    # ngram_range=(1, 2) means the model also sees two-word phrases
    # like "never received", "amazing service", "still waiting".
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        ngram_range=(1, 2),
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # ── Train Logistic Regression ──────────────────────────────────────────────
    
    print("\nTraining Logistic Regression ...")
    model = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    model.fit(X_train_vec, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred  = model.predict(X_test_vec)
    # predict_proba returns [prob_negative, prob_positive]; take column 1 for ROC-AUC
    y_proba = model.predict_proba(X_test_vec)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc  = roc_auc_score(y_test, y_proba)

   
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring="accuracy")

    print(f"\n{'─'*45}")
    print(f"  Test Accuracy   : {accuracy * 100:.2f}%")
    print(f"  ROC-AUC Score   : {roc_auc:.4f}")
    print(f"  CV Accuracy     : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"{'─'*45}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["NEGATIVE", "POSITIVE"]))


    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["NEGATIVE", "POSITIVE"],
        yticklabels=["NEGATIVE", "POSITIVE"],
    )
    plt.title("Sentiment Model — Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(CM_IMAGE_PATH, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {CM_IMAGE_PATH}")


    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    metrics = {
        "accuracy": accuracy,
        "roc_auc":  roc_auc,
        "cv_mean":  float(cv_scores.mean()),
        "cv_std":   float(cv_scores.std()),
    }
    with open(METRICS_PATH, "wb") as f:
        pickle.dump(metrics, f)

    print(f"Artefacts saved → {MODEL_PATH}, {VECTORIZER_PATH}, {METRICS_PATH}")
    return metrics


def predict_sentiment(text: str, _model=None, _vectorizer=None) -> dict:

    if _model is None:
        with open(MODEL_PATH, "rb") as f:
            _model = pickle.load(f)
    if _vectorizer is None:
        with open(VECTORIZER_PATH, "rb") as f:
            _vectorizer = pickle.load(f)

    vec        = _vectorizer.transform([text])
    label_id   = int(_model.predict(vec)[0])
    proba      = _model.predict_proba(vec)[0]      # [neg_prob, pos_prob]

    label = "POSITIVE" if label_id == 1 else "NEGATIVE"

    return {
        "label":         label,
        "confidence":    round(float(proba[label_id]), 4),
        "positive_prob": round(float(proba[1]), 4),
        "negative_prob": round(float(proba[0]), 4),
    }


if __name__ == "__main__":
    train_sentiment_model()
