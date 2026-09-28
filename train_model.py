"""
train_model.py
--------------
Training script for the News Article Categorization project.

What this script does:
  1.  Load the news dataset (dataset/news.csv)
  2.  Inspect the data (shape, category distribution)
  3.  Handle missing values
  4.  Preprocess each article with the NLP pipeline
  5.  Split into training and testing sets (80 / 20)
  6.  Convert text to TF-IDF feature vectors
  7.  Train a Logistic Regression classifier
  8.  Evaluate: accuracy, precision, recall, F1-score, confusion matrix
  9.  Save the trained model and TF-IDF vectorizer to the model/ directory

Run this script once before launching the Streamlit app:
    python train_model.py
"""

import os
import sys
import joblib
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model    import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ── Import our custom preprocessing pipeline ───────────────────────────────────
# Add the project root to sys.path so the import works from any location
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from src.preprocessing import preprocess_text

# ── File paths ─────────────────────────────────────────────────────────────────
DATASET_PATH     = os.path.join(PROJECT_ROOT, "dataset", "news.csv")
MODEL_DIR        = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH       = os.path.join(MODEL_DIR, "news_classifier.pkl")
VECTORIZER_PATH  = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
CONF_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")

# Category order (used for consistent display)
CATEGORIES = ["Business", "Entertainment", "Politics", "Sports", "Technology"]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Load Dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(path):
    """Load the CSV dataset and return a DataFrame."""
    if not os.path.exists(path):
        print(
            f"\n[ERROR] Dataset not found at '{path}'.\n"
            "Please run first:  python dataset/setup_dataset.py\n"
        )
        sys.exit(1)

    df = pd.read_csv(path)
    print(f"[1] Dataset loaded — {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Inspect and Clean
# ══════════════════════════════════════════════════════════════════════════════

def inspect_and_clean(df):
    """Print basic stats and handle missing values."""
    print(f"\n[2] Columns  : {list(df.columns)}")
    print(f"    Categories:\n{df['category'].value_counts().to_string()}")

    # Drop rows where text or category is missing
    before = len(df)
    df = df.dropna(subset=["text", "category"])
    df = df[df["text"].str.strip() != ""]
    after = len(df)

    if before != after:
        print(f"    Removed {before - after} rows with missing/empty text.")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Preprocess Text
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_dataset(df):
    """Apply the NLP preprocessing pipeline to every article."""
    print("\n[3] Preprocessing text … (this may take a moment)")
    df = df.copy()
    df["clean_text"] = df["text"].apply(preprocess_text)

    # Drop rows where preprocessing produces an empty string
    df = df[df["clean_text"].str.strip() != ""]
    print(f"    Done. {len(df)} articles remain after preprocessing.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Train / Test Split
# ══════════════════════════════════════════════════════════════════════════════

def split_data(df):
    """Split the dataset into 80% training and 20% testing."""
    X = df["clean_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,   # fixed seed for reproducibility
        stratify=y,        # keep category proportions equal in both sets
    )

    print(f"\n[4] Train / Test split — Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — TF-IDF Vectorization
# ══════════════════════════════════════════════════════════════════════════════

def vectorize(X_train, X_test):
    """
    Convert clean text into TF-IDF feature vectors.

    TF-IDF (Term Frequency – Inverse Document Frequency) measures how
    important a word is to a document relative to the whole corpus.
    Words common in all articles (e.g., 'said') get a low score;
    distinctive words (e.g., 'touchdown', 'parliament') get a high score.
    """
    vectorizer = TfidfVectorizer(
        max_features=10_000,   # use at most the top 10,000 most common words
        ngram_range=(1, 2),    # consider single words AND two-word phrases
        sublinear_tf=True,     # apply log normalisation to term frequencies
    )

    X_train_vec = vectorizer.fit_transform(X_train)   # fit on train, transform train
    X_test_vec  = vectorizer.transform(X_test)         # transform test (no re-fitting)

    print(f"[5] TF-IDF features — Train matrix: {X_train_vec.shape}, "
          f"Test matrix: {X_test_vec.shape}")

    return vectorizer, X_train_vec, X_test_vec


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Train Logistic Regression
# ══════════════════════════════════════════════════════════════════════════════

def train_model(X_train_vec, y_train):
    """
    Train a Logistic Regression classifier.

    Logistic Regression predicts the probability of each category and
    chooses the one with the highest probability. It works very well with
    TF-IDF features for text classification tasks.
    """
    print("\n[6] Training Logistic Regression …")

    model = LogisticRegression(
        max_iter=1000,     # allow enough iterations to converge
        random_state=42,
        C=1.0,             # regularisation strength (default; 1.0 is a good start)
        solver="lbfgs",    # efficient optimiser for multi-class problems
    )
    model.fit(X_train_vec, y_train)
    print("    Training complete.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Evaluate
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test_vec, y_test):
    """
    Evaluate the trained model on the held-out test set.
    Prints accuracy, precision, recall, F1-score, and classification report.
    """
    y_pred = model.predict(X_test_vec)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n[7] -- Evaluation Results ------------------------------------------")
    print(f"    Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"    Precision : {precision:.4f}")
    print(f"    Recall    : {recall:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print("\n    Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return y_pred


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Plot Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_test, y_pred, labels, save_path):
    """
    Generate and save a colour-coded confusion matrix heatmap.
    Each row = actual class, each column = predicted class.
    """
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,          # write the count inside each cell
        fmt="d",             # format as integer
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Confusion Matrix — News Article Categorization", fontsize=14)
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[8] Confusion matrix saved to: {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Save Model and Vectorizer
# ══════════════════════════════════════════════════════════════════════════════

def save_artifacts(model, vectorizer):
    """Save the trained model and TF-IDF vectorizer as .pkl files."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,      MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"[9] Model saved      : {MODEL_PATH}")
    print(f"    Vectorizer saved : {VECTORIZER_PATH}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  News Article Categorization — Model Training")
    print("=" * 60)

    df                          = load_dataset(DATASET_PATH)
    df                          = inspect_and_clean(df)
    df                          = preprocess_dataset(df)
    X_train, X_test, y_train, y_test = split_data(df)
    vectorizer, X_train_vec, X_test_vec = vectorize(X_train, X_test)
    model                       = train_model(X_train_vec, y_train)
    y_pred                      = evaluate_model(model, X_test_vec, y_test)

    # Determine which category labels are actually present in the test set
    present_labels = sorted(y_test.unique().tolist())
    plot_confusion_matrix(y_test, y_pred, present_labels, CONF_MATRIX_PATH)

    save_artifacts(model, vectorizer)

    print("\n" + "=" * 60)
    print("  Training complete! Run the app with:")
    print("  streamlit run app.py")
    print("=" * 60)
