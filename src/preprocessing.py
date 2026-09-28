"""
preprocessing.py
----------------
Reusable NLP text-preprocessing pipeline for the News Article Categorization
project.

Pipeline steps:
  1. Convert text to lowercase
  2. Remove URLs
  3. Remove punctuation and special characters
  4. Remove digits / numbers
  5. Tokenize (split into individual words)
  6. Remove English stop-words  (using NLTK)
  7. Lemmatize each token        (using NLTK WordNetLemmatizer)
  8. Rejoin tokens into a clean string

Usage:
    from src.preprocessing import preprocess_text
    clean = preprocess_text("Apple Inc. unveiled its latest iPhone model today.")
"""

import re
import nltk

# ── Download required NLTK resources (safe to call multiple times) ─────────────
# These will only download if not already present on the machine.
nltk.download("punkt",        quiet=True)   # tokenizer models
nltk.download("punkt_tab",    quiet=True)   # updated tokenizer data (NLTK ≥ 3.8)
nltk.download("stopwords",    quiet=True)   # English stop-word list
nltk.download("wordnet",      quiet=True)   # WordNet lexical database for lemmatizer
nltk.download("omw-1.4",      quiet=True)   # Open Multilingual WordNet (required by wordnet)

from nltk.corpus   import stopwords
from nltk.stem     import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── Initialise tools once (not inside the function to avoid repeated overhead) ─
STOP_WORDS  = set(stopwords.words("english"))
LEMMATIZER  = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    Clean and normalise a raw news article string.

    Parameters
    ----------
    text : str
        Raw input text (a full news article or snippet).

    Returns
    -------
    str
        A single space-separated string of clean, lemmatized tokens.
        Returns an empty string if the input is empty or not a string.
    """
    # ── Guard: handle None or non-string input ─────────────────────────────────
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1 – Lowercase
    text = text.lower()

    # Step 2 – Remove URLs  (http://... or https://... or www....)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Step 3 – Remove punctuation and special characters, keep only letters & spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Step 4 – Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()

    # Step 5 – Tokenize into a list of individual words
    tokens = word_tokenize(text)

    # Step 6 – Remove stop-words and very short tokens (length ≤ 1)
    tokens = [
        token for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]

    # Step 7 – Lemmatize: reduce each word to its base/root form
    #          e.g. "running" → "run", "companies" → "company"
    tokens = [LEMMATIZER.lemmatize(token) for token in tokens]

    # Step 8 – Rejoin into a single clean string
    return " ".join(tokens)


# ── Quick test when run directly ───────────────────────────────────────────────
if __name__ == "__main__":
    sample = (
        "Apple Inc. unveiled its latest iPhone model today, featuring "
        "a new AI-powered chip. The company's stock rose 3.5% after "
        "the announcement. Visit https://apple.com for more details."
    )
    print("Original :", sample)
    print("Processed:", preprocess_text(sample))
