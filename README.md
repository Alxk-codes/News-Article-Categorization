# News Article Categorization Using NLP

A simple undergraduate-level Natural Language Processing mini-project that automatically classifies news articles into one of five categories using **TF-IDF** feature extraction and **Logistic Regression**.

---

## Introduction

Every day, thousands of news articles are published online. Reading each one and manually sorting it into the correct topic (Sports, Technology, Business, etc.) is slow and error-prone. This project uses Natural Language Processing (NLP) to automatically classify a news article into the correct category in under a second.

The project is built entirely with simple, well-understood tools — no deep learning or complex APIs — so that every step of the pipeline is easy to understand and explain.

---

## Problem Statement

With the explosion of online news, manually organizing large volumes of articles is not scalable. An automated categorization system helps:

- News aggregators organize content for readers
- Search engines index articles by topic
- Journalists find relevant articles faster

---

## Objectives

1. Build an NLP preprocessing pipeline (tokenization, stop-word removal, lemmatization).
2. Convert text into numerical features using TF-IDF.
3. Train a Logistic Regression classifier to predict the news category.
4. Evaluate the model with standard metrics (accuracy, precision, recall, F1-score).
5. Create a clean, interactive Streamlit web application for demonstration.

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.x | Core programming language |
| NLTK | Tokenization, stop-word removal, lemmatization |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | TF-IDF vectorization, Logistic Regression, evaluation |
| Matplotlib | Plotting the confusion matrix |
| Seaborn | Styling the confusion matrix heatmap |
| Streamlit | Web-based user interface |
| Joblib | Saving and loading model files |

---

## NLP Techniques

### Tokenization
Splitting a sentence into individual words or tokens.
> "Apple unveiled a new phone." → ["Apple", "unveiled", "a", "new", "phone", "."]

### Stop-word Removal
Removing common English words that carry little meaning for classification (e.g., "the", "is", "a", "and").

### Lemmatization
Reducing a word to its base form so that variations are treated as the same word.
> "running" → "run" | "companies" → "company" | "better" → "good"

### TF-IDF (Term Frequency – Inverse Document Frequency)
Converts text into a numerical matrix where each word is given a score:
- **TF**: How frequently does the word appear in *this* document?
- **IDF**: How rare is the word across *all* documents?

Words like "touchdown" (common in sports, rare elsewhere) get a high score; words like "said" (common everywhere) get a low score.

---

## Machine Learning Model

### Logistic Regression
Despite its name, Logistic Regression is a *classification* algorithm. It calculates the probability that an article belongs to each category and predicts the category with the highest probability.

It is:
- Fast to train
- Easy to interpret
- Works very well with TF-IDF text features

---

## Workflow

```
Input News Article
        ↓
  Lowercasing → URL Removal → Punctuation Removal
        ↓
    Tokenization (split into words)
        ↓
    Stop-word Removal
        ↓
    Lemmatization
        ↓
  TF-IDF Vectorization (text → numbers)
        ↓
  Logistic Regression (numbers → category)
        ↓
   Predicted Category + Confidence Score
```

---

## Dataset

The project uses the **BBC News dataset** (D. Greene & P. Cunningham, 2006), which contains ~2,225 real news articles across five categories that perfectly match the project requirements:

| Category | Description |
|---|---|
| **Sports** | Football, cricket, tennis, athletics, etc. |
| **Technology** | Gadgets, software, AI, cybersecurity, etc. |
| **Business** | Stock markets, companies, economy, finance |
| **Politics** | Government, elections, policy, international affairs |
| **Entertainment** | Films, music, celebrities, television |

If the BBC dataset cannot be downloaded, the setup script automatically generates a self-contained synthetic dataset.

---

## Model Evaluation

After training, the following metrics are computed on the **held-out test set (20%)**:

| Metric | What it means |
|---|---|
| **Accuracy** | % of articles classified correctly overall |
| **Precision** | Of all articles predicted as category X, how many actually were X? |
| **Recall** | Of all actual category X articles, how many did the model find? |
| **F1-Score** | Harmonic mean of Precision and Recall — balances both |

A **Confusion Matrix** heatmap is also saved to `model/confusion_matrix.png`. Each cell shows how many articles of one category were predicted as another — a perfect model has all values on the diagonal.

> **Note**: All metrics are computed from the actual trained model — no results are hardcoded.

---

## Project Structure

```
News-Article-Categorization/
│
├── dataset/
│   ├── setup_dataset.py    ← Download or generate the dataset
│   └── news.csv            ← Generated by setup_dataset.py
│
├── model/
│   ├── news_classifier.pkl      ← Saved after training
│   ├── tfidf_vectorizer.pkl     ← Saved after training
│   └── confusion_matrix.png     ← Saved after training
│
├── src/
│   └── preprocessing.py    ← NLP pipeline (preprocess_text function)
│
├── train_model.py          ← Full training script
├── app.py                  ← Streamlit web application
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
└── .gitignore
```

---

## Installation

**Prerequisites**: Python 3.8 or higher.

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# OR
source venv/bin/activate   # Mac / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Handle NLTK resources

The preprocessing script downloads all required NLTK data automatically on first run:

```python
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

If you are on a machine without internet access, you can manually download these from [https://www.nltk.org/data.html](https://www.nltk.org/data.html) and place them in your NLTK data directory.

---

## Setup Dataset

```bash
python dataset/setup_dataset.py
```

This will attempt to download the BBC News dataset. If unavailable, it generates a synthetic dataset automatically.

---

## Training

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train the Logistic Regression model
- Print evaluation metrics in the terminal
- Save the confusion matrix image to `model/confusion_matrix.png`
- Save the model files to the `model/` directory

---

## Running the Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Example

**Input:**
> "Apple unveiled its latest MacBook Pro powered by the new M4 chip, promising a 40% improvement in machine learning tasks. The laptop is expected to go on sale next month."

**Output:**
```
Predicted Category: Technology
Confidence: 91.3%
```

---

## Limitations

- Performance depends on the quality and size of the training dataset.
- The model may struggle with articles that mix multiple topics (e.g., a story about a tech company's sports sponsorship).
- The model only predicts the five categories it was trained on — it cannot detect articles outside these categories.
- Short articles (1–2 sentences) may produce lower-confidence predictions.

---

## Future Scope

- Add more categories (e.g., Health, Science, Travel).
- Use a larger dataset for improved accuracy.
- Explore better NLP models such as word embeddings (Word2Vec, GloVe) or pre-trained models (BERT).
- Implement multilingual news classification to support regional language news.
- Add a batch-upload feature to classify multiple articles at once.

---

## References

- D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering". Proc. ICML 2006. [Link](http://mlg.ucd.ie/datasets/bbc.html)
- Scikit-learn documentation: [https://scikit-learn.org](https://scikit-learn.org)
- NLTK documentation: [https://www.nltk.org](https://www.nltk.org)
- Streamlit documentation: [https://docs.streamlit.io](https://docs.streamlit.io)
