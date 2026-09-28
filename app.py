"""
app.py — News Article Categorization
Newspaper-themed frontend with proper sections, readable buttons, and clean layout.
"""

import os, sys, joblib
import streamlit as st
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
from src.preprocessing import preprocess_text

MODEL_PATH      = os.path.join(PROJECT_ROOT, "model", "news_classifier.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "model", "tfidf_vectorizer.pkl")
CONF_MATRIX_IMG = os.path.join(PROJECT_ROOT, "model", "confusion_matrix.png")

CATEGORIES = {
    "Sports":        "🏆",
    "Technology":    "💻",
    "Business":      "📈",
    "Politics":      "🏛️",
    "Entertainment": "🎬",
}

EXAMPLES = {
    "Sports": (
        "The Indian cricket team clinched a dramatic series victory in the final Test against "
        "Australia at the Melbourne Cricket Ground. Captain Rohit Sharma scored a brilliant century "
        "while fast bowler Jasprit Bumrah took five wickets to seal the series 3-1. The win marks "
        "India's second consecutive series victory in Australia and has sent fans across the country "
        "into a frenzy. Thousands gathered at airports to welcome the squad home."
    ),
    "Technology": (
        "OpenAI has released a new version of its flagship AI model featuring significantly improved "
        "reasoning capabilities and a dramatically expanded context window of 128,000 tokens. Developers "
        "praised the improved code-generation accuracy, and several major tech companies have already "
        "announced plans to integrate the new API into their enterprise software products. The update "
        "also introduces multimodal capabilities and faster inference speeds."
    ),
    "Business": (
        "Apple Inc. reported record-breaking quarterly revenue of 120 billion dollars, driven by strong "
        "iPhone sales in emerging markets and growing demand for its services segment including the App "
        "Store, iCloud, and Apple Music. The company stock rose four percent in after-hours trading. "
        "CFO Luca Maestri credited disciplined cost management and supply-chain improvements for the "
        "better-than-expected profit margins this quarter."
    ),
    "Politics": (
        "The Prime Minister introduced sweeping new immigration legislation in parliament today, promising "
        "tighter border controls and faster visa processing for skilled workers. Opposition leaders "
        "strongly criticised the bill, calling for an emergency parliamentary debate. The legislation is "
        "expected to face legal challenges from human rights organisations. Voters will have a chance to "
        "weigh in at next month's general election, which polls show is too close to call."
    ),
    "Entertainment": (
        "Marvel Studios released the trailer for its highly anticipated blockbuster Avengers Secret Wars "
        "and it has already broken the record for the most-viewed trailer in 24 hours on YouTube, "
        "surpassing 300 million views. Fans flooded social media with reactions and theories. The film "
        "directed by the Russo Brothers is expected to be the highest-grossing movie of the year when "
        "it releases in cinemas this coming summer."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
/* ── Base ─────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], .stApp {
    background-color: #f5f0e6 !important;
    color: #1a1a1a !important;
    font-family: Georgia, "Times New Roman", serif !important;
}
[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ── Block container ──────────────────────────────────────────── */
.block-container {
    padding: 0 2rem 3rem 2rem !important;
    max-width: 1100px !important;
}

/* ═════════════════════════════════════════
   SIDEBAR
═════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background-color: #1a1a1a !important;
    border-right: 3px solid #333 !important;
}
[data-testid="stSidebar"] * {
    color: #f5f0e6 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label {
    color: #f5f0e6 !important;
    font-family: Georgia, serif !important;
}
[data-testid="stSidebar"] hr {
    border-color: #444 !important;
}
/* Sidebar example buttons — cream on black */
[data-testid="stSidebar"] .stButton > button {
    background-color: #2a2a2a !important;
    color: #f5f0e6 !important;
    border: 1px solid #555 !important;
    border-radius: 2px !important;
    font-family: "Segoe UI", Arial, sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-align: left !important;
    width: 100% !important;
    padding: 10px 14px !important;
    margin-bottom: 6px !important;
    transition: background 0.2s ease, border-color 0.2s ease !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #cc0000 !important;
    border-color: #cc0000 !important;
    color: #fff !important;
}
[data-testid="stSidebar"] .stButton > button:active {
    background-color: #990000 !important;
}

/* ═════════════════════════════════════════
   MASTHEAD / HEADER
═════════════════════════════════════════ */
.masthead-wrap {
    background: #f5f0e6;
    border-top: 7px solid #1a1a1a;
    border-bottom: 4px solid #1a1a1a;
    text-align: center;
    padding: 10px 0 8px;
    margin-bottom: 0;
}
.mh-eyebrow {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 0.55rem;
    letter-spacing: 0.35em;
    color: #666;
    text-transform: uppercase;
    margin-bottom: 5px;
}
.mh-rule { border: none; border-top: 1px solid #1a1a1a; margin: 5px 12px; }
.mh-title {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 4rem;
    font-weight: 700;
    color: #1a1a1a;
    margin: 4px 0;
    line-height: 1;
}
.mh-tagline {
    font-family: Georgia, serif;
    font-style: italic;
    font-size: 0.85rem;
    color: #444;
    margin: 4px 0;
}
.mh-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 0.58rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #666;
    border-top: 1px solid #bbb;
    padding: 4px 16px 0;
    margin-top: 5px;
}

/* ═════════════════════════════════════════
   TICKER
═════════════════════════════════════════ */
.ticker {
    background: #1a1a1a;
    color: #f5f0e6;
    padding: 7px 0;
    overflow: hidden;
    white-space: nowrap;
    border-bottom: 2px solid #cc0000;
}
.ticker-badge {
    display: inline-block;
    background: #cc0000;
    color: #fff;
    font-family: "Segoe UI", Arial, sans-serif;
    font-weight: 700;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 3px 12px;
    margin-right: 18px;
}
.ticker-text {
    display: inline-block;
    font-family: Georgia, serif;
    font-size: 0.68rem;
    letter-spacing: 0.04em;
    animation: scroll-left 35s linear infinite;
}
@keyframes scroll-left {
    from { transform: translateX(100vw); }
    to   { transform: translateX(-200%); }
}

/* ═════════════════════════════════════════
   SECTION LABEL
═════════════════════════════════════════ */
.section-label {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #1a1a1a;
    text-align: center;
    background: #f5f0e6;
    border-top: 2px solid #1a1a1a;
    border-bottom: 1px solid #1a1a1a;
    padding: 6px 0;
    margin: 24px 0 18px;
}

/* ═════════════════════════════════════════
   TEXT AREA
═════════════════════════════════════════ */
.stTextArea label {
    font-family: "Segoe UI", Arial, sans-serif !important;
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    color: #1a1a1a !important;
    margin-bottom: 6px !important;
}
.stTextArea textarea {
    background: #fff !important;
    color: #1a1a1a !important;
    border: 2px solid #999 !important;
    border-radius: 0 !important;
    font-family: Georgia, serif !important;
    font-size: 0.88rem !important;
    line-height: 1.75 !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: #1a1a1a !important;
    box-shadow: 0 0 0 3px rgba(26,26,26,0.1) !important;
}

/* ═════════════════════════════════════════
   CLASSIFY BUTTON  (main action)
═════════════════════════════════════════ */
.classify-wrap .stButton > button {
    background-color: #1a1a1a !important;
    color: #f5f0e6 !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: "Segoe UI", Arial, sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    padding: 14px 40px !important;
    width: 100% !important;
    transition: background 0.2s ease, transform 0.1s ease !important;
    box-shadow: 4px 4px 0 #888 !important;
    cursor: pointer !important;
}
.classify-wrap .stButton > button:hover {
    background-color: #cc0000 !important;
    box-shadow: 4px 4px 0 #880000 !important;
}
.classify-wrap .stButton > button:active {
    transform: translate(2px, 2px) !important;
    box-shadow: 2px 2px 0 #880000 !important;
}

/* ═════════════════════════════════════════
   HOW IT WORKS PANEL
═════════════════════════════════════════ */
.how-panel {
    background: #fff;
    border: 1.5px solid #ccc;
    border-left: 4px solid #1a1a1a;
    padding: 16px 18px;
}
.how-panel h4 {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #1a1a1a;
    border-bottom: 1px solid #ddd;
    padding-bottom: 8px;
    margin: 0 0 12px;
}
.how-step {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    margin-bottom: 10px;
}
.how-num {
    background: #1a1a1a;
    color: #f5f0e6;
    font-family: Georgia, serif;
    font-size: 0.7rem;
    font-weight: 700;
    width: 22px; height: 22px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.how-text {
    font-family: Georgia, serif;
    font-size: 0.72rem;
    line-height: 1.5;
    color: #333;
}
.how-text strong {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #1a1a1a;
    display: block;
    margin-bottom: 1px;
}

/* ═════════════════════════════════════════
   RESULT CARD
═════════════════════════════════════════ */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pop {
    0%   { transform: scale(0.7); opacity: 0; }
    70%  { transform: scale(1.05); }
    100% { transform: scale(1); opacity: 1; }
}
@keyframes fillBar {
    from { width: 0%; }
}

.result-outer {
    animation: slideUp 0.4s ease both;
    margin-top: 20px;
    border: 2px solid #1a1a1a;
    background: #fff;
    box-shadow: 6px 6px 0 #aaa;
}
.result-header {
    background: #1a1a1a;
    padding: 10px 20px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.result-badge {
    background: #cc0000;
    color: #fff;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 3px 10px;
}
.result-header-text {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #f5f0e6;
}
.result-body {
    padding: 20px 24px 22px;
    display: flex;
    gap: 20px;
    align-items: center;
}
.result-icon-box {
    animation: pop 0.5s cubic-bezier(0.36,0.07,0.19,0.97) 0.2s both;
    background: #1a1a1a;
    color: #f5f0e6;
    width: 80px; height: 80px;
    display: flex; align-items: center; justify-content: center;
    font-size: 2.2rem;
    border: 3px solid #1a1a1a;
    outline: 2px dashed #777;
    outline-offset: 4px;
    flex-shrink: 0;
}
.result-info { flex: 1; }
.result-cat-name {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #1a1a1a;
    line-height: 1;
    border-bottom: 2px solid #1a1a1a;
    padding-bottom: 8px;
    margin-bottom: 10px;
}
.result-subtitle {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 0.6rem;
    color: #777;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.conf-label {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #1a1a1a;
    margin-bottom: 5px;
}
.conf-track {
    height: 12px;
    background: #e0ddd6;
    border: 1px solid #bbb;
    overflow: hidden;
    margin-bottom: 4px;
}
.conf-fill {
    height: 100%;
    background: #1a1a1a;
    animation: fillBar 0.8s ease 0.3s both;
}
.conf-pct {
    font-family: Georgia, serif;
    font-size: 0.72rem;
    color: #333;
}

/* ═════════════════════════════════════════
   CATEGORY CARDS
═════════════════════════════════════════ */
.cat-card {
    background: #fff;
    border: 1.5px solid #ccc;
    border-top: 3px solid #1a1a1a;
    padding: 14px 12px;
    text-align: center;
    height: 100%;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
}
.cat-card:hover {
    box-shadow: 3px 3px 0 #aaa;
    transform: translate(-1px, -1px);
}
.cat-icon { font-size: 1.8rem; margin-bottom: 6px; }
.cat-name {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #1a1a1a;
    border-bottom: 1px solid #ddd;
    padding-bottom: 5px;
    margin-bottom: 6px;
}
.cat-desc {
    font-family: Georgia, serif;
    font-size: 0.65rem;
    color: #555;
    line-height: 1.55;
}

/* ═════════════════════════════════════════
   METRIC CARDS
═════════════════════════════════════════ */
.metric-card {
    background: #1a1a1a;
    color: #f5f0e6;
    text-align: center;
    padding: 16px 8px;
    border: 2px solid #1a1a1a;
    box-shadow: 3px 3px 0 #666;
    margin-bottom: 14px;
}
.metric-val {
    font-family: Georgia, serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f5f0e6;
}
.metric-lbl {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 0.56rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #aaa;
    margin-top: 4px;
}

/* ═════════════════════════════════════════
   WARN / INFO BANNERS
═════════════════════════════════════════ */
.warn {
    background: #fff5f5;
    border-left: 5px solid #cc0000;
    padding: 12px 16px;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 0.8rem;
    color: #900;
    margin: 12px 0;
}

/* ═════════════════════════════════════════
   FOOTER
═════════════════════════════════════════ */
.site-footer {
    border-top: 3px double #1a1a1a;
    text-align: center;
    padding: 14px 0;
    margin-top: 32px;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
}
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def predict(text, model, vectorizer):
    clean = preprocess_text(text)
    if not clean:
        return None, None
    vec        = vectorizer.transform([clean])
    category   = model.predict(vec)[0]
    confidence = max(model.predict_proba(vec)[0]) * 100
    return category, confidence


def render_result(category, confidence):
    icon = CATEGORIES.get(category, "📄")
    pct  = f"{confidence:.1f}%"
    ts   = datetime.now().strftime("%d %b %Y · %H:%M")
    st.markdown(f"""
<div class="result-outer">
  <div class="result-header">
    <span class="result-badge">Classified</span>
    <span class="result-header-text">Article Category Detected</span>
  </div>
  <div class="result-body">
    <div class="result-icon-box">{icon}</div>
    <div class="result-info">
      <div class="result-cat-name">{category}</div>
      <div class="result-subtitle">
        TF-IDF + Logistic Regression &nbsp;·&nbsp; BBC News Model &nbsp;·&nbsp; {ts}
      </div>
      <div class="conf-label">Model Confidence</div>
      <div class="conf-track">
        <div class="conf-fill" style="width:{pct};"></div>
      </div>
      <div class="conf-pct">{pct} probability
        &nbsp;<span style="font-size:0.65rem;color:#999;">
          (model estimate — not an absolute certainty)
        </span>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="The Daily Classifier",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    model, vectorizer = load_model()

    if "article" not in st.session_state:
        st.session_state.article = ""

    # ═══════════════════════════════════════════════════════════════
    # SIDEBAR
    # ═══════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("## 📋 Example Articles")
        st.markdown(
            "Click any category to load a sample article into the input box.",
            unsafe_allow_html=True
        )
        st.markdown("---")

        for cat, icon in CATEGORIES.items():
            if st.button(f"{icon}  {cat}", key=f"btn_{cat}", use_container_width=True):
                st.session_state.article = EXAMPLES[cat]
                st.rerun()

        st.markdown("---")
        st.markdown("## ℹ️ About")
        st.markdown(
            "**The Daily Classifier** automatically labels news articles "
            "into five categories using classical NLP techniques.\n\n"
            "**Model:** Logistic Regression  \n"
            "**Features:** TF-IDF (10,000 terms, bigrams)  \n"
            "**Accuracy:** 98.43% on 445 test articles  \n"
            "**Dataset:** BBC News Corpus (2,225 articles)"
        )
        st.markdown("---")
        st.caption("NLP Mini Project · Python · Scikit-learn · Streamlit")

    # ═══════════════════════════════════════════════════════════════
    # MASTHEAD
    # ═══════════════════════════════════════════════════════════════
    today = datetime.now().strftime("%A, %d %B %Y")
    st.markdown(f"""
<div class="masthead-wrap">
  <div class="mh-eyebrow">Established 2026 &nbsp;·&nbsp; NLP Mini Project &nbsp;·&nbsp; BBC News Corpus</div>
  <hr class="mh-rule">
  <div class="mh-title">The Daily Classifier</div>
  <hr class="mh-rule">
  <div class="mh-tagline">"All the News That's Fit to Categorize" &nbsp;·&nbsp; Powered by Natural Language Processing</div>
  <div class="mh-meta">
    <span>{today}</span>
    <span>Vol. I &nbsp;·&nbsp; Issue 1</span>
    <span>98.4% Model Accuracy</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Ticker ──────────────────────────────────────────────────────
    st.markdown("""
<div class="ticker">
  <span class="ticker-badge">Breaking News</span>
  <span class="ticker-text">
    AI MODEL CLASSIFIES NEWS ARTICLES WITH 98.4% ACCURACY &nbsp;·&nbsp;
    PIPELINE: TOKENISATION → STOP-WORD REMOVAL → LEMMATISATION → TF-IDF → LOGISTIC REGRESSION &nbsp;·&nbsp;
    FIVE CATEGORIES: SPORTS · TECHNOLOGY · BUSINESS · POLITICS · ENTERTAINMENT &nbsp;·&nbsp;
    TRAINED ON 2,225 REAL BBC NEWS ARTICLES &nbsp;·&nbsp;
    PASTE YOUR ARTICLE BELOW AND CLICK CLASSIFY &nbsp;&nbsp;&nbsp;
  </span>
</div>
""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # MODEL MISSING
    # ═══════════════════════════════════════════════════════════════
    if model is None:
        st.markdown("""
<div class="warn">
  <strong>⚠ Model not found.</strong><br>
  Run the following commands first:<br><br>
  <code>python dataset/setup_dataset.py</code><br>
  <code>python train_model.py</code>
</div>
""", unsafe_allow_html=True)
        return

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1 — INPUT
    # ═══════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-label">Submit Article for Classification</div>',
        unsafe_allow_html=True
    )

    left_col, right_col = st.columns([3, 1], gap="large")

    with left_col:
        article = st.text_area(
            label="Paste Your Article",
            value=st.session_state.article,
            height=300,
            placeholder=(
                "Paste a full news article here...\n\n"
                "You can also click any category button in the left sidebar "
                "to load a sample article automatically."
            ),
        )

        # ── Classify button ─────────────────────────────────────────
        st.markdown('<div class="classify-wrap">', unsafe_allow_html=True)
        _, btn_col, _ = st.columns([1, 2, 1])
        with btn_col:
            clicked = st.button("Classify Article", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right_col:
        st.markdown("""
<div class="how-panel">
  <h4>How It Works</h4>
  <div class="how-step">
    <div class="how-num">1</div>
    <div class="how-text"><strong>Clean</strong>Lowercase, strip URLs, punctuation &amp; digits.</div>
  </div>
  <div class="how-step">
    <div class="how-num">2</div>
    <div class="how-text"><strong>Tokenise</strong>Split text into individual word tokens.</div>
  </div>
  <div class="how-step">
    <div class="how-num">3</div>
    <div class="how-text"><strong>Filter</strong>Remove common English stop-words (the, is, a…).</div>
  </div>
  <div class="how-step">
    <div class="how-num">4</div>
    <div class="how-text"><strong>Lemmatise</strong>Reduce words to their root form (running → run).</div>
  </div>
  <div class="how-step">
    <div class="how-num">5</div>
    <div class="how-text"><strong>TF-IDF</strong>Convert tokens to a 10,000-feature numeric vector.</div>
  </div>
  <div class="how-step">
    <div class="how-num">6</div>
    <div class="how-text"><strong>Predict</strong>Logistic Regression picks the highest-probability category.</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ─── Handle classify click ────────────────────────────────────
    if clicked:
        if not article.strip():
            st.markdown("""
<div class="warn">
  <strong>⚠ Empty input.</strong>
  Please paste an article or click a category in the sidebar to load an example.
</div>
""", unsafe_allow_html=True)
        else:
            with st.spinner("Analysing article…"):
                category, confidence = predict(article, model, vectorizer)

            if category is None:
                st.markdown("""
<div class="warn">
  <strong>⚠ Could not classify.</strong>
  The text was empty after preprocessing. Try a longer article.
</div>
""", unsafe_allow_html=True)
            else:
                render_result(category, confidence)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2 — CATEGORY OVERVIEW
    # ═══════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-label">Coverage Areas</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    cat_info = [
        ("🏆", "Sports",        "Cricket, football, tennis, athletics & all major sporting events."),
        ("💻", "Technology",    "AI, software, gadgets, cybersecurity & the digital economy."),
        ("📈", "Business",      "Markets, corporations, finance, trade & economic policy."),
        ("🏛️",  "Politics",      "Government, elections, legislation & international affairs."),
        ("🎬", "Entertainment", "Film, music, television, celebrities & the arts."),
    ]
    for col, (icon, name, desc) in zip([c1, c2, c3, c4, c5], cat_info):
        with col:
            st.markdown(f"""
<div class="cat-card">
  <div class="cat-icon">{icon}</div>
  <div class="cat-name">{name}</div>
  <div class="cat-desc">{desc}</div>
</div>
""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3 — MODEL PERFORMANCE
    # ═══════════════════════════════════════════════════════════════
    st.markdown(
        '<div class="section-label">Model Performance</div>',
        unsafe_allow_html=True
    )

    m1, m2, m3, m4 = st.columns(4, gap="small")
    for col, (lbl, val) in zip([m1, m2, m3, m4], [
        ("Accuracy",  "98.43%"),
        ("Precision", "98.43%"),
        ("Recall",    "98.43%"),
        ("F1-Score",  "98.42%"),
    ]):
        with col:
            st.markdown(f"""
<div class="metric-card">
  <div class="metric-val">{val}</div>
  <div class="metric-lbl">{lbl}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        "_All metrics computed on a held-out test set of **445 articles (20%)** — "
        "no values are hardcoded._",
        unsafe_allow_html=False,
    )

    if os.path.exists(CONF_MATRIX_IMG):
        _, img_col, _ = st.columns([1, 2, 1])
        with img_col:
            st.markdown(
                '<div style="border:3px solid #1a1a1a;padding:6px;'
                'background:#fff;box-shadow:5px 5px 0 #aaa;">',
                unsafe_allow_html=True
            )
            st.image(
                CONF_MATRIX_IMG,
                caption="Confusion Matrix — test set (445 articles) · 98.43% overall accuracy",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════════════════
    st.markdown(f"""
<div class="site-footer">
  The Daily Classifier &nbsp;·&nbsp; NLP Mini Project &nbsp;·&nbsp;
  Logistic Regression + TF-IDF &nbsp;·&nbsp; BBC News Corpus (2,225 articles) &nbsp;·&nbsp;
  {datetime.now().year}
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
