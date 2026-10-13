
import streamlit as st
import joblib
import re
import string
from pathlib import Path

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# ---------- Styling ----------
st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 0;
}
.subtitle {
    text-align: center;
    color: #666;
    font-size: 17px;
    margin-bottom: 30px;
}
.result {
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    margin-top: 20px;
}
.real {
    background: #e8f7ee;
    border: 2px solid #35a866;
}
.fake {
    background: #fdecec;
    border: 2px solid #e05252;
}
.metric-box {
    padding: 14px;
    border-radius: 12px;
    background: #f5f5f5;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ---------- Text cleaning ----------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# ---------- Load trained files ----------
BASE_DIR = Path(__file__).parent
model_path = BASE_DIR / "model.jb"
vectorizer_path = BASE_DIR / "vectorizer.jb"

if not model_path.exists() or not vectorizer_path.exists():
    st.error(
        "Model files not found. Put model.jb and vectorizer.jb in the same folder as app.py."
    )
    st.stop()

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# ---------- UI ----------
st.markdown('<div class="main-title">📰 Fake News Detector</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered news classification using TF-IDF & Logistic Regression</div>',
    unsafe_allow_html=True
)

news_text = st.text_area(
    "Enter News Article",
    height=220,
    placeholder="Paste the news article or news text here..."
)

col1, col2 = st.columns([1, 1])

with col1:
    analyze = st.button("🔍 Analyze News", use_container_width=True)

with col2:
    clear = st.button("🗑️ Clear", use_container_width=True)

if clear:
    st.rerun()

if analyze:
    if not news_text.strip():
        st.warning("Please enter some news text first.")
    else:
        cleaned = clean_text(news_text)
        transformed = vectorizer.transform([cleaned])

        prediction = model.predict(transformed)[0]
        probabilities = model.predict_proba(transformed)[0]

        fake_probability = probabilities[0] * 100
        real_probability = probabilities[1] * 100

        if prediction == 0:
            st.markdown(
                f"""
                <div class="result fake">
                    <h2>🔴 FAKE NEWS</h2>
                    <p>The model classified this news as potentially fake.</p>
                    <h3>Confidence: {fake_probability:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result real">
                    <h2>🟢 REAL NEWS</h2>
                    <p>The model classified this news as likely real.</p>
                    <h3>Confidence: {real_probability:.2f}%</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### Prediction Probabilities")

        m1, m2 = st.columns(2)

        with m1:
            st.metric("🔴 Fake", f"{fake_probability:.2f}%")

        with m2:
            st.metric("🟢 Real", f"{real_probability:.2f}%")

        st.progress(real_probability / 100)

st.divider()

with st.expander("ℹ️ About this project"):
    st.write(
        "This project uses a TF-IDF text vectorizer and Logistic Regression "
        "machine-learning model to classify news as Fake or Real."
    )
    st.write("The model is trained using the project's Fake.csv and True.csv datasets.")

st.caption("⚠️ This is an ML-based prediction tool, not a replacement for professional fact-checking.")
