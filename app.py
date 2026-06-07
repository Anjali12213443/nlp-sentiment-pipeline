import streamlit as st
import pickle
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Analyze sentiment using **TF-IDF + Logistic Regression** or **BERT**")
st.divider()

# ── Load models ───────────────────────────────────────────
@st.cache_resource
def load_baseline():
    with open('models/tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/logistic_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained('models/bert_sentiment')
    model = BertForSequenceClassification.from_pretrained('models/bert_sentiment')
    model.eval()
    return tokenizer, model

# ── Prediction functions ──────────────────────────────────
def predict_baseline(text, vectorizer, model):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return pred, max(prob)

def predict_bert(text, tokenizer, model):
    inputs = tokenizer(
        text, return_tensors='pt',
        max_length=128, truncation=True, padding='max_length'
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
    pred = np.argmax(probs)
    return pred, max(probs)

# ── UI ────────────────────────────────────────────────────
model_choice = st.radio(
    "Choose a model:",
    ["TF-IDF + Logistic Regression (Fast)", "BERT (Accurate)"],
    horizontal=True
)

user_input = st.text_area(
    "Enter a movie review:",
    height=150,
    placeholder="e.g. This movie was absolutely fantastic! The acting was superb..."
)

if st.button("Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Please enter a review first!")
    else:
        with st.spinner("Analyzing..."):
            if "BERT" in model_choice:
                tokenizer, model = load_bert()
                pred, confidence = predict_bert(user_input, tokenizer, model)
            else:
                vectorizer, model = load_baseline()
                pred, confidence = predict_baseline(user_input, vectorizer, model)

        label = "POSITIVE 😊" if pred == 1 else "NEGATIVE 😞"
        color = "green" if pred == 1 else "red"

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment", label)
        with col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")

        st.progress(float(confidence))

        if pred == 1:
            st.success("This review expresses a **positive** sentiment!")
        else:
            st.error("This review expresses a **negative** sentiment!")

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("📊 Model Performance")
    st.markdown("""
    | Model | Accuracy |
    |-------|----------|
    | TF-IDF + LR | 88.5% |
    | BERT | 85.0%* |
    
    *BERT trained on 2k samples vs 25k for baseline
    """)
    st.divider()
    st.markdown("**Built by Anjali Velu Ramalingam**")
    st.markdown("NLP Sentiment Analysis Pipeline")