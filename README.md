# 🎬 NLP Sentiment Analysis Pipeline

An end-to-end Natural Language Processing pipeline that analyzes sentiment of movie reviews using both classical ML and modern deep learning approaches — built with real IMDb data, two trained models, and a live interactive web app.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![BERT](https://img.shields.io/badge/BERT-HuggingFace-yellow)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)

---

## 🌟 Live Demo

Run locally with:
```bash
streamlit run app.py
```

Type any movie review and get an instant sentiment prediction with confidence score from either model.

---

## 📊 Model Performance

| Model | Training Data | Accuracy | Training Time |
|-------|--------------|----------|---------------|
| TF-IDF + Logistic Regression | 25,000 reviews | 88.5% | ~30 seconds |
| BERT (Fine-tuned) | 2,000 reviews | 85.0% | ~30 minutes |

> **Note:** BERT's lower accuracy here is due to training on 12.5x less data than the baseline. With equal data, BERT typically scores 93–95%.

---

## 🏗️ Project Structure

```
nlp-sentiment-pipeline/
├── notebooks/
│   ├── 01_data_loading.ipynb        # Dataset loading & exploration
│   ├── 02_preprocessing.ipynb       # Text cleaning pipeline
│   ├── 03_baseline_model.ipynb      # TF-IDF + Logistic Regression
│   ├── 04_bert_model.ipynb          # BERT fine-tuning
│   └── 05_model_comparison.ipynb    # Side-by-side model comparison
├── outputs/
│   ├── baseline_confusion_matrix.png
│   └── model_comparison.png
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## 🔧 Tech Stack

| Category | Tools |
|----------|-------|
| NLP & Deep Learning | HuggingFace Transformers, BERT (`bert-base-uncased`) |
| Classical ML | Scikit-learn, TF-IDF, Logistic Regression |
| Text Processing | NLTK, spaCy, Regex |
| Data | HuggingFace Datasets, Pandas, NumPy |
| Web App | Streamlit |
| Framework | Python, PyTorch |

---

## 🚀 Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/Anjali12213443/nlp-sentiment-pipeline.git
cd nlp-sentiment-pipeline

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate        # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# 5. Download spaCy model
python3 -m spacy download en_core_web_sm

# 6. Run the Streamlit app
streamlit run app.py
```

---

## 🔍 Pipeline Overview

### Step 1 — Data Loading
- Dataset: IMDb Movie Reviews via HuggingFace (`stanfordnlp/imdb`)
- 25,000 training + 25,000 test samples
- Perfectly balanced: 50% positive, 50% negative

### Step 2 — Text Preprocessing
- Lowercase conversion
- HTML tag removal (IMDb reviews contain HTML)
- URL and punctuation removal
- Tokenization
- Stopword removal using NLTK

### Step 3 — Baseline Model (TF-IDF + Logistic Regression)
- TF-IDF vectorization with 10,000 features and bigrams
- Logistic Regression classifier
- **Accuracy: 88.5%** on 25,000 test samples

### Step 4 — BERT Fine-tuning
- Pre-trained `bert-base-uncased` from HuggingFace
- Fine-tuned for binary sequence classification
- Trained for 2 epochs on 2,000 samples (CPU)
- **Accuracy: 85.0%** on 500 test samples

### Step 5 — Streamlit Web App
- Choose between TF-IDF (fast) or BERT (accurate)
- Live sentiment prediction with confidence score
- Visual progress bar and color-coded results

---

## 📸 App Screenshots

### Positive Review — 97% Confidence
> *"This movie was absolutely incredible! The acting was superb, the plot kept me on the edge of my seat."*
> → **POSITIVE 😊 — 97.0%**

### Negative Review — 99.3% Confidence
> *"What a waste of time. The story made no sense, the acting was terrible."*
> → **NEGATIVE 😞 — 99.3%**

---

## 📁 Key Files

| File | Description |
|------|-------------|
| `app.py` | Streamlit app — loads both models and serves predictions |
| `notebooks/02_preprocessing.ipynb` | Text cleaning pipeline with before/after examples |
| `notebooks/03_baseline_model.ipynb` | Full baseline training, evaluation, and confusion matrix |
| `notebooks/04_bert_model.ipynb` | BERT fine-tuning with training loss tracking |

---

## 👩‍💻 Author

**Anjali Velu Ramalingam**
M.S. Data Science & Analytics | Minor in Artificial Intelligence

[GitHub](https://github.com/Anjali12213443)
