# Aspect-Based and General Sentiment Analysis Web Application

**BE IT (Bachelor of Engineering in Information Technology) — Data Science Lab Mini Project**  
*Academic Year & Laboratory Curriculum Submission*

---

## 📌 Project Overview
Sentiment Analysis is a cornerstone Natural Language Processing (NLP) technique. However, traditional sentiment analysis models predict only an aggregate polarity (Positive / Negative / Neutral) across an entire review. In real-world customer reviews, sentences frequently contain conflicting polarities across different features (e.g., *"The camera is brilliant, but the battery life is atrocious"*).

This web application combines:
1. **General Sentiment Analysis**: Uses **Hugging Face DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`) to compute document/sentence-level sentiment with confidence scores.
2. **Aspect-Based Sentiment Analysis (ABSA)**: Deconstructs multi-aspect reviews into clause-level context windows and extracts individual polarities for specific aspects (e.g., *Camera*, *Battery*, *Service*, *Price*).
3. **Batch Review Analytics**: Uploads CSV datasets, runs parallel classification, computes sentiment breakdown charts (Donut + Bar plots), generates Positive vs. Negative Word Clouds, and exports predictions.

---

## 🚀 Key Features

- **Text Input & ABSA Tab**:
  - Live sentiment analysis with real-time Softmax probability meters.
  - Aspect-level entity polarity breakdown with confidence meters.
  - Domain presets (Electronics, Restaurants, Automotive, Custom).
  - Configurable Neutral Margin Threshold slider.
- **Batch CSV Analysis Tab**:
  - Upload custom review CSVs or load the built-in lab benchmark dataset with 1 click.
  - Interactive column selector and row slice selector.
  - Visual analytics: Donut chart, review count distribution, and Word Clouds.
  - Download annotated CSV with predictions and confidence scores.
- **Model Architecture & Specifications Tab**:
  - DistilBERT deep dive (66M parameters, 6 Transformer blocks, 12 attention heads).
  - Preprocessing pipeline: WordPiece tokenization, `[CLS]`, `[SEP]`, Softmax derivation.
- **Examiner Viva Voce Lab Prep Tab**:
  - 5 high-yield viva questions with model answers and examiner concepts.

---

## 🛠️ Tech Stack & Prerequisites

- **Language**: Python 3.10+ (Recommended: Python 3.10, 3.11, or 3.12)
- **Web UI**: Streamlit 1.32+
- **Deep Learning Framework**: PyTorch & Hugging Face Transformers
- **Pretrained Weights**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Data & Visualizations**: Pandas, NumPy, Matplotlib, Seaborn, WordCloud

---

## 📦 Installation & Setup Guide

### Step 1: Navigate to the Project Directory
```bash
cd "C:\Users\Abhishek\.gemini\antigravity\scratch\sentiment_analysis_webapp"
```

### Step 2: Create and Activate a Virtual Environment
```bash
# Windows PowerShell / CMD
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application
```bash
streamlit run app.py
```
*The app will automatically open in your default browser at `http://localhost:8501`.*

---

## 📂 Project Directory Structure

```text
sentiment_analysis_webapp/
│
├── app.py                   # Main Streamlit web application & inference pipeline
├── requirements.txt         # Pinned library dependencies
├── sample_reviews.csv       # Benchmark sample dataset for Batch Review testing
└── README.md                # Comprehensive documentation & Viva Voce guide
```

---

## 🎓 Examiner Viva Voce Q&A

### Q1: What is the WordPiece tokenization algorithm, and why are `[CLS]` and `[SEP]` tokens required?
- **WordPiece Tokenization** breaks words into subword units rather than whole words or single characters. Rare or Out-Of-Vocabulary (OOV) words are segmented into common sub-units prefixed with `##` (e.g., *"unhelpful"* $\rightarrow$ `["un", "##help", "##ful"]`). This bounds the vocabulary to ~30,522 tokens while completely eliminating OOV errors.
- **`[CLS]` (Classification Token)**: Placed at sequence index 0. The Transformer's self-attention mechanism aggregates whole-sentence context into this token's final hidden state, which is fed into the linear classification head.
- **`[SEP]` (Separator Token)**: Appended at the end of sentences to designate boundaries between sequence segments.

### Q2: What is Knowledge Distillation, and how does DistilBERT achieve comparable performance to BERT?
- **Knowledge Distillation** is a model compression method where a compact "student" model (DistilBERT: 6 layers, 66M parameters) learns from an ensemble or larger "teacher" model (BERT-base: 12 layers, 110M parameters).
- The student is optimized using a triple loss function:
  1. *Distillation Loss ($L_{ce}$)*: Matches the student's softened probability distribution to the teacher's soft targets via temperature scaling.
  2. *Masked Language Modeling Loss ($L_{mlm}$)*: Standard token prediction loss.
  3. *Cosine Embedding Loss ($L_{cos}$)*: Aligns the orientation of student and teacher hidden vectors.
- **Outcome**: DistilBERT is **40% smaller**, **60% faster**, and preserves **97%** of BERT's language comprehension score.

### Q3: Why does General Sentiment Analysis fail on complex customer reviews, and how does ABSA resolve this?
- **Limitation of General Sentiment**: Natural customer reviews are often mixed: *"The screen is brilliant, but the battery life is atrocious."* A general document-level classifier either averages these signals into a misleading "Neutral" or falsely condemns the entire product as "Negative".
- **ABSA Resolution**: Aspect-Based Sentiment Analysis decouples the review into `(Target Aspect, Sentiment Polarity)` tuples:
  - `Aspect: Screen` $\rightarrow$ **Positive (98%)**
  - `Aspect: Battery` $\rightarrow$ **Negative (96%)**
- This gives granular, actionable insights to engineering, QA, and product teams.

### Q4: Explain Precision, Recall, and F1-Score in Sentiment Analysis. Why is Accuracy insufficient?
- **Accuracy Paradox**: In production datasets, reviews are heavily skewed (e.g., 90% positive, 10% negative). A trivial model predicting everything as "Positive" achieves 90% accuracy but fails entirely to catch critical complaints.
- **Precision ($\frac{TP}{TP + FP}$)**: Out of all reviews predicted as Negative, how many were truly Negative? High precision prevents false alarms.
- **Recall ($\frac{TP}{TP + FN}$)**: Out of all actual Negative reviews written by users, how many did the model identify? High recall ensures no critical dissatisfaction is missed.
- **F1-Score ($2 \times \frac{Precision \times Recall}{Precision + Recall}$)**: The harmonic mean of Precision and Recall, providing an honest performance indicator on imbalanced data.

### Q5: How does Softmax convert logits into confidence scores, and how did you implement the Neutral class?
- **Logits to Probabilities**: The linear output layer yields raw logits $z_i \in (-\infty, \infty)$. Softmax normalizes them into a valid probability distribution:
  $$P(y_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$
  where $\sum P(y_i) = 1.0$ and $0 \le P(y_i) \le 1$.
- **Neutral Class Calibration**: SST-2 is natively trained on binary polarity (Positive / Negative). To handle neutral sentiment, we formulate a confidence margin $\delta$:
  $$\text{If } |P_{pos} - P_{neg}| \le \delta \implies \text{Class} = \textbf{NEUTRAL}$$
  If the difference exceeds $\delta$, the dominant class determines polarity.
