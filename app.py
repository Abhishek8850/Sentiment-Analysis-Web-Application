"""
================================================================================
BE IT Data Science Lab - Mini Project
Project Title: Aspect-Based and General Sentiment Analysis Web Application
Tech Stack   : Python 3.10+, Streamlit, Hugging Face Transformers (DistilBERT),
               Pandas, Matplotlib, Seaborn, WordCloud
================================================================================
"""

import re
import io
from typing import List, Dict, Tuple, Optional
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress unnecessary huggingface / torch warnings in UI
import warnings
warnings.filterwarnings("ignore")

# Hugging Face Transformers
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch


# ==============================================================================
# 1. STREAMLIT APP CONFIGURATION & THEMING
# ==============================================================================
st.set_page_config(
    page_title="Aspect & General Sentiment Analysis | BE IT DS Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished styling, metric cards, and badge accents
st.markdown("""
<style>
    /* Metric and card accents */
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #4a6fa5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .badge-pos {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-neg {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .badge-neu {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
    }
    .viva-box {
        background-color: #f1f5f9;
        border-left: 4px solid #0284c7;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# 2. MODEL LOADING & CACHING (HUGGING FACE TRANSFORMERS)
# ==============================================================================
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

@st.cache_resource(show_spinner="Loading DistilBERT Pretrained Sentiment Model into memory...")
def load_sentiment_pipeline():
    """
    Loads and caches the Hugging Face Transformers pipeline for sequence classification.
    DistilBERT SST-2 is a 66M parameter model distilled from BERT, offering ~97% of BERT's
    performance while running 60% faster with lower memory footprint.
    
    Returns:
        pipeline: Hugging Face text-classification pipeline with Softmax probabilities.
    """
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    # Modern transformers use top_k=None instead of return_all_scores=True
    try:
        classifier = pipeline(
            task="sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
            device=device
        )
    except Exception:
        classifier = pipeline(
            task="sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            device=device
        )
    return classifier


# ==============================================================================
# 3. CORE INFERENCE & ASPECT-BASED SENTIMENT ENGINE
# ==============================================================================
def _parse_pipeline_output(raw_output) -> Dict[str, float]:
    """
    Safely normalizes Hugging Face pipeline output across all versions and return formats.
    Supports:
    - List of dicts: [{'label': 'POSITIVE', 'score': 0.99}, {'label': 'NEGATIVE', 'score': 0.01}]
    - Nested list: [[{'label': 'POSITIVE', 'score': 0.99}, ...]]
    - Single dict or single-item list: [{'label': 'POSITIVE', 'score': 0.99}]
    """
    scores_dict = {}

    # Unwrap nested lists if present: [[{...}, {...}]] -> [{...}, {...}]
    if isinstance(raw_output, list) and len(raw_output) > 0 and isinstance(raw_output[0], list):
        raw_output = raw_output[0]

    if isinstance(raw_output, list):
        for entry in raw_output:
            if isinstance(entry, dict) and "label" in entry and "score" in entry:
                scores_dict[str(entry["label"]).upper()] = float(entry["score"])
    elif isinstance(raw_output, dict) and "label" in raw_output and "score" in raw_output:
        lbl = str(raw_output["label"]).upper()
        scores_dict[lbl] = float(raw_output["score"])

    # Ensure both POSITIVE and NEGATIVE probabilities are always populated
    if "POSITIVE" in scores_dict and "NEGATIVE" not in scores_dict:
        scores_dict["NEGATIVE"] = max(0.0, 1.0 - scores_dict["POSITIVE"])
    elif "NEGATIVE" in scores_dict and "POSITIVE" not in scores_dict:
        scores_dict["POSITIVE"] = max(0.0, 1.0 - scores_dict["NEGATIVE"])
    elif "POSITIVE" not in scores_dict and "NEGATIVE" not in scores_dict:
        scores_dict["POSITIVE"] = 0.5
        scores_dict["NEGATIVE"] = 0.5

    return scores_dict


def classify_text_sentiment(text: str, classifier, neutral_margin: float = 0.20) -> Dict:
    """
    Performs sentiment classification with 3-class resolution (Positive, Negative, Neutral).
    
    Mathematical Formulation:
    - DistilBERT fine-tuned on SST-2 outputs raw unnormalized logits for POSITIVE and NEGATIVE.
    - Softmax produces probabilities P(Positive) and P(Negative) such that P(Pos) + P(Neg) = 1.0.
    - If the absolute margin |P(Pos) - P(Neg)| < neutral_margin, the statement lacks
      definitive polarity and is calibrated as NEUTRAL.
    - Otherwise, the higher probability determines Positive or Negative.

    Args:
        text (str): Cleaned input sentence or paragraph.
        classifier: Transformers pipeline.
        neutral_margin (float): Decision margin below which sentiment is judged Neutral.

    Returns:
        dict: Predicted label, confidence score, and raw class distribution.
    """
    if not text or not text.strip():
        return {
            "label": "NEUTRAL",
            "score": 0.50,
            "pos_prob": 0.50,
            "neg_prob": 0.50,
            "neu_prob": 1.00
        }

    # Model inference
    # DistilBERT has a 512-token limit; truncate text cleanly if extremely long
    truncated_text = text[:1500]
    raw_output = classifier(truncated_text)

    # Safely parse pipeline output across any version of transformers
    scores_dict = _parse_pipeline_output(raw_output)
    pos_prob = scores_dict.get("POSITIVE", 0.0)
    neg_prob = scores_dict.get("NEGATIVE", 0.0)

    # Margin-based Neutral Calibration
    prob_difference = abs(pos_prob - neg_prob)

    if prob_difference <= neutral_margin:
        label = "NEUTRAL"
        # In the neutral zone, confidence is highest when the difference is smallest
        confidence = 1.0 - (prob_difference / neutral_margin) * 0.5
        neu_prob = 1.0 - prob_difference
    elif pos_prob > neg_prob:
        label = "POSITIVE"
        confidence = pos_prob
        neu_prob = 1.0 - prob_difference
    else:
        label = "NEGATIVE"
        confidence = neg_prob
        neu_prob = 1.0 - prob_difference

    return {
        "label": label,
        "score": round(confidence, 4),
        "pos_prob": round(pos_prob, 4),
        "neg_prob": round(neg_prob, 4),
        "neu_prob": round(max(0.0, min(1.0, neu_prob)), 4)
    }


def extract_aspect_sentiments(text: str, aspects: List[str], classifier, neutral_margin: float = 0.20) -> List[Dict]:
    """
    Aspect-Based Sentiment Analysis (ABSA):
    Breaks down multi-aspect reviews into clauses or context windows around specific target entities/aspects.
    Example:
    "The camera is stunning, but the battery life is completely pathetic."
    - Aspect 'camera': Evaluates clause "The camera is stunning" -> POSITIVE
    - Aspect 'battery': Evaluates clause "battery life is completely pathetic" -> NEGATIVE

    Args:
        text (str): Full review text.
        aspects (list): Target aspect terms (e.g., ['camera', 'battery', 'screen', 'price']).
        classifier: Transformers pipeline.
        neutral_margin (float): Margin for neutral classification.

    Returns:
        list of dicts containing each aspect, its extracted context, and sentiment score.
    """
    results = []
    # Split text into grammatical clauses or sentences using punctuation boundaries
    clauses = re.split(r'[.,;!]+|\bbut\b|\bhowever\b|\balthough\b|\bwhile\b', text, flags=re.IGNORECASE)
    clauses = [c.strip() for c in clauses if len(c.strip()) > 2]

    for aspect in aspects:
        aspect_clean = aspect.strip().lower()
        if not aspect_clean:
            continue

        matched_contexts = []
        # Find clauses containing the aspect keyword
        for clause in clauses:
            if re.search(rf'\b{re.escape(aspect_clean)}\b', clause, re.IGNORECASE):
                matched_contexts.append(clause)

        if matched_contexts:
            context_clause = " and ".join(matched_contexts)
        else:
            # Fallback context window: 8 words before and after aspect mention if in full text
            pattern = rf'((?:\S+\s+){{0,8}}\b{re.escape(aspect_clean)}\b(?:\s+\S+){{0,8}})'
            match = re.search(pattern, text, re.IGNORECASE)
            context_clause = match.group(1).strip() if match else None

        if context_clause:
            aspect_pred = classify_text_sentiment(context_clause, classifier, neutral_margin=neutral_margin)
            results.append({
                "aspect": aspect_clean.title(),
                "context": context_clause,
                "label": aspect_pred["label"],
                "score": aspect_pred["score"],
                "pos_prob": aspect_pred["pos_prob"],
                "neg_prob": aspect_pred["neg_prob"]
            })

    return results


# ==============================================================================
# 4. VISUALIZATION UTILITIES (PIE CHARTS & WORD CLOUDS)
# ==============================================================================
PALETTE = {
    "POSITIVE": "#2ecc71",  # Soft Green
    "NEGATIVE": "#e74c3c",  # Soft Red
    "NEUTRAL":  "#f39c12"   # Warm Amber
}

def plot_sentiment_breakdown(df: pd.DataFrame, label_col: str = "Predicted_Sentiment"):
    """
    Renders side-by-side Sentiment Distribution Donut Chart and Count Bar Chart.
    """
    counts = df[label_col].value_counts()
    categories = [cat for cat in ["POSITIVE", "NEGATIVE", "NEUTRAL"] if cat in counts.index]
    colors = [PALETTE.get(cat, "#95a5a6") for cat in categories]
    values = [counts[cat] for cat in categories]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Donut / Pie Chart
    axes[0].pie(
        values,
        labels=categories,
        colors=colors,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor='white', linewidth=2)
    )
    axes[0].set_title("Sentiment Proportions", fontsize=14, fontweight="bold", pad=15)

    # Count Bar Plot
    sns.barplot(x=categories, y=values, palette=colors, ax=axes[1], hue=categories, legend=False)
    axes[1].set_title("Review Count by Polarity", fontsize=14, fontweight="bold", pad=15)
    axes[1].set_ylabel("Number of Reviews", fontsize=11)
    axes[1].set_xlabel("Sentiment Category", fontsize=11)
    for i, v in enumerate(values):
        axes[1].text(i, v + (max(values)*0.02), str(v), ha='center', fontweight='bold')

    plt.tight_layout()
    return fig


def generate_wordcloud_or_keywords(texts: List[str], title: str, colormap: str = "Greens"):
    """
    Generates a WordCloud for a list of reviews.
    Falls back gracefully to a frequency bar plot if wordcloud library encounters an issue.
    """
    combined_text = " ".join([str(t) for t in texts if pd.notna(t)]).strip()
    
    if not combined_text:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No reviews available in this category.", ha="center", va="center")
        ax.axis("off")
        return fig

    try:
        from wordcloud import WordCloud, STOPWORDS
        custom_stopwords = set(STOPWORDS).union({
            "product", "item", "review", "one", "got", "will", "really", "even"
        })
        wc = WordCloud(
            width=650,
            height=320,
            background_color="white",
            colormap=colormap,
            stopwords=custom_stopwords,
            max_words=60,
            random_state=42
        ).generate(combined_text)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        return fig

    except Exception:
        # Fallback: Top 10 frequent words bar chart
        tokens = re.findall(r'\b[A-Za-z]{3,}\b', combined_text.lower())
        stop_words = {"the", "and", "this", "that", "with", "for", "was", "are", "have", "you", "not", "but"}
        filtered = [w for w in tokens if w not in stop_words]
        word_series = pd.Series(filtered).value_counts().head(10)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=word_series.values, y=word_series.index, ax=ax, palette="Blues_r")
        ax.set_title(f"{title} (Top Keywords)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Frequency")
        plt.tight_layout()
        return fig


# ==============================================================================
# 5. SIDEBAR: PROJECT METRICS & HYPERPARAMETERS
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=75)
    st.title("DS Lab Control Center")
    st.caption("BE IT Data Science Lab Project")
    
    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    st.info(f"**Architecture**: DistilBERT\n\n**Weights**: `{MODEL_NAME}`\n\n**Task**: Sequence Classification")
    
    neutral_threshold = st.slider(
        "Neutral Margin Threshold (|P_pos - P_neg|)",
        min_value=0.05,
        max_value=0.50,
        value=0.20,
        step=0.05,
        help="If the difference between Positive and Negative probability is less than this margin, the prediction is calibrated as NEUTRAL."
    )
    
    st.markdown("---")
    st.subheader("📌 Quick Presets for ABSA")
    preset_domain = st.selectbox(
        "Select Aspect Domain:",
        ["Electronics / Laptops / Mobiles", "Restaurants & Food", "Automotive / Cars", "Custom Input"]
    )
    
    if preset_domain == "Electronics / Laptops / Mobiles":
        default_aspects = "battery, camera, display, processor, sound, build quality"
    elif preset_domain == "Restaurants & Food":
        default_aspects = "food, service, ambience, price, hospitality, hygiene"
    elif preset_domain == "Automotive / Cars":
        default_aspects = "mileage, suspension, braking, handling, comfort, space"
    else:
        default_aspects = "quality, price, support, delivery"

    st.markdown("---")
    st.markdown(
        "**Author**: BE IT Student\n\n"
        "**Semester**: VIII / Data Science\n\n"
        "**Frameworks**: Hugging Face, PyTorch, Streamlit"
    )


# ==============================================================================
# 6. MAIN APPLICATION HEADER & TABS
# ==============================================================================
st.title("🧠 Aspect-Based & General Sentiment Analysis")
st.markdown(
    "A deep learning web application leveraging **Hugging Face DistilBERT** for general sentence polarity "
    "and targeted **Aspect-Based Sentiment Analysis (ABSA)** on user reviews."
)

# Load model pipeline into memory (cached across interactions)
classifier_pipeline = load_sentiment_pipeline()

tab_single, tab_batch, tab_specs, tab_viva = st.tabs([
    "💬 Single Text & Aspect Analysis",
    "📊 Batch CSV Analysis",
    "🔬 Model Architecture & Specs",
    "🎓 Viva Voce Lab Prep"
])


# ==============================================================================
# TAB 1: SINGLE TEXT & ASPECT-BASED SENTIMENT ANALYSIS
# ==============================================================================
with tab_single:
    st.subheader("1. General Sentiment Classification")
    sample_text = (
        "The screen display is breathtaking and vibrant, but the battery life is completely pathetic "
        "and customer support was quite unhelpful."
    )
    user_input = st.text_area(
        "Type or paste a customer review or statement below:",
        value=sample_text,
        height=110,
        placeholder="Enter sentence here..."
    )

    col_btn, col_clear = st.columns([1, 6])
    analyze_click = col_btn.button("🔍 Analyze Sentiment", type="primary")

    if analyze_click or user_input:
        with st.spinner("Analyzing semantics with DistilBERT..."):
            pred = classify_text_sentiment(user_input, classifier_pipeline, neutral_margin=neutral_threshold)

        st.markdown("#### 🎯 Overall Classification Result")
        res_col1, res_col2, res_col3, res_col4 = st.columns(4)

        # Polarity badge rendering
        if pred["label"] == "POSITIVE":
            badge_html = "<span class='badge-pos'>POSITIVE 😊</span>"
        elif pred["label"] == "NEGATIVE":
            badge_html = "<span class='badge-neg'>NEGATIVE 😞</span>"
        else:
            badge_html = "<span class='badge-neu'>NEUTRAL 😐</span>"

        res_col1.markdown(f"**Predicted Polarity**:<br>{badge_html}", unsafe_allow_html=True)
        res_col2.metric("Confidence Score", f"{pred['score']*100:.1f}%")
        res_col3.metric("Positive Probability", f"{pred['pos_prob']*100:.1f}%")
        res_col4.metric("Negative Probability", f"{pred['neg_prob']*100:.1f}%")

        # Visual confidence score progress meters
        st.markdown("**Probability Distribution:**")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.caption(f"Positive Softmax Probability: {pred['pos_prob']*100:.1f}%")
            st.progress(float(pred['pos_prob']))
        with p_col2:
            st.caption(f"Negative Softmax Probability: {pred['neg_prob']*100:.1f}%")
            st.progress(float(pred['neg_prob']))

        st.markdown("---")

        # ----------------------------------------------------------------------
        # Aspect-Based Sentiment Analysis (ABSA) Section
        # ----------------------------------------------------------------------
        st.subheader("2. Aspect-Based Sentiment Analysis (ABSA)")
        st.caption(
            "ABSA pinpoints specific targets/features within the same sentence to resolve mixed or conflicting reviews."
        )

        aspect_input = st.text_input(
            "Target Aspects (comma-separated):",
            value=default_aspects,
            help="Enter specific product features to inspect individual polarities."
        )

        target_aspect_list = [a.strip() for a in aspect_input.split(",") if a.strip()]

        if target_aspect_list:
            absa_results = extract_aspect_sentiments(
                user_input, target_aspect_list, classifier_pipeline, neutral_margin=neutral_threshold
            )

            if absa_results:
                st.markdown(f"**Found {len(absa_results)} aspect(s) in review:**")
                absa_cols = st.columns(min(len(absa_results), 3))

                for idx, res in enumerate(absa_results):
                    col = absa_cols[idx % 3]
                    with col:
                        with st.container():
                            st.markdown(f"### 🏷️ `{res['aspect']}`")
                            if res["label"] == "POSITIVE":
                                st.markdown("<span class='badge-pos'>POSITIVE</span>", unsafe_allow_html=True)
                            elif res["label"] == "NEGATIVE":
                                st.markdown("<span class='badge-neg'>NEGATIVE</span>", unsafe_allow_html=True)
                            else:
                                st.markdown("<span class='badge-neu'>NEUTRAL</span>", unsafe_allow_html=True)

                            st.write(f"**Confidence**: `{res['score']*100:.1f}%`")
                            st.caption(f"**Context**: *\"{res['context']}\"*")
                            st.divider()

                # Aspect Summary Table
                absa_df = pd.DataFrame(absa_results)[["aspect", "label", "score", "context"]]
                absa_df.columns = ["Target Aspect", "Detected Polarity", "Confidence", "Context Clause"]
                st.dataframe(absa_df, use_container_width=True)
            else:
                st.info("ℹ️ None of the specified aspects were detected in the input review. Try adding aspects present in the text above.")


# ==============================================================================
# TAB 2: BATCH CSV ANALYSIS
# ==============================================================================
with tab_batch:
    st.subheader("📁 Batch Review Processing")
    st.markdown(
        "Upload a `.csv` dataset containing multiple customer reviews to run batch inference, "
        "render polarity proportions, inspect word clouds, and download predictions."
    )

    upload_col, sample_col = st.columns([3, 2])
    with upload_col:
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    with sample_col:
        st.write("Or use the built-in lab benchmark dataset:")
        load_sample = st.button("🧪 Load Sample Reviews Dataset")

    df_batch = None
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded uploaded dataset: {df_batch.shape[0]} rows, {df_batch.shape[1]} columns.")
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")
    elif load_sample:
        try:
            df_batch = pd.read_csv("sample_reviews.csv")
            st.success(f"Loaded benchmark dataset with {len(df_batch)} customer reviews.")
        except Exception:
            # Inline fallback if sample file isn't found
            df_batch = pd.DataFrame({
                "review_id": [1, 2, 3, 4, 5, 6],
                "review_text": [
                    "The battery life is amazing and display is bright!",
                    "Terrible product. Broke down after three days.",
                    "The camera takes great photos, but UI is sluggish.",
                    "Average build quality for this price point.",
                    "Fast shipping and pristine condition packaging.",
                    "Customer service was rude and completely unhelpful."
                ]
            })
            st.info("Loaded default fallback benchmark dataset.")

    if df_batch is not None:
        st.markdown("---")
        st.subheader("Data Preview & Column Selection")
        st.dataframe(df_batch.head(5), use_container_width=True)

        # Detect likely text column automatically
        text_col_candidates = [c for c in df_batch.columns if any(k in c.lower() for k in ["text", "review", "comment", "feedback"])]
        default_index = df_batch.columns.get_loc(text_col_candidates[0]) if text_col_candidates else 0

        target_col = st.selectbox(
            "Select Review Text Column to Classify:",
            options=df_batch.columns,
            index=int(default_index)
        )

        max_rows = st.slider("Number of rows to analyze (for quick interactive lab demo):", 5, min(500, len(df_batch)), min(50, len(df_batch)))

        if st.button("🚀 Run Batch Sentiment Classification", type="primary"):
            df_subset = df_batch.head(max_rows).copy()
            progress_bar = st.progress(0)
            status_text = st.empty()

            labels, confidences, pos_probs, neg_probs = [], [], [], []

            total_items = len(df_subset)
            for idx, text_val in enumerate(df_subset[target_col]):
                pred_res = classify_text_sentiment(str(text_val), classifier_pipeline, neutral_margin=neutral_threshold)
                labels.append(pred_res["label"])
                confidences.append(pred_res["score"])
                pos_probs.append(pred_res["pos_prob"])
                neg_probs.append(pred_res["neg_prob"])

                # Update progress bar
                progress_bar.progress((idx + 1) / total_items)
                status_text.caption(f"Inferencing review {idx + 1} of {total_items}...")

            df_subset["Predicted_Sentiment"] = labels
            df_subset["Confidence_Score"] = confidences
            df_subset["Pos_Probability"] = pos_probs
            df_subset["Neg_Probability"] = neg_probs

            status_text.success("Classification completed successfully!")

            # ------------------------------------------------------------------
            # Batch KPI Metrics
            # ------------------------------------------------------------------
            st.markdown("### 📈 Batch Summary Metrics")
            kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
            total_n = len(df_subset)
            pos_n = (df_subset["Predicted_Sentiment"] == "POSITIVE").sum()
            neg_n = (df_subset["Predicted_Sentiment"] == "NEGATIVE").sum()
            neu_n = (df_subset["Predicted_Sentiment"] == "NEUTRAL").sum()
            avg_conf = df_subset["Confidence_Score"].mean()

            kpi1.metric("Total Analyzed", total_n)
            kpi2.metric("Positive Reviews", f"{pos_n} ({pos_n/total_n*100:.1f}%)")
            kpi3.metric("Negative Reviews", f"{neg_n} ({neg_n/total_n*100:.1f}%)")
            kpi4.metric("Neutral Reviews", f"{neu_n} ({neu_n/total_n*100:.1f}%)")
            kpi5.metric("Avg Confidence", f"{avg_conf*100:.1f}%")

            # ------------------------------------------------------------------
            # Visualizations: Distribution & Word Clouds
            # ------------------------------------------------------------------
            st.markdown("---")
            st.subheader("📊 Sentiment Proportions & Charts")
            fig_charts = plot_sentiment_breakdown(df_subset)
            st.pyplot(fig_charts)

            st.markdown("---")
            st.subheader("☁️ Word Clouds (Positive vs. Negative Themes)")
            wc_col1, wc_col2 = st.columns(2)

            pos_texts = df_subset[df_subset["Predicted_Sentiment"] == "POSITIVE"][target_col].tolist()
            neg_texts = df_subset[df_subset["Predicted_Sentiment"] == "NEGATIVE"][target_col].tolist()

            with wc_col1:
                fig_pos_wc = generate_wordcloud_or_keywords(pos_texts, "Top Positive Themes", colormap="Greens")
                st.pyplot(fig_pos_wc)

            with wc_col2:
                fig_neg_wc = generate_wordcloud_or_keywords(neg_texts, "Top Negative Themes", colormap="Reds")
                st.pyplot(fig_neg_wc)

            # ------------------------------------------------------------------
            # Processed Table & CSV Export
            # ------------------------------------------------------------------
            st.markdown("---")
            st.subheader("📋 Processed Results Table")
            st.dataframe(df_subset, use_container_width=True)

            csv_buffer = io.BytesIO()
            df_subset.to_csv(csv_buffer, index=False, encoding='utf-8')
            csv_data = csv_buffer.getvalue()

            st.download_button(
                label="📥 Download Processed CSV with Predictions",
                data=csv_data,
                file_name="sentiment_predictions_output.csv",
                mime="text/csv",
                type="primary"
            )


# ==============================================================================
# TAB 3: MODEL ARCHITECTURE & PIPELINE SPECS
# ==============================================================================
with tab_specs:
    st.subheader("🔬 Deep Learning Architecture & Pipeline Specifications")
    st.markdown("""
    This project deploys **DistilBERT** (*distilbert-base-uncased-finetuned-sst-2-english*), a state-of-the-art 
    Transformer architecture fine-tuned on the Stanford Sentiment Treebank (SST-2) corpus.
    """)

    spec_col1, spec_col2 = st.columns(2)
    with spec_col1:
        st.markdown("""
        #### 🏛️ Architectural Overview
        - **Base Model**: DistilBERT (Knowledge-distilled variant of BERT)
        - **Parameters**: 66 Million (compared to BERT-Base's 110M)
        - **Layers (Transformers Blocks)**: 6 (vs 12 in BERT)
        - **Attention Heads**: 12
        - **Hidden Size**: 768
        - **Distillation Speedup**: ~60% faster inference on CPU
        - **Retained Language Capability**: ~97% of BERT's GLUE benchmark score
        """)

    with spec_col2:
        st.markdown("""
        #### 🔄 Preprocessing & Inference Lifecycle
        1. **Text Normalization**: Lowercasing, punctuation boundary extraction, and whitespace stripping.
        2. **WordPiece Tokenization**:
           - Subword segmentation (e.g., `"unbelievable"` $\\rightarrow$ `["un", "##belie", "##vable"]`).
           - Prepends `[CLS]` classification token and appends `[SEP]` separation token.
        3. **Feed-Forward & Self-Attention**:
           - 6 layers of Multi-Head Self-Attention capture contextual word dependencies.
        4. **Softmax Output**:
           - Logits from the pooled `[CLS]` token pass through a linear layer and Softmax function:
           $$P(y_i) = \\frac{e^{z_i}}{\\sum_{j} e^{z_j}}$$
        """)

    st.markdown("---")
    st.subheader("General Sentiment vs. Aspect-Based Sentiment Analysis (ABSA)")
    st.markdown("""
    | Evaluation Criterion | General Sentiment Analysis | Aspect-Based Sentiment Analysis (ABSA) |
    | :--- | :--- | :--- |
    | **Granularity** | Document or sentence-level aggregate | Fine-grained entity/feature level |
    | **Output** | Single scalar polarity (e.g., Positive) | Vector of `(aspect, sentiment, score)` tuples |
    | **Handling Mixed Reviews** | Often misclassifies as neutral or averages out contradictory clauses | Decouples distinct polarities (e.g., +Camera, -Battery) |
    | **Business Value** | Overall brand perception tracking | Actionable product engineering feedback |
    """)


# ==============================================================================
# TAB 4: VIVA VOCE & LAB EXAMINER GUIDE
# ==============================================================================
with tab_viva:
    st.subheader("🎓 BE IT Data Science Lab - Examiner Viva Voce Prep")
    st.markdown(
        "Here are **5 fundamental technical questions** frequently posed by examiners during external lab evaluations, "
        "along with concise, technically rigorous model answers."
    )

    with st.expander("Q1: What is the WordPiece tokenization algorithm, and why are [CLS] and [SEP] tokens required?", expanded=True):
        st.markdown("""
        <div class="viva-box">
        <b>Examiner Concept:</b> NLP Preprocessing, Tokenization, Vocabulary Handling.<br><br>
        <b>Model Answer:</b>
        <ul>
            <li><b>WordPiece Tokenization</b> breaks text into subword units rather than full words or characters. If a word is rare or out-of-vocabulary (OOV), it splits it into frequent prefixes/suffixes marked with <code>##</code> (e.g., <i>"unhelpful"</i> &rarr; <code>["un", "##help", "##ful"]</code>). This bounds vocabulary size (~30,522 tokens) while completely eliminating OOV errors.</li>
            <li><b><code>[CLS]</code> (Classification Token)</b>: Prepended at index 0 of every input sequence. Its final hidden state representation aggregates whole-sentence context through self-attention, serving as the pooled representation input to the sequence classification head.</li>
            <li><b><code>[SEP]</code> (Separator Token)</b>: Appended at the end of a sentence or between two text segments to indicate sequence boundaries.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Q2: What is Knowledge Distillation, and how does DistilBERT achieve comparable performance to BERT?"):
        st.markdown("""
        <div class="viva-box">
        <b>Examiner Concept:</b> Model Compression, Teacher-Student Architecture.<br><br>
        <b>Model Answer:</b>
        <ul>
            <li><b>Knowledge Distillation</b> is a model compression technique where a compact student model (DistilBERT, 6 layers) is trained under the supervision of a large teacher model (BERT-base, 12 layers).</li>
            <li>The student is trained on a triple loss function:
                <ol>
                    <li><b>Distillation Loss ($L_{ce}$)</b>: Matches the student's soft probability output distribution to the teacher's soft probabilities using a temperature parameter $T$ (cross-entropy on softened logits).</li>
                    <li><b>Masked Language Modeling Loss ($L_{mlm}$)</b>: Standard supervised masked token prediction loss.</li>
                    <li><b>Cosine Embedding Loss ($L_{cos}$)</b>: Aligns the directional orientation of student and teacher hidden state vectors.</li>
                </ol>
            </li>
            <li><b>Outcome:</b> DistilBERT is <b>40% smaller</b>, <b>60% faster</b>, and preserves <b>97%</b> of BERT's language comprehension capability.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Q3: Why does General Sentiment Analysis fail on complex customer reviews, and how does ABSA resolve this?"):
        st.markdown("""
        <div class="viva-box">
        <b>Examiner Concept:</b> ABSA vs. Document Polarity, Real-World NLP Bottlenecks.<br><br>
        <b>Model Answer:</b>
        <ul>
            <li><b>The Problem with General Sentiment:</b> Real-world reviews are rarely uniformly positive or negative. For example: <i>"The screen is brilliant, but the battery life is atrocious."</i> A general classifier will either average these opposing signals into a misleading <b>Neutral</b> or falsely label the entire laptop as Negative.</li>
            <li><b>How ABSA Resolves This:</b> Aspect-Based Sentiment Analysis decomposes the text into <code>(Target Aspect, Sentiment Polarity)</code> pairs. By locating aspect references and evaluating localized syntactic clauses or attention context windows, ABSA reports:
                <br>&bull; <code>Aspect: Screen &rarr; POSITIVE (98%)</code>
                <br>&bull; <code>Aspect: Battery &rarr; NEGATIVE (96%)</code>
            </li>
            <li>This provides granular, actionable feedback to engineering and product teams.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Q4: Explain Precision, Recall, and F1-Score in the context of Sentiment Analysis. Why is Accuracy insufficient?"):
        st.markdown(r"""
        <div class="viva-box">
        <b>Examiner Concept:</b> Model Evaluation Metrics, Class Imbalance.<br><br>
        <b>Model Answer:</b>
        <ul>
            <li><b>Accuracy Paradox:</b> In real customer datasets, positive reviews frequently outnumber negative reviews (e.g., 90% positive, 10% negative). A naive classifier that predicts everything as Positive achieves 90% accuracy but fails completely at detecting customer churn or complaints.</li>
            <li><b>Precision ($\frac{TP}{TP + FP}$):</b> Out of all reviews the model classified as Negative, what proportion were truly Negative? High precision minimizes false alarms.</li>
            <li><b>Recall ($\frac{TP}{TP + FN}$):</b> Out of all actual Negative reviews written by users, what proportion did the model successfully catch? High recall ensures critical negative feedback is never missed.</li>
            <li><b>F1-Score ($2 \times \frac{Precision \times Recall}{Precision + Recall}$):</b> The harmonic mean of Precision and Recall, providing a balanced, honest evaluation metric on imbalanced review datasets.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Q5: How does Softmax convert logits to confidence scores, and how did you implement the Neutral class?"):
        st.markdown(r"""
        <div class="viva-box">
        <b>Examiner Concept:</b> Activation Functions, Logits, Threshold Calibration.<br><br>
        <b>Model Answer:</b>
        <ul>
            <li><b>Logits to Probabilities:</b> The classification head produces unnormalized real-valued logits $z_i \in (-\infty, \infty)$. The <b>Softmax function</b> exponentiates and normalizes them:
            $$P(y = i) = \frac{e^{z_i / T}}{\sum_{j=1}^{K} e^{z_j / T}}$$
            This guarantees all probabilities lie in $[0, 1]$ and sum exactly to $1.0$.</li>
            <li><b>Neutral Class Formulation:</b> SST-2 is natively trained on binary polarity (Positive/Negative). To support a realistic 3-class system (Positive, Negative, Neutral):
                <br>We introduce a <b>Confidence Margin ($\delta$)</b>: If $|P_{pos} - P_{neg}| \le \delta$ (e.g., $0.20$), the model's logits reflect ambiguity or moderate tone without decisive polarity. The sample is calibrated as <b>NEUTRAL</b>. If the difference exceeds $\delta$, the dominant class determines polarity.
            </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
