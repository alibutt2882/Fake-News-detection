import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_curve, auc
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FakeShield · Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME / GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* Root palette */
:root {
    --bg:        #0a0c14;
    --surface:   #10131f;
    --card:      #161a2b;
    --border:    #252a42;
    --accent:    #5b6ef5;
    --accent2:   #e84393;
    --teal:      #00d4aa;
    --gold:      #f5c842;
    --red:       #ff4d6d;
    --green:     #2ecc71;
    --text:      #e8eaf0;
    --muted:     #7c82a3;
    --radius:    14px;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Hide default header ── */
header[data-testid="stHeader"] { display: none !important; }

/* ── Main padding ── */
.main .block-container { padding: 2rem 2.5rem; max-width: 1300px; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #1a1f3a 0%, #0d1023 60%, #0a0c14 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(91,110,245,.25) 0%, transparent 70%);
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 40%;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(232,67,147,.15) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #5b6ef5, #e84393, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: .5rem;
}
.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: .04em;
}
.badge {
    display: inline-block;
    background: rgba(91,110,245,.15);
    border: 1px solid rgba(91,110,245,.4);
    border-radius: 40px;
    padding: .25rem .75rem;
    font-size: .75rem;
    color: #a0aaff;
    letter-spacing: .06em;
    text-transform: uppercase;
    font-weight: 500;
    margin-bottom: 1rem;
}

/* ── Section headings ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: .25rem;
}
.section-rule {
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
    border: none;
    margin-bottom: 1.5rem;
    margin-top: .25rem;
}

/* ── Metric cards ── */
.metric-row { display: flex; gap: 1rem; margin-bottom: 1.5rem; }
.metric-card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    border-radius: 3px 0 0 3px;
}
.metric-card.blue::before  { background: var(--accent);  }
.metric-card.pink::before  { background: var(--accent2); }
.metric-card.teal::before  { background: var(--teal);    }
.metric-card.gold::before  { background: var(--gold);    }
.metric-label { font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; margin-bottom: .3rem; }
.metric-value { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 700; }
.metric-card.blue  .metric-value { color: var(--accent);  }
.metric-card.pink  .metric-value { color: var(--accent2); }
.metric-card.teal  .metric-value { color: var(--teal);    }
.metric-card.gold  .metric-value { color: var(--gold);    }

/* ── Result verdict ── */
.verdict-box {
    border-radius: var(--radius);
    padding: 1.75rem 2rem;
    margin: 1.5rem 0;
    border: 1.5px solid;
    position: relative;
    overflow: hidden;
}
.verdict-box.fake {
    background: rgba(255,77,109,.08);
    border-color: rgba(255,77,109,.4);
}
.verdict-box.real {
    background: rgba(46,204,113,.08);
    border-color: rgba(46,204,113,.4);
}
.verdict-emoji { font-size: 2.5rem; margin-bottom: .5rem; }
.verdict-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    margin-bottom: .3rem;
}
.verdict-box.fake  .verdict-title { color: var(--red);   }
.verdict-box.real  .verdict-title { color: var(--green); }
.verdict-desc { color: var(--muted); font-size: .9rem; }

/* ── Streamlit button overrides ── */
.stButton>button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: .04em !important;
    padding: .55rem 1.6rem !important;
    transition: opacity .2s ease !important;
}
.stButton>button:hover { opacity: .85 !important; }

/* ── Textarea ── */
.stTextArea textarea {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .93rem !important;
}
.stTextArea textarea:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(91,110,245,.2) !important; }

/* ── selectbox / slider labels ── */
.stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label {
    color: var(--muted) !important;
    font-size: .85rem !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: var(--muted) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; }

/* ── Model confidence bar ── */
.conf-bar-wrap { margin: .6rem 0; }
.conf-label { display: flex; justify-content: space-between; font-size: .83rem; color: var(--muted); margin-bottom: .25rem; }
.conf-bar-bg { background: var(--border); border-radius: 40px; height: 8px; overflow: hidden; }
.conf-bar { height: 100%; border-radius: 40px; transition: width .6s ease; }

/* ── Info / warning boxes ── */
.info-box {
    background: rgba(91,110,245,.08);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: .9rem 1.1rem;
    font-size: .88rem;
    color: var(--muted);
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & COLOURS
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e8eaf0", family="DM Sans"),
    title_font=dict(family="Syne", size=16),
    xaxis=dict(showgrid=False, zeroline=False, color="#7c82a3"),
    yaxis=dict(showgrid=True, gridcolor="#252a42", zeroline=False, color="#7c82a3"),
)
MODEL_COLORS = {
    "Logistic Regression":      "#5b6ef5",
    "Decision Tree":            "#e84393",
    "Gradient Boosting":        "#00d4aa",
    "Random Forest":            "#f5c842",
}
MODEL_ABBR = {
    "Logistic Regression":  "LR",
    "Decision Tree":        "DT",
    "Gradient Boosting":    "GBC",
    "Random Forest":        "RFC",
}


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def wordopt(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & TRAINING  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_train():
    """Generate synthetic data + train all four models."""
    rng = np.random.default_rng(42)

    # ── synthetic word bags ──────────────────────────────────────────────────
    real_words = [
        "government official statement press secretary",
        "reuters report confirmed sources police",
        "study published journal research university",
        "election results certified board officials",
        "scientist says climate data evidence",
        "hospital authorities confirmed patient data",
        "budget approved congress senate vote",
        "prime minister announced economic policy",
        "court ruling judge decision appeal",
        "investigation found no evidence fraud",
    ]
    fake_words = [
        "shocking secret they don't want you to know",
        "deep state conspiracy cover-up exposed truth",
        "mainstream media lies government controls narrative",
        "miracle cure doctors hiding from public",
        "globalist agenda revealed exposed whistleblower",
        "you won't believe what happened wake up sheeple",
        "breaking bombshell explosive revelation truth seeker",
        "big pharma suppressed cure hidden agenda",
        "elite cabal plans new world order exposed",
        "viral truth censored shadow banned platforms",
    ]

    def make_texts(bags, n):
        texts = []
        for _ in range(n):
            bag = bags[rng.integers(0, len(bags))]
            extra = " ".join(rng.choice(bag.split(), size=rng.integers(5, 15)))
            texts.append(bag + " " + extra)
        return texts

    n = 2000
    texts  = make_texts(real_words, n) + make_texts(fake_words, n)
    labels = [0] * n + [1] * n          # 0 = real, 1 = fake

    df = pd.DataFrame({"text": texts, "class": labels})
    df["text"] = df["text"].apply(wordopt)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["class"], test_size=0.25, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    models = {
        "Logistic Regression":  LogisticRegression(max_iter=300),
        "Decision Tree":        DecisionTreeClassifier(max_depth=12, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=80, random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=80, random_state=42),
    }

    results, trained = {}, {}
    for name, clf in models.items():
        clf.fit(Xtr, y_train)
        preds = clf.predict(Xte)
        probs = clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") else None
        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds)
        fpr, tpr, _ = roc_curve(y_test, probs) if probs is not None else (None, None, None)
        roc_auc = auc(fpr, tpr) if fpr is not None else None
        results[name] = dict(
            accuracy=accuracy_score(y_test, preds),
            report=report, cm=cm,
            fpr=fpr, tpr=tpr, roc_auc=roc_auc,
        )
        trained[name] = clf

    return trained, vectorizer, results, (X_test, y_test)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def predict_news(text: str, models_dict, vectorizer):
    cleaned = wordopt(text)
    vec = vectorizer.transform([cleaned])
    out = {}
    for name, clf in models_dict.items():
        pred = clf.predict(vec)[0]
        prob = clf.predict_proba(vec)[0] if hasattr(clf, "predict_proba") else None
        out[name] = {
            "label":       "Fake" if pred == 1 else "Real",
            "prob_fake":   float(prob[1]) if prob is not None else (1.0 if pred else 0.0),
            "prob_real":   float(prob[0]) if prob is not None else (0.0 if pred else 1.0),
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def plot_accuracy_bar(results):
    names  = list(results.keys())
    accs   = [results[n]["accuracy"] * 100 for n in names]
    colors = [MODEL_COLORS[n] for n in names]

    fig = go.Figure(go.Bar(
        x=names, y=accs,
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{a:.1f}%" for a in accs],
        textposition="outside",
        textfont=dict(size=13, family="Syne", color="#e8eaf0"),
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title="Model Accuracy (%)",
        yaxis=dict(range=[0, 110], **PLOTLY_THEME["yaxis"]),
        bargap=0.35, height=380,
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def plot_roc_all(results):
    fig = go.Figure()
    for name, res in results.items():
        if res["fpr"] is not None:
            fig.add_trace(go.Scatter(
                x=res["fpr"], y=res["tpr"],
                mode="lines",
                name=f"{MODEL_ABBR[name]}  (AUC={res['roc_auc']:.3f})",
                line=dict(color=MODEL_COLORS[name], width=2.5),
            ))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             line=dict(dash="dash", color="#7c82a3", width=1),
                             showlegend=False))
    fig.update_layout(
        **PLOTLY_THEME,
        title="ROC Curves – All Models",
        xaxis=dict(title="False Positive Rate", **PLOTLY_THEME["xaxis"]),
        yaxis=dict(title="True Positive Rate",  **PLOTLY_THEME["yaxis"]),
        height=420,
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def plot_confusion(cm, name):
    labels = ["Real", "Fake"]
    text = [[str(v) for v in row] for row in cm]
    color = MODEL_COLORS[name]

    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        textfont=dict(size=18, family="Syne"),
        colorscale=[[0, "#10131f"], [1, color]],
        showscale=False,
    ))
    fig.update_layout(
        **PLOTLY_THEME,
        title=f"Confusion Matrix – {name}",
        xaxis=dict(title="Predicted", **PLOTLY_THEME["xaxis"]),
        yaxis=dict(title="Actual",    **PLOTLY_THEME["yaxis"]),
        height=340,
        margin=dict(t=50, b=10, l=10, r=10),
    )
    return fig


def plot_precision_recall(results):
    metrics = ["precision", "recall", "f1-score"]
    classes = ["0", "1"]   # 0=Real, 1=Fake
    class_labels = ["Real", "Fake"]

    rows, cols = 1, 3
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[m.capitalize() for m in metrics],
                        horizontal_spacing=0.08)

    for ci, metric in enumerate(metrics, 1):
        for class_idx, (cls, cls_label) in enumerate(zip(classes, class_labels)):
            vals = [results[n]["report"][cls][metric] for n in results]
            fig.add_trace(go.Bar(
                name=cls_label, x=list(results.keys()), y=vals,
                marker_color=[MODEL_COLORS[n] for n in results] if class_idx == 0
                             else [f"rgba({int(MODEL_COLORS[n][1:3],16)},{int(MODEL_COLORS[n][3:5],16)},{int(MODEL_COLORS[n][5:],16)},0.5)" for n in results],
                showlegend=(ci == 1),
                offsetgroup=class_idx,
            ), row=1, col=ci)

    fig.update_layout(
        **PLOTLY_THEME,
        title="Precision / Recall / F1 by Class",
        barmode="group",
        bargap=0.2,
        height=420,
        margin=dict(t=60, b=10, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    for ci in range(1, 4):
        fig.update_yaxes(range=[0, 1.15], row=1, col=ci)
    return fig


def plot_prediction_gauge(prob_fake, model_name):
    color = MODEL_COLORS[model_name]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob_fake * 100, 1),
        number=dict(suffix="%", font=dict(family="Syne", size=28, color=color)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#7c82a3", tickfont=dict(size=10)),
            bar=dict(color=color, thickness=0.6),
            bgcolor="#161a2b",
            borderwidth=0,
            steps=[
                dict(range=[0, 40],  color="rgba(46,204,113,.15)"),
                dict(range=[40, 60], color="rgba(245,200,66,.1)"),
                dict(range=[60, 100],color="rgba(255,77,109,.15)"),
            ],
            threshold=dict(line=dict(color="#fff", width=2), thickness=0.75, value=50),
        ),
        title=dict(text=MODEL_ABBR[model_name], font=dict(family="Syne", size=14, color="#7c82a3")),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaf0"),
        height=200,
        margin=dict(t=30, b=10, l=20, r=20),
    )
    return fig


def plot_radar(results):
    categories = ["Accuracy", "Precision\n(Real)", "Recall\n(Real)",
                  "Precision\n(Fake)", "Recall\n(Fake)", "F1\n(Fake)"]
    fig = go.Figure()
    for name, res in results.items():
        r = res["report"]
        vals = [
            res["accuracy"],
            r["0"]["precision"], r["0"]["recall"],
            r["1"]["precision"], r["1"]["recall"],
            r["1"]["f1-score"],
        ]
        vals_closed = vals + [vals[0]]
        cats_closed = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals_closed, theta=cats_closed,
            fill="toself",
            fillcolor=f"{MODEL_COLORS[name]}26",
            line=dict(color=MODEL_COLORS[name], width=2),
            name=name,
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8eaf0", family="DM Sans"),
        title=dict(text="Model Comparison Radar", font=dict(family="Syne", size=16)),
        polar=dict(
            bgcolor="#10131f",
            angularaxis=dict(color="#7c82a3", gridcolor="#252a42"),
            radialaxis=dict(visible=True, range=[0, 1], color="#7c82a3", gridcolor="#252a42"),
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        height=450,
        margin=dict(t=60, b=10, l=10, r=10),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                background:linear-gradient(90deg,#5b6ef5,#e84393);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                margin-bottom:.25rem;">🛡️ FakeShield</div>
    <div style="color:#7c82a3;font-size:.8rem;margin-bottom:1.5rem;">
        AI-powered news credibility analysis
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🔍 Detect News", "📊 Model Analytics", "🧪 Batch Analysis"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("<div style='color:#7c82a3;font-size:.8rem;'>Models</div>", unsafe_allow_html=True)
    for name, color in MODEL_COLORS.items():
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:.5rem;margin:.3rem 0;'>"
            f"<span style='width:10px;height:10px;border-radius:50%;background:{color};display:inline-block;'></span>"
            f"<span style='font-size:.82rem;color:#e8eaf0;'>{name}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("<div style='color:#7c82a3;font-size:.75rem;text-align:center;'>Built with Streamlit + scikit-learn</div>",
                unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS  (with pretty progress)
# ─────────────────────────────────────────────────────────────────────────────
if "trained_models" not in st.session_state:
    with st.spinner("⚡ Training models…"):
        prog = st.progress(0)
        time.sleep(0.3); prog.progress(25)
        models_dict, vectorizer, results, test_data = load_and_train()
        prog.progress(75); time.sleep(0.2); prog.progress(100); prog.empty()
        st.session_state.trained_models  = models_dict
        st.session_state.vectorizer      = vectorizer
        st.session_state.results         = results
        st.session_state.test_data       = test_data

models_dict = st.session_state.trained_models
vectorizer  = st.session_state.vectorizer
results     = st.session_state.results


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: DETECT NEWS
# ─────────────────────────────────────────────────────────────────────────────
if "Detect" in page:
    st.markdown("""
    <div class="hero">
      <div class="badge">🛡️ AI Detection Engine</div>
      <div class="hero-title">FakeShield</div>
      <div class="hero-sub">Paste any news article or headline — four independent classifiers will analyse it instantly.</div>
    </div>
    """, unsafe_allow_html=True)

    news_input = st.text_area(
        "News Text",
        placeholder="Paste your news article, headline, or paragraph here…",
        height=200,
        label_visibility="collapsed",
    )

    col_btn, col_clear = st.columns([1, 6])
    with col_btn:
        analyse = st.button("🔍 Analyse", use_container_width=True)
    with col_clear:
        if st.button("✕ Clear", use_container_width=False):
            news_input = ""

    if analyse and news_input.strip():
        with st.spinner("Analysing…"):
            time.sleep(0.4)
            preds = predict_news(news_input, models_dict, vectorizer)

        # ── Majority verdict ────────────────────────────────────────────────
        votes = [v["label"] for v in preds.values()]
        majority = "Fake" if votes.count("Fake") >= 2 else "Real"
        avg_fake = np.mean([v["prob_fake"] for v in preds.values()])

        if majority == "Fake":
            st.markdown(f"""
            <div class="verdict-box fake">
              <div class="verdict-emoji">🚨</div>
              <div class="verdict-title">Likely Fake News</div>
              <div class="verdict-desc">
                Majority consensus: <b>{votes.count('Fake')}/4</b> models flagged this as fake.
                Average fake-probability: <b>{avg_fake*100:.1f}%</b>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-box real">
              <div class="verdict-emoji">✅</div>
              <div class="verdict-title">Likely Real News</div>
              <div class="verdict-desc">
                Majority consensus: <b>{votes.count('Real')}/4</b> models consider this credible.
                Average fake-probability: <b>{avg_fake*100:.1f}%</b>
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Per-Model Breakdown</div><hr class="section-rule">', unsafe_allow_html=True)

        # ── Gauge row ───────────────────────────────────────────────────────
        gcols = st.columns(4)
        for idx, (name, res) in enumerate(preds.items()):
            with gcols[idx]:
                st.plotly_chart(plot_prediction_gauge(res["prob_fake"], name),
                                use_container_width=True, config={"displayModeBar": False})

        # ── Confidence bars ─────────────────────────────────────────────────
        for name, res in preds.items():
            color   = MODEL_COLORS[name]
            label   = res["label"]
            pf      = res["prob_fake"] * 100
            pr      = res["prob_real"] * 100
            verdict = "🔴 FAKE" if label == "Fake" else "🟢 REAL"
            st.markdown(f"""
            <div style="background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
                        padding:1rem 1.3rem;margin-bottom:.75rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.7rem;">
                <span style="font-family:'Syne',sans-serif;font-weight:700;color:{color};">{name}</span>
                <span style="font-size:.85rem;font-weight:600;color:{'#ff4d6d' if label=='Fake' else '#2ecc71'};">{verdict}</span>
              </div>
              <div class="conf-label"><span>Real</span><span>{pr:.1f}%</span></div>
              <div class="conf-bar-bg"><div class="conf-bar" style="width:{pr:.1f}%;background:#2ecc71;"></div></div>
              <div style="height:.5rem;"></div>
              <div class="conf-label"><span>Fake</span><span>{pf:.1f}%</span></div>
              <div class="conf-bar-bg"><div class="conf-bar" style="width:{pf:.1f}%;background:#ff4d6d;"></div></div>
            </div>
            """, unsafe_allow_html=True)

        # ── Voting summary donut ─────────────────────────────────────────────
        real_votes = votes.count("Real")
        fake_votes = votes.count("Fake")
        donut = go.Figure(go.Pie(
            labels=["Real", "Fake"],
            values=[real_votes, fake_votes],
            hole=0.65,
            marker=dict(colors=["#2ecc71", "#ff4d6d"],
                        line=dict(color="#0a0c14", width=3)),
            textinfo="label+value",
            textfont=dict(family="Syne", size=14),
        ))
        donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e8eaf0", family="DM Sans"),
            title=dict(text="Model Vote Distribution", font=dict(family="Syne", size=15)),
            showlegend=False,
            height=300,
            margin=dict(t=50, b=0, l=0, r=0),
            annotations=[dict(text=f"{'FAKE' if fake_votes>real_votes else 'REAL'}",
                              x=0.5, y=0.5, font=dict(size=18, family="Syne", color="#e8eaf0"),
                              showarrow=False)],
        )
        st.plotly_chart(donut, use_container_width=True, config={"displayModeBar": False})

    elif analyse:
        st.warning("Please enter some news text first.")

    else:
        st.markdown("""
        <div class="info-box">
          💡 Paste any news article or snippet above and click <b>Analyse</b>. All four models will vote on its credibility
          and you'll see confidence scores, gauge meters, and a voting summary.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif "Analytics" in page:
    st.markdown('<div class="section-title">Model Performance Analytics</div><hr class="section-rule">', unsafe_allow_html=True)

    # ── Top-level metrics ───────────────────────────────────────────────────
    best_name = max(results, key=lambda n: results[n]["accuracy"])
    best_acc  = results[best_name]["accuracy"]
    avg_acc   = np.mean([results[n]["accuracy"] for n in results])
    best_auc  = max(results[n]["roc_auc"] for n in results if results[n]["roc_auc"])

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card blue">
        <div class="metric-label">Best Accuracy</div>
        <div class="metric-value">{best_acc*100:.1f}%</div>
        <div style="color:#7c82a3;font-size:.78rem;margin-top:.3rem;">{best_name}</div>
      </div>
      <div class="metric-card pink">
        <div class="metric-label">Avg Accuracy</div>
        <div class="metric-value">{avg_acc*100:.1f}%</div>
        <div style="color:#7c82a3;font-size:.78rem;margin-top:.3rem;">across 4 models</div>
      </div>
      <div class="metric-card teal">
        <div class="metric-label">Best ROC AUC</div>
        <div class="metric-value">{best_auc:.3f}</div>
        <div style="color:#7c82a3;font-size:.78rem;margin-top:.3rem;">discriminative power</div>
      </div>
      <div class="metric-card gold">
        <div class="metric-label">Models Trained</div>
        <div class="metric-value">4</div>
        <div style="color:#7c82a3;font-size:.78rem;margin-top:.3rem;">LR · DT · GBC · RFC</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Accuracy & ROC", "🎯 Confusion Matrices", "📐 Precision / Recall", "🕸️ Radar"])

    with tab1:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(plot_accuracy_bar(results), use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.plotly_chart(plot_roc_all(results), use_container_width=True, config={"displayModeBar": False})

    with tab2:
        r1c1, r1c2 = st.columns(2)
        r2c1, r2c2 = st.columns(2)
        cols_pairs = [(r1c1, "Logistic Regression"), (r1c2, "Decision Tree"),
                      (r2c1, "Gradient Boosting"),   (r2c2, "Random Forest")]
        for col, name in cols_pairs:
            with col:
                st.plotly_chart(plot_confusion(results[name]["cm"], name),
                                use_container_width=True, config={"displayModeBar": False})

    with tab3:
        st.plotly_chart(plot_precision_recall(results), use_container_width=True, config={"displayModeBar": False})

    with tab4:
        st.plotly_chart(plot_radar(results), use_container_width=True, config={"displayModeBar": False})

    # ── Detailed table ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title" style="margin-top:1.5rem;">Detailed Metrics Table</div><hr class="section-rule">', unsafe_allow_html=True)
    table_rows = []
    for name, res in results.items():
        r = res["report"]
        table_rows.append({
            "Model":            name,
            "Accuracy":         f"{res['accuracy']*100:.2f}%",
            "Precision (Real)": f"{r['0']['precision']:.3f}",
            "Recall (Real)":    f"{r['0']['recall']:.3f}",
            "F1 (Real)":        f"{r['0']['f1-score']:.3f}",
            "Precision (Fake)": f"{r['1']['precision']:.3f}",
            "Recall (Fake)":    f"{r['1']['recall']:.3f}",
            "F1 (Fake)":        f"{r['1']['f1-score']:.3f}",
            "ROC AUC":          f"{res['roc_auc']:.4f}" if res['roc_auc'] else "—",
        })
    st.dataframe(pd.DataFrame(table_rows).set_index("Model"), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: BATCH ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif "Batch" in page:
    st.markdown('<div class="section-title">Batch News Analysis</div><hr class="section-rule">', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
      📋 Enter multiple news articles separated by <b>---</b> (three dashes). Each article will be analysed
      independently and a summary table + distribution chart will be shown.
    </div>""", unsafe_allow_html=True)

    batch_input = st.text_area(
        "Batch Input",
        placeholder="Paste article 1 here\n---\nPaste article 2 here\n---\nPaste article 3 here",
        height=280,
        label_visibility="collapsed",
    )

    if st.button("🔍 Analyse All", use_container_width=False):
        articles = [a.strip() for a in batch_input.split("---") if a.strip()]
        if not articles:
            st.warning("Please enter at least one article separated by ---")
        else:
            with st.spinner(f"Analysing {len(articles)} article(s)…"):
                batch_results = []
                prog = st.progress(0)
                for i, art in enumerate(articles):
                    preds = predict_news(art, models_dict, vectorizer)
                    votes = [v["label"] for v in preds.values()]
                    majority = "Fake" if votes.count("Fake") >= 2 else "Real"
                    avg_fake = np.mean([v["prob_fake"] for v in preds.values()])
                    batch_results.append({
                        "#":          i + 1,
                        "Snippet":    art[:80] + ("…" if len(art) > 80 else ""),
                        "Verdict":    majority,
                        "Fake Score": f"{avg_fake*100:.1f}%",
                        "LR":         preds["Logistic Regression"]["label"],
                        "DT":         preds["Decision Tree"]["label"],
                        "GBC":        preds["Gradient Boosting"]["label"],
                        "RFC":        preds["Random Forest"]["label"],
                        "_avg_fake":  avg_fake,
                    })
                    prog.progress(int((i + 1) / len(articles) * 100))

            df_batch = pd.DataFrame(batch_results)

            # ── Distribution pie ─────────────────────────────────────────────
            fake_count = (df_batch["Verdict"] == "Fake").sum()
            real_count = (df_batch["Verdict"] == "Real").sum()
            pie = go.Figure(go.Pie(
                labels=["Real", "Fake"],
                values=[real_count, fake_count],
                hole=0.55,
                marker=dict(colors=["#2ecc71", "#ff4d6d"],
                            line=dict(color="#0a0c14", width=3)),
                textfont=dict(family="Syne", size=14),
            ))
            pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8eaf0", family="DM Sans"),
                title=dict(text="Batch Verdict Distribution", font=dict(family="Syne", size=15)),
                height=320,
                margin=dict(t=50, b=0, l=0, r=0),
            )

            # ── Fake-score horizontal bar ────────────────────────────────────
            hbar = go.Figure(go.Bar(
                y=[f"Article {r['#']}" for _, r in df_batch.iterrows()],
                x=[r["_avg_fake"] * 100 for _, r in df_batch.iterrows()],
                orientation="h",
                marker=dict(
                    color=[r["_avg_fake"] for _, r in df_batch.iterrows()],
                    colorscale=[[0, "#2ecc71"], [0.5, "#f5c842"], [1, "#ff4d6d"]],
                    showscale=False,
                ),
                text=[f"{r['_avg_fake']*100:.1f}%" for _, r in df_batch.iterrows()],
                textposition="outside",
                textfont=dict(color="#e8eaf0", size=12),
            ))
            hbar.update_layout(
                **PLOTLY_THEME,
                title="Fake-News Score per Article",
                xaxis=dict(range=[0, 120], title="Fake %", **PLOTLY_THEME["xaxis"]),
                height=max(280, len(articles) * 50 + 80),
                margin=dict(t=50, b=10, l=10, r=60),
            )

            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(pie,  use_container_width=True, config={"displayModeBar": False})
            with c2:
                st.plotly_chart(hbar, use_container_width=True, config={"displayModeBar": False})

            # ── Table ────────────────────────────────────────────────────────
            st.markdown('<div class="section-title">Results Table</div><hr class="section-rule">', unsafe_allow_html=True)
            display_df = df_batch.drop(columns=["_avg_fake"]).set_index("#")
            st.dataframe(display_df, use_container_width=True)