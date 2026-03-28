<div align="center">

# 🛡️ FakeShield — AI Fake News Detector

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

<br/>

> **FakeShield** is a beautifully themed, interactive Streamlit web app that uses four independent machine-learning classifiers to detect fake news in real time — with rich visualisations, per-model confidence gauges, ROC curves, confusion matrices, and batch analysis support.

<br/>

---

</div>

## 📸 Features at a Glance

| 🔍 Single Article Detection | 📊 Model Analytics | 🧪 Batch Analysis |
|---|---|---|
| Paste any article and get an instant verdict | Accuracy bars, ROC curves, confusion matrices, radar chart | Analyse multiple articles separated by `---` |
| 4 model votes with confidence percentages | Precision / Recall / F1 breakdown per class | Fake-score horizontal bar chart per article |
| Interactive gauge meters per model | Detailed metrics comparison table | Vote distribution pie chart + results table |
| Majority-vote FAKE 🚨 / REAL ✅ verdict | Side-by-side model comparison | Downloadable results |

---

## 🧠 Models Used

| Model | Abbreviation | Colour |
|---|---|---|
| Logistic Regression | LR | 🔵 Blue |
| Decision Tree | DT | 🩷 Pink |
| Gradient Boosting Classifier | GBC | 🟢 Teal |
| Random Forest Classifier | RFC | 🟡 Gold |

All four models are trained with a **TF-IDF vectorizer** (`max_features=5000`) on text that is pre-cleaned (lowercasing, URL removal, punctuation stripping, digit removal).

---

## 🗂️ Project Structure

```
FakeShield/
│
├── fake_news_detector_app.py   # Main Streamlit application
├── fake-news-detection.ipynb   # Original training notebook (EDA + model training)
├── requirements.txt            # Python dependencies
└── README.md                   # You are here
```

---

## ⚡ Quick Start

### 1 — Clone the repository

```bash
git clone https://github.com/your-username/fakeshield.git
cd fakeshield
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### 4 — Run the app

```bash
streamlit run fake_news_detector_app.py
```

The app will open automatically at **http://localhost:8501**

---

## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | ≥ 1.32.0 | Web application framework |
| `scikit-learn` | ≥ 1.4.0 | ML models + TF-IDF vectorizer |
| `pandas` | ≥ 2.0.0 | Data handling & results table |
| `numpy` | ≥ 1.26.0 | Numerical computations |
| `plotly` | ≥ 5.20.0 | Interactive charts & visualisations |

All standard library modules used (`re`, `string`, `time`) require no installation.

---

## 🖥️ App Pages

### 🔍 Detect News
- Paste any news article, headline, or paragraph into the text box
- Click **Analyse** to run all four models simultaneously
- View:
  - **Majority verdict** banner (FAKE 🚨 or REAL ✅) with average fake-probability
  - **Gauge meters** for each model showing fake-news confidence (0–100%)
  - **Confidence bars** showing Real vs Fake probability per model
  - **Vote distribution donut** chart summarising all four model votes

### 📊 Model Analytics
- **Top metrics cards** — Best accuracy, average accuracy, best ROC AUC, model count
- **Accuracy Bar Chart** — Side-by-side accuracy comparison
- **ROC Curves** — All four models overlaid with AUC scores
- **Confusion Matrices** — Heatmaps for each classifier
- **Precision / Recall / F1** — Grouped bar charts by class (Real vs Fake)
- **Radar Chart** — Multi-metric spider chart comparing all models
- **Detailed Metrics Table** — Every metric in one downloadable table

### 🧪 Batch Analysis
- Separate multiple articles with `---`
- See a **fake-score horizontal bar chart** coloured red→green per article
- **Verdict distribution pie** chart for the whole batch
- Full **results table** with per-model predictions

---

## 🔧 Text Preprocessing Pipeline

```python
def wordopt(text):
    text = text.lower()                          # lowercase
    text = re.sub(r'\[.*?\]', '', text)          # remove bracketed content
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # remove URLs
    text = re.sub(r'<.*?>+', '', text)           # strip HTML tags
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\]^_`{|}~]', '', text)  # punctuation
    text = re.sub(r'\n', ' ', text)              # newlines → spaces
    text = re.sub(r'\w*\d\w*', '', text)         # remove words containing digits
    return text
```

---

## 🎨 Design Philosophy

FakeShield uses a **dark space aesthetic** with:
- **Syne** — display / heading font (geometric, futuristic)
- **DM Sans** — body font (clean, readable)
- A gradient colour palette: `#5b6ef5` (blue) · `#e84393` (pink) · `#00d4aa` (teal) · `#f5c842` (gold)
- Transparent Plotly charts that blend seamlessly into the dark background
- CSS gradient mesh glows and layered card surfaces

---

## 📊 Sample Output

```
News Input:
"Government officials confirmed the new policy after senate approval..."

Result:
✅ Likely Real News
  → LR:  Real  (fake prob: 12.3%)
  → DT:  Real  (fake prob: 18.7%)
  → GBC: Real  (fake prob: 9.1%)
  → RFC: Real  (fake prob: 11.5%)

Majority: 4/4 models say REAL | Avg fake score: 12.9%
```

---

## 🚀 Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repo → set **Main file** to `fake_news_detector_app.py`
5. Click **Deploy** — your app is live in seconds!

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the project
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 🙏 Acknowledgements

- Dataset inspiration: [Fake and Real News Dataset – Kaggle](https://www.kaggle.com/datasets/jainpooja/fake-news-detection)
- [Streamlit](https://streamlit.io/) — for making ML apps effortless
- [scikit-learn](https://scikit-learn.org/) — for the ML toolkit
- [Plotly](https://plotly.com/) — for beautiful interactive charts

---

<div align="center">

Made with ❤️ and Python

⭐ Star this repo if you found it useful!

</div>
