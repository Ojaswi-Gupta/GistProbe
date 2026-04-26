# GistProbe — NLP-Driven Web Content Analyzer

> An end-to-end NLP pipeline that crawls any URL, analyses text using unsupervised machine learning, and surfaces semantic clusters, sentiment scores, and extractive summaries through an interactive web dashboard.

**Built by Ojaswi Gupta**

---

## ✨ What It Does

Paste any URL — a news article, blog, Wikipedia page, or financial site — and GistProbe runs it through a 5-stage NLP pipeline:

| Stage | What Happens |
|---|---|
| **1. Crawl** | BeautifulSoup scrapes the DOM with rotating user-agents & robots.txt compliance |
| **2. Extract** | Multi-tier text extraction (targeted DOM → trafilatura fallback) |
| **3. Analyse** | NLTK deduplication + TextBlob sentiment polarity & subjectivity scoring |
| **4. Cluster** | TF-IDF vectorization + K-Means with automatic `k` selection via Silhouette Score |
| **5. Summarize** | Extractive NLP summary using TF-IDF document density ranking |

---

## 🧠 Key Technical Decisions

### Silhouette Score for Optimal `k`
Rather than hardcoding a fixed number of clusters, GistProbe mathematically evaluates **k = 2 to 10** and selects the value with the highest Silhouette Score. A score closer to `1.0` indicates well-separated, meaningful clusters.

```python
for k in range(2, max_k):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels)  # Evaluate separation quality
    if score > best_score:
        best_score, best_k = score, k
```

### No LLM API — Pure NLP Math
GistProbe deliberately avoids paid LLM APIs (OpenAI, Claude, etc.). All summarization and clustering is done using classical NLP and linear algebra. This keeps it free, fast, and interpretable.

### Near-Duplicate Removal
Uses `SequenceMatcher` to compute string similarity ratios, filtering out paragraphs with >85% overlap before clustering — ensuring cluster quality isn't diluted by repeated boilerplate.

---

## 🛠️ Tech Stack

**Backend**
- `Python 3.11`, `Flask` — web framework & routing
- `BeautifulSoup4`, `Requests`, `Trafilatura` — multi-tier web crawling
- `NLTK` — text tokenization & cleaning
- `TextBlob` — sentiment polarity & subjectivity scoring
- `scikit-learn` — TF-IDF Vectorizer, K-Means, Silhouette Score
- `Pandas`, `NumPy` — data manipulation

**Frontend**
- `Bootstrap 5` — responsive layout
- `Chart.js` — interactive donut charts (sentiment & topic distribution)
- `DataTables.js` — filterable, paginated results table
- Glassmorphism CSS + cursor-glow effect (vanilla JS `requestAnimationFrame`)

---

## 🖥️ Screenshots

> Probe BBC News, The Verge, Wikipedia, or any URL. Results appear after a 5-stage live terminal animation.

| Homepage | Results Dashboard |
|---|---|
| URL input + feature grid + tech stack | Sentiment chart + cluster donut + AI summary |

---

## 🚀 Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/Ojaswi-Gupta/GistProbe.git
cd GistProbe

# 2. Create & activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download required NLTK data (first run only)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 5. Start the server
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 🚢 Deployment

> ⚠️ **Do NOT deploy to Vercel or Netlify.** This app runs heavy NLP + ML tasks that exceed serverless 10s timeout limits and 250MB bundle limits.

**Recommended: Render or Railway**

```bash
# Start command (add gunicorn to requirements.txt first)
gunicorn app:app --timeout 120 --workers 2
```

1. Connect your GitHub repo to [Render](https://render.com)
2. Set **Build Command:** `pip install -r requirements.txt`
3. Set **Start Command:** `gunicorn app:app --timeout 120`
4. Select a **Standard** instance (not free tier — needs >512MB RAM for sklearn)

---

## 📂 Project Structure

```
GistProbe/
├── app.py              # Flask routes & pipeline orchestration
├── crawler.py          # Multi-tier web scraper (BS4 + trafilatura)
├── analyser.py         # Text cleaning, deduplication & sentiment scoring
├── clustering.py       # TF-IDF vectorization, K-Means, Silhouette Score
├── templates/
│   └── index.html      # Full-stack SPA-like dashboard (Bootstrap + Chart.js)
├── requirements.txt
└── README.md
```

---

## 📊 What the Dashboard Shows

- **AI Executive Summary** — top 3 most information-dense sentences ranked by TF-IDF score
- **Overall Sentiment** — donut chart (Positive / Negative / Neutral) — click to filter table
- **Topic Distribution** — donut chart of K-Means cluster sizes
- **Cluster Dominance** — ranked list of topic clusters with item counts
- **Extracted Insights Table** — full paginated/filterable DataTable of all scraped text
- **Execution Logs Modal** — simulated NLP pipeline logs + model diagnostics (Silhouette Score, Optimal k, Vocabulary Size)

---

## 📝 Resume Bullet

> *"Built GistProbe, a full-stack NLP web application that crawls and semantically clusters web content using K-Means + TF-IDF, with automated Silhouette Score optimization for k selection, real-time sentiment analysis via TextBlob, and an interactive Flask dashboard featuring Chart.js visualizations and DataTables."*

---

## 📜 License

Created by **Ojaswi Gupta**. All rights reserved.
