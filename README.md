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

## 🚀 NEW: Advanced Features

GistProbe has recently been upgraded with powerful LLM integrations and interactive visualizers to push the boundaries of media analysis:

### 🤖 Chat with Website & Fact Check Mode
Chat directly with the scraped contents of any webpage using **Llama-3 (via Groq API)**. Enable the **Fact Check Mode** toggle to force the AI to cross-reference claims in the article against real-world knowledge, highlighting biases and inaccuracies in real time.

### 🕸️ Interactive Entity Knowledge Graph
Entities (People, Organizations, Locations) extracted via **spaCy** are mapped into a beautiful interactive network graph using **vis.js**. The physics engine groups entities based on sentence co-occurrences. Clicking on an entity node instantly filters the raw data tables to show exactly what the article says about them!

### ⚖️ Multi-URL Debate Mode
Compare two URLs side-by-side. The Llama-3 AI acts as an Expert Media Analyst, reading both articles and generating an **Executive Comparison Summary** that highlights differences in tone, framing, biases, and factual omissions between the two sources.

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

### Near-Duplicate Removal
Uses `SequenceMatcher` to compute string similarity ratios, filtering out paragraphs with >85% overlap before clustering — ensuring cluster quality isn't diluted by repeated boilerplate.

---

## 🛠️ Tech Stack

**Backend & AI**
- `Python 3.11`, `Flask` — web framework & routing
- `BeautifulSoup4`, `Requests`, `Trafilatura` — multi-tier web crawling
- `NLTK`, `spaCy` — text tokenization, cleaning & Named Entity Recognition (NER)
- `Groq API (Llama-3.1)` — conversational agent and media debate analysis
- `TextBlob` — sentiment polarity & subjectivity scoring
- `scikit-learn` — TF-IDF Vectorizer, K-Means, Silhouette Score
- `wordcloud` — TF-IDF visual representation
- `Pandas`, `NumPy` — data manipulation

**Frontend**
- `Bootstrap 5` — responsive layout
- `vis.js` — interactive physics-based network graphs
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

# 4. Download required spaCy & NLTK models (first run only)
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 5. Set up your Groq API Key
export GROQ_API_KEY="your-api-key-here"

# 6. Start the server
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

### ⚡ Demo Mode

After running at least one successful probe, visit **http://127.0.0.1:5000/demo** to instantly load the cached result — no waiting for a crawl. Perfect for live demos and interviews.

### 🧪 Running Tests

```bash
python -m unittest tests -v
```

11 unit tests cover the core NLP pipeline: text cleaning, deduplication, sentiment analysis, clustering, and extractive summarization.

---

## 🚢 Deployment

> ⚠️ **Do NOT deploy to Vercel or Netlify.** This app runs heavy NLP + ML tasks that exceed serverless 10s timeout limits and 250MB bundle limits.

**Recommended: Render or Railway**

1. Connect your GitHub repo to [Render](https://render.com)
2. Add your `GROQ_API_KEY` to the Environment Variables.
3. Set **Build Command:** `pip install -r requirements.txt && python -m spacy download en_core_web_sm`
4. Set **Start Command:** `gunicorn app:app --timeout 120`
5. Select a **Standard** instance (not free tier — needs >512MB RAM for sklearn)

---

## 📂 Project Structure

```
GistProbe/
├── app.py              # Flask routes & pipeline orchestration
├── crawler.py          # Multi-tier web scraper (BS4 + trafilatura)
├── analyser.py         # Text cleaning, deduplication & sentiment scoring
├── clustering.py       # TF-IDF vectorization, K-Means, Silhouette Score
├── ner.py              # Named Entity Recognition (spaCy)
├── wordcloud_gen.py    # TF-IDF visualization generator
├── tests.py            # Unit tests for the core NLP pipeline
├── templates/
│   ├── index.html      # Full-stack SPA-like dashboard (Bootstrap + Chart.js)
│   └── compare.html    # Side-by-side URL comparison dashboard
├── requirements.txt
└── README.md
```

---

## 📊 What the Dashboard Shows

- **Interactive Knowledge Graph** — physics-based network of how entities connect.
- **Chatbot / Fact Check Mode** — interrogate the webpage's claims in real time.
- **Compare Mode** — side-by-side Llama-3 debate analysis of two different URLs.
- **AI Executive Summary** — top 3 most information-dense sentences ranked by TF-IDF score.
- **Word Cloud & Sentiment** — visual representation of TF-IDF term importance and sentiment donut.
- **Topic Distribution** — donut chart of K-Means cluster sizes.
- **Extracted Insights Table** — full paginated/filterable DataTable of all scraped text.

---

## 📝 Resume Bullet

> *"Built GistProbe, a full-stack NLP web application that crawls and semantically clusters web content using K-Means + TF-IDF, featuring a spaCy+vis.js Interactive Knowledge Graph and Llama-3 (Groq API) integration for real-time web chat, fact-checking, and multi-URL media debate analysis."*

---

## 📜 License

Created by **Ojaswi Gupta**. All rights reserved.
