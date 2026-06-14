# GistProbe: Real-Time Unsupervised Web NLP Analyzer

> **🎥 Watch the 1-Minute Video Demo Here: [Insert YouTube/Loom Link Here]**

**Built by Ojaswi Gupta** | **Domain:** Natural Language Processing (NLP) & Unsupervised Machine Learning

An end-to-end Machine Learning web application that dynamically crawls any URL, analyzes text using unsupervised machine learning, and surfaces semantic clusters, sentiment scores, and extractive summaries through an interactive web dashboard.

---

## 🌟 Why This Project Stands Out

GistProbe is not a standard wrapper around an API. It is a full-fledged NLP pipeline that builds custom datasets in real-time and applies mathematical heuristics to understand text:

- **Dynamic Dataset Generation:** Uses **Playwright** and **BeautifulSoup** to scrape the DOM dynamically, bypassing basic blocks.
- **Unsupervised Optimization:** Instead of hardcoding K-Means clusters, it dynamically evaluates **k = 2 to 10** and selects the value with the highest **Silhouette Score** for the specific webpage.
- **Intelligent Deduplication:** Uses `SequenceMatcher` to compute string similarity ratios, filtering out paragraphs with >85% overlap to ensure cluster quality.
- **RAG Generative AI Chat:** Integrates **Sentence-Transformers**, **FAISS Vector DB**, and **Llama-3.1 (via Groq API)**. The system vectorizes text, performs mathematical semantic searches, and implements a true Retrieval-Augmented Generation pipeline to answer questions without hallucinations.
- **MLOps Telemetry:** Integrates **Weights & Biases (WandB)** to securely log mathematically optimized k-values, Silhouette Scores, and vocabulary sizes during K-Means loops.
- **Background Automation:** Integrates **Flask-APScheduler** to autonomously re-scrape user-subscribed URLs daily, graphing subjectivity and sentiment shifts over time.
- **Audio Summaries:** Leverages the **gTTS API** to automatically generate downloadable MP3 spoken summaries of the abstractive text chunks.

---

## ⚙️ System Architecture & Data Flow

```mermaid
graph TD
    A[User Inputs URL or APScheduler Triggers] -->|Flask Route| B[Playwright Scraper]
    B -->|Raw HTML| C[BeautifulSoup Parser]
    C -->|Raw Text Nodes| D[Regex & NLTK Cleaner]
    D -->|>85% Similarity Filter| E[Cleaned Dataset]
    
    E --> F[Scikit-learn TF-IDF]
    F --> G[K-Means Clustering]
    G -->|Silhouette Score Optimization| H[Semantic Topic Clusters]
    
    E --> I[TextBlob Sentiment Analysis]
    I --> J[Polarity & Subjectivity Scores]
    
    E --> K[spaCy NER]
    K --> L[vis.js Knowledge Graph]
    
    H & J & L --> M[Flask UI & SQLite Database]
    M <--> N[FAISS Vector DB]
    N <--> Q[Groq Llama-3.1 RAG Engine]
    H --> O[gTTS Text-to-Speech API]
    G --> R[WandB MLOps Tracking]
    O --> P[Downloadable MP3 Audio]
```

### ML Pipeline Stages

| Stage | Component | What Happens |
|---|---|---|
| **1. Crawl** | Playwright & BeautifulSoup | Scrapes the DOM with rotating user-agents & extracts raw text nodes. |
| **2. Clean** | Python (Regex) & NLTK | Removes non-alphanumeric noise, normalizes text, and removes near-duplicates (>85% similarity). |
| **3. Sentiment**| TextBlob | Computes sentiment polarity & subjectivity scoring for every extracted sentence. |
| **4. Cluster** | scikit-learn (TF-IDF & K-Means) | Vectorizes text, computes optimal `k` via Silhouette Score, logs metrics to **WandB**, and assigns sentences to semantic clusters. |
| **5. Entity** | spaCy (en_core_web_sm) | Performs Named Entity Recognition (NER) to map People, Organizations, and Locations. |
| **6. Synthesize**| FAISS & Groq API | Computes Hugging Face embeddings, performs semantic similarity searches, and implements RAG to generate grounded abstractive AI summaries. |

---

## 💾 Database Schema

GistProbe uses **SQLAlchemy** with an SQLite database to cache NLP results and manage automated subscriptions.

**1. `User` Table** (OAuth handled via Authlib Google Login)
- `id` (Integer, Primary Key)
- `email` (String)
- `name` (String)

**2. `URLSubscription` Table** (Handles Background Automation)
- `id` (Integer, Primary Key)
- `user_id` (Foreign Key -> User.id)
- `url` (String)
- `frequency` (String) — *e.g., 'daily', 'weekly'*

**3. `ProbeResult` Table** (Caches expensive ML outputs)
- `id` (Integer, Primary Key)
- `user_id` (Foreign Key -> User.id)
- `url` (String)
- `timestamp` (DateTime)
- `total_items` (Integer)
- `avg_subjectivity` (Float)
- `results_json` (Text) — *Stores JSON blobs of K-Means cluster arrays, SpaCy entity graphs, and TF-IDF metrics for instant loading.*

---

## 🛠️ Tech Stack

**Machine Learning & NLP:**
- `scikit-learn` — TF-IDF Vectorizer, K-Means Clustering, Silhouette Score
- `spaCy` — Named Entity Recognition (NER)
- `NLTK` — Text tokenization & stop words
- `TextBlob` — Sentiment polarity & subjectivity scoring
- `Sentence-Transformers` & `FAISS` — Local embedding generation and Vector Database retrieval
- `Groq API (Llama-3.1)` — Conversational RAG agent and media debate analysis
- `Weights & Biases (WandB)` — MLOps experiment tracking and telemetry
- `gTTS API` — Text-to-Speech audio generation

**Backend & Automation:**
- `Python 3.11` & `Flask` — Web framework & orchestration
- `Flask-APScheduler` — Background job execution for automated tracking
- `Playwright` & `BeautifulSoup4` — Dynamic web crawling
- `SQLAlchemy` — ORM for Database management

**Frontend:**
- `Bootstrap 5` — Responsive layout with custom glassmorphism
- `vis.js` — Interactive physics-based network graphs (Knowledge Graph)
- `Chart.js` — Interactive donut and line charts (Sentiment tracking & Topic Distribution)

---

## 🚀 Advanced Features

### ⏱️ Automated Sentiment Tracking
Users can "subscribe" to specific URLs. A background APScheduler job automatically re-crawls these URLs daily without user intervention, plotting historical shifts in sentiment and subjectivity over time to detect changing narratives.

### 🕸️ Interactive Entity Knowledge Graph
Entities (People, Organizations, Locations) extracted via **spaCy** are mapped into an interactive network graph using **vis.js**. The physics engine groups entities based on sentence co-occurrences.

### 🤖 Chat with Website & Audio Summaries
Chat directly with the scraped contents using **Llama-3**, and instantly generate playable MP3 audio summaries of the article via the **gTTS** integration. Enable **Fact Check Mode** to force the AI to cross-reference claims against real-world knowledge.

---

## 📂 Project Structure

```text
GistProbe/
├── app.py              # Flask routes, pipeline orchestration, Schedulers & DB Models
├── crawler.py          # Playwright & BS4 Web Scraper
├── analyser.py         # Text cleaning, NLTK deduplication & TextBlob sentiment
├── clustering.py       # Scikit-learn TF-IDF, K-Means & Silhouette optimization
├── ner.py              # spaCy Named Entity Recognition engine
├── wordcloud_gen.py    # TF-IDF visual representation logic
├── audio_gen.py        # gTTS API text-to-speech script & file cleanup
├── tests.py            # Unit tests for the core ML pipeline
├── templates/
│   ├── index.html      # Main Dashboard View (33:67 Chatbot/Graph Layout)
│   └── history.html    # Tabular user history and active subscription manager
└── requirements.txt    # Strict environment dependencies
```

---

## ⚙️ Local Setup (Run it yourself!)

Follow these exact steps to run the complete NLP pipeline on your local machine:

**1. Clone the repository**
```bash
git clone https://github.com/Ojaswi-Gupta/GistProbe.git
cd GistProbe
```

**2. Create & activate a virtual environment**
```bash
# Mac/Linux:
python3 -m venv venv
source venv/bin/activate

# Windows:
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download ML Models & Browser Binaries (First run only)**
```bash
playwright install chromium
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

**5. Set up Environment Variables**
Create a `.env` file in the root directory and add your API keys. You can get a free API key from the [Groq Console](https://console.groq.com/).
```env
GROQ_API_KEY="your-groq-api-key-here"
FLASK_SECRET_KEY="any-random-string"
# Google OAuth (Optional, only needed if you want to test the login feature)
GOOGLE_CLIENT_ID="optional"
GOOGLE_CLIENT_SECRET="optional"
```

**6. Start the Server**
```bash
python app.py
```

**7. Access the App**
Open your browser and navigate to: **http://127.0.0.1:5000**

---

## 📝 Resume Bullet Example

> *"Built GistProbe, a full-stack NLP web application that dynamically scrapes and semantically clusters web content using K-Means + TF-IDF. Engineered an automated background pipeline with Flask-APScheduler to track sentiment shifts, featuring a spaCy Interactive Knowledge Graph, gTTS audio generation, and Llama-3 integration for real-time web chat."*

---

**License:** Created by Ojaswi Gupta. All rights reserved.
