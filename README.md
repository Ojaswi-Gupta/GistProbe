# GistProbe: NLP-Driven Web Crawling & Clustering Engine

GistProbe is an advanced, resilient web scraping and NLP clustering application built by **Team Flux**. It acts as a data ingestion pipeline that autonomously extracts meaningful text from any given URL (news, articles, blogs), cleans it, and uses Machine Learning to group the content into semantic clusters.

## 🚀 Features

- **Multi-Tiered Extraction Engine:**
  - **Tier 1 (Targeted DOM Parsing):** Smartly extracts text from headings, `<a>` tags, `<article>` containers, and specific CSS classes using `BeautifulSoup`.
  - **Tier 2 (Deep Extraction):** Falls back to `trafilatura`—a state-of-the-art NLP library—to rip full article bodies when complex nested HTML is encountered.
- **Resilient & Evasive:** Uses rotating, realistic browser User-Agents and HTTP retry logic (with exponential backoff) to bypass basic anti-bot systems.
- **Smart Junk Filtering:** Algorithms walk up the DOM tree to identify and strip out navigational menus, footers, advertisements, and cookie banners.
- **Intelligent NLP Clustering:**
  - Performs text cleaning while preserving Unicode characters and critical numbers.
  - Removes near-duplicates using mathematical Sequence Matching (>85% similarity threshold).
  - Uses **TF-IDF Vectorization** and **K-Means Clustering** to group text.
  - Automatically calculates the optimal `K` (number of clusters) using the **Silhouette Score**.
  - Generates human-readable cluster names using Bigrams/Trigrams.
- **Legal & Polite:** Respects `robots.txt` rules and implements rate-limiting delays.

## 🛠️ Tech Stack

- **Backend:** Python, Flask
- **Scraping:** Requests, BeautifulSoup4, lxml, Trafilatura
- **Machine Learning (NLP):** Scikit-learn (TF-IDF, K-Means, Silhouette Score), Pandas, Numpy
- **Frontend:** HTML5, Bootstrap 5, Chart.js

## 💻 Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ojaswi-Gupta/GistProbe.git
   cd GistProbe
   ```

2. **Create a virtual environment:**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your web browser.

## 🚢 Deployment (Render / Railway)

This application performs heavy NLP and Machine Learning tasks. It is **NOT recommended** for serverless platforms like Vercel or Netlify due to their strict 10-second timeout limits and 250MB size limits.

**To deploy on Render:**
1. Connect your GitHub repository to Render.
2. Select **Web Service**.
3. Set the Build Command:
   ```bash
   pip install -r requirements.txt
   ```
4. Set the Start Command:
   ```bash
   gunicorn app:app --timeout 120
   ```
   *(Note: You will need to add `gunicorn` to your `requirements.txt`)*

## 📝 License

Created by **Team Flux**. All rights reserved.
