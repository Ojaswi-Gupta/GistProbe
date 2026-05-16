from flask import Flask, render_template, request, send_file, redirect, url_for
import pandas as pd
import uuid
import os
import glob
import json

from crawler import scrape_url
from analyser import clean_text_data, compute_sentiment
from clustering import perform_clustering

app = Flask(__name__)
active_files = {}

CACHE_FILE = "_probe_cache.json"


def _save_cache(url, table_html, cluster_counts, sentiment_counts, avg_subjectivity, takeaways, metrics, total_items):
    """Save the last successful probe result to disk for instant demo loading."""
    try:
        cache = {
            "url": url,
            "table": table_html,
            "cluster_counts": cluster_counts,
            "sentiment_counts": sentiment_counts,
            "avg_subjectivity": avg_subjectivity,
            "takeaways": takeaways,
            "metrics": metrics,
            "total_items": total_items,
        }
        with open(CACHE_FILE, "w") as f:
            json.dump(cache, f)
        print(f"Cache saved for: {url}")
    except Exception as e:
        print(f"Cache save error: {e}")


def _load_cache():
    """Load the last probe result from disk cache."""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"Cache load error: {e}")
    return None


def _cleanup_old_results():
    """Remove old result CSV files to prevent disk clutter."""
    for f in glob.glob("results_*.csv"):
        try:
            os.remove(f)
        except OSError:
            pass


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    try:
        url = request.form["url"].strip()

        if not url:
            return render_template("index.html", error="Please enter a valid URL.")

        # Clean up old result files
        _cleanup_old_results()

        # --- Run the 3-Phase Pipeline ---
        # Phase 1: Crawl
        df = scrape_url(url)

        if df.empty:
            return render_template(
                "index.html",
                error="No content could be extracted from this URL. The site may be blocking requests, or the page has no readable text content.",
                url=url,
            )

        # Phase 2: Analyse
        df = clean_text_data(df)

        if df.empty or len(df) < 2:
            return render_template(
                "index.html",
                error="After cleaning and deduplication, not enough unique text was found. Try a different URL with more content.",
                url=url,
            )

        # Phase 2.5: Sentiment Analysis
        df, sentiment_counts, avg_subjectivity = compute_sentiment(df)

        # Phase 3: Cluster and Summarize
        df, cluster_counts, takeaways, metrics = perform_clustering(df)

        # Save results
        session_id = str(uuid.uuid4())[:8]
        filename = f"results_{session_id}.csv"
        df.to_csv(filename, index=False)
        active_files["latest"] = filename

        # Build display table
        if "Sentiment" in df.columns:
            display_df = df[["text", "cluster", "Sentiment"]].rename(columns={"cluster": "Cluster"})
        else:
            display_df = df[["text", "cluster"]].rename(columns={"cluster": "Cluster"})
            
        table_html = display_df.to_html(classes="table table-hover", table_id="resultsTable", index=False)

        # Cache the result for instant /demo loading
        _save_cache(url, table_html, cluster_counts, sentiment_counts, avg_subjectivity, takeaways, metrics, len(df))

        return render_template(
            "index.html",
            table=table_html,
            cluster_counts=cluster_counts,
            sentiment_counts=sentiment_counts,
            avg_subjectivity=avg_subjectivity,
            takeaways=takeaways,
            metrics=metrics,
            url=url,
            total_items=len(df),
            cached=False,
        )

    except Exception as e:
        print(f"Pipeline error: {e}")
        return render_template(
            "index.html",
            error=f"Something went wrong: {str(e)}. Please check the URL and try again.",
            url=request.form.get("url", ""),
        )


@app.route("/demo")
def demo():
    """Load the last successful probe result from cache — instant demo mode."""
    cache = _load_cache()
    if not cache:
        return redirect(url_for("home"))

    return render_template(
        "index.html",
        table=cache["table"],
        cluster_counts=cache["cluster_counts"],
        sentiment_counts=cache["sentiment_counts"],
        avg_subjectivity=cache["avg_subjectivity"],
        takeaways=cache["takeaways"],
        metrics=cache["metrics"],
        url=cache["url"],
        total_items=cache["total_items"],
        cached=True,
    )


@app.route("/download")
def download():
    path = active_files.get("latest")
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "File not found. Please run a probe first."


if __name__ == "__main__":
    app.run(debug=True)