from flask import Flask, render_template, request, send_file
import pandas as pd
import uuid
import os
import glob

from crawler import scrape_url
from analyser import clean_text_data
from clustering import perform_clustering

app = Flask(__name__)
active_files = {}


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

        # Phase 3: Cluster
        df, cluster_counts = perform_clustering(df)

        # Save results
        session_id = str(uuid.uuid4())[:8]
        filename = f"results_{session_id}.csv"
        df.to_csv(filename, index=False)
        active_files["latest"] = filename

        # Build display table: show 'text' and 'cluster' number
        display_df = df[["text", "cluster"]].rename(columns={"cluster": "Cluster"})
        table_html = display_df.to_html(classes="table table-hover", index=False)

        return render_template(
            "index.html",
            table=table_html,
            cluster_counts=cluster_counts,
            url=url,
            total_items=len(df),
        )

    except Exception as e:
        print(f"Pipeline error: {e}")
        return render_template(
            "index.html",
            error=f"Something went wrong: {str(e)}. Please check the URL and try again.",
            url=request.form.get("url", ""),
        )


@app.route("/download")
def download():
    path = active_files.get("latest")
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "File not found. Please run a probe first."


if __name__ == "__main__":
    app.run(debug=True)