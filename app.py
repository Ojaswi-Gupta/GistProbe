from flask import Flask, render_template, request, send_file
import pandas as pd
import uuid
import os

from crawler import scrape_url
from analyser import clean_text_data
from clustering import perform_clustering

app = Flask(__name__)
active_files = {}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    try:
        url = request.form["url"]
        
        # Run Pipeline
        df = scrape_url(url)
        df = clean_text_data(df)
        df, cluster_counts = perform_clustering(df)

        # Unique session for CSV
        session_id = str(uuid.uuid4())[:8]
        filename = f"results_{session_id}.csv"
        df.to_csv(filename, index=False)
        active_files['latest'] = filename

        # table: Show 'text' and 'cluster' (number)
        display_df = df[["text", "cluster"]].rename(columns={"cluster": "Cluster"})
        table_html = display_df.to_html(classes="table table-hover", index=False)

        # UPDATED: Pass url=url back to the template
        return render_template(
            "index.html", 
            table=table_html, 
            cluster_counts=cluster_counts, 
            url=url
        )

    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/download")
def download():
    path = active_files.get('latest')
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "File not found."

if __name__ == "__main__":
    app.run(debug=True)