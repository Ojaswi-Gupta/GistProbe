from flask import Flask, render_template, request, send_file, redirect, url_for, jsonify, Response, has_request_context
from werkzeug.middleware.proxy_fix import ProxyFix
import pandas as pd
import uuid
import os
import glob
import json
import time
import io
from dotenv import load_dotenv

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user, login_required
from authlib.integrations.flask_client import OAuth
from authlib.integrations.base_client.errors import OAuthError
from groq import Groq
from flask_apscheduler import APScheduler

from crawler import scrape_url
from analyser import clean_text_data, compute_sentiment
from clustering import perform_clustering, compute_similarity
from wordcloud_gen import generate_wordcloud, cleanup_old_wordclouds
from ner import extract_entities
from audio_gen import generate_summary_audio, cleanup_old_audio
from rag import build_faiss_index, retrieve_context

load_dotenv()

latest_anon_csvs = {}

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "fallback-secret-key")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///gistprobe.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

scheduler = APScheduler()
scheduler.init_app(app)
scheduler.start()

db = SQLAlchemy(app)

with app.app_context():
    db.create_all()
login_manager = LoginManager(app)
login_manager.login_view = 'login'

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv('GOOGLE_CLIENT_ID'),
    client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
    access_token_url='https://accounts.google.com/o/oauth2/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    userinfo_endpoint='https://www.googleapis.com/oauth2/v1/userinfo',
    client_kwargs={'scope': 'email profile'}
)


chat_contexts = {}
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None
except Exception as e:
    print(f"Failed to initialize Groq: {e}")
    groq_client = None

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(120))
    probes = db.relationship('ProbeResult', backref='user', lazy=True)
    subscriptions = db.relationship('Subscription', backref='user', lazy=True, cascade="all, delete-orphan")

class Subscription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    url = db.Column(db.String(2048), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    __table_args__ = (db.UniqueConstraint('user_id', 'url', name='_user_url_uc'),)

class ProbeResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    url = db.Column(db.String(2048), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    total_items = db.Column(db.Integer)
    avg_subjectivity = db.Column(db.Float)
    results_json = db.Column(db.Text)
    csv_data = db.Column(db.Text)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/login')
def login():
    redirect_uri = url_for('auth', _external=True)
    return google.authorize_redirect(redirect_uri)

@app.route('/login/callback')
def auth():
    try:
        token = google.authorize_access_token()
    except OAuthError:
        # User canceled the login prompt or access was denied
        return redirect(url_for('home'))
        
    userinfo = token.get('userinfo')
    if not userinfo:
        userinfo = google.get('userinfo').json()
    email = userinfo.get('email')
    name = userinfo.get('name')
    user = User.query.filter_by(email=email).first()
    if not user:
        user = User(email=email, name=name)
        db.session.add(user)
        db.session.commit()
    login_user(user)
    return redirect(url_for('home'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@scheduler.task('interval', id='run_probes', minutes=1)
def run_scheduled_probes():
    """Background task to run probes for all subscribed URLs."""
    with app.app_context():
        print("[SCHEDULER] Running scheduled probes...")
        _cleanup_old_results()
        subscriptions = Subscription.query.all()
        if not subscriptions:
            return
            
        urls_to_users = {}
        for sub in subscriptions:
            urls_to_users.setdefault(sub.url, []).append(sub.user_id)
            
        for url, user_ids in urls_to_users.items():
            print(f"[SCHEDULER] Processing URL: {url} for {len(user_ids)} users")
            try:
                # Process URL without caching yet (to avoid current_user dependency)
                result = _process_url(url, cache_result=False)
                if "error" not in result:
                    # Save cache for each subscribed user manually
                    for uid in user_ids:
                        _save_cache(
                            url=result["url"],
                            table_html=result["table"],
                            cluster_counts=result["cluster_counts"],
                            sentiment_counts=result["sentiment_counts"],
                            avg_subjectivity=result["avg_subjectivity"],
                            takeaways=result["takeaways"],
                            metrics=result["metrics"],
                            total_items=result["total_items"],
                            wordcloud_image=result["wordcloud_image"],
                            entities=result["entities"],
                            graph_data=result["graph_data"],
                            audio_file=result["audio_file"],
                            override_user_id=uid,
                            csv_string=result.get("csv_string")
                        )
            except Exception as e:
                print(f"[SCHEDULER] Error processing {url}: {e}")

def _save_cache(url, table_html, cluster_counts, sentiment_counts, avg_subjectivity, takeaways, metrics, total_items, wordcloud_image=None, entities=None, graph_data=None, audio_file=None, override_user_id=None, csv_string=None, **kwargs):
    """Save the last successful probe result to database for the current user or override user."""
    target_user_id = override_user_id if override_user_id else (current_user.id if current_user.is_authenticated else None)
    if target_user_id:
        try:
            cache = {
                "table": table_html,
                "cluster_counts": cluster_counts,
                "sentiment_counts": sentiment_counts,
                "takeaways": takeaways,
                "metrics": metrics,
                "wordcloud_image": wordcloud_image,
                "entities": entities,
                "graph_data": graph_data,
                "audio_file": audio_file,
                "processing_time_sec": kwargs.get("processing_time_sec") if kwargs else None,
                "human_time_mins": kwargs.get("human_time_mins") if kwargs else None,
            }
            probe = ProbeResult(
                user_id=target_user_id,
                url=url,
                total_items=total_items,
                avg_subjectivity=avg_subjectivity,
                results_json=json.dumps(cache),
                csv_data=csv_string
            )
            db.session.add(probe)
            db.session.commit()
        except Exception as e:
            print(f"DB save error: {e}")

def _load_cache():
    """Load the last probe result from database."""
    if current_user.is_authenticated:
        try:
            probe = ProbeResult.query.filter_by(user_id=current_user.id).order_by(ProbeResult.timestamp.desc()).first()
            if probe:
                cache = json.loads(probe.results_json)
                cache['url'] = probe.url
                cache['total_items'] = probe.total_items
                cache['avg_subjectivity'] = probe.avg_subjectivity
                return cache
        except Exception as e:
            print(f"DB load error: {e}")
    return None


def _cleanup_old_results():
    """Remove old word clouds and audio files to prevent disk clutter."""
    cleanup_old_wordclouds()
    cleanup_old_audio()


@app.route("/")
def home():
    return render_template("index.html")


def _process_url(url, cache_result=False):
    """Run the full NLP pipeline on a single URL and return context dict."""
    start_time = time.time()
    # Phase 1: Crawl
    df = scrape_url(url)
    if df.empty:
        return {"error": "No content could be extracted from this URL. The site may be blocking requests, or the page has no readable text content.", "url": url}

    # Phase 2: Analyse
    df = clean_text_data(df)
    if df.empty or len(df) < 2:
        return {"error": "After cleaning and deduplication, not enough unique text was found. Try a different URL with more content.", "url": url}

    # Phase 2.5: Sentiment Analysis
    df, sentiment_counts, avg_subjectivity = compute_sentiment(df)

    # Phase 2.8: Named Entity Recognition
    entities, graph_data = extract_entities(df)

    # Phase 3: Cluster and Summarize
    df, cluster_counts, takeaways, metrics, tfidf_data = perform_clustering(df)

    # Phase 4: Word Cloud
    wordcloud_image = None
    session_id = str(uuid.uuid4())[:8]
    if tfidf_data:
        wordcloud_image = generate_wordcloud(tfidf_data["matrix"], tfidf_data["terms"], session_id)

    # Phase 5: Audio Summary
    audio_file = generate_summary_audio(takeaways, session_id)

    # Save results in memory
    csv_string = df.to_csv(index=False)
    if has_request_context() and not current_user.is_authenticated:
        latest_anon_csvs[request.remote_addr] = csv_string

    # Build display table
    if "Sentiment" in df.columns:
        display_df = df[["text", "cluster", "Sentiment"]].rename(columns={"cluster": "Cluster"})
    else:
        display_df = df[["text", "cluster"]].rename(columns={"cluster": "Cluster"})
        
    table_html = display_df.to_html(classes="table table-hover", table_id="resultsTable", index=False)

    processing_time_sec = round(time.time() - start_time, 2)
    human_time_mins = round(len(df) * 5 / 60, 1)

    if cache_result:
        _save_cache(url, table_html, cluster_counts, sentiment_counts, avg_subjectivity, takeaways, metrics, len(df), wordcloud_image, entities, graph_data, audio_file, csv_string=csv_string, processing_time_sec=processing_time_sec, human_time_mins=human_time_mins)

    return {
        "table": table_html,
        "cluster_counts": cluster_counts,
        "sentiment_counts": sentiment_counts,
        "avg_subjectivity": avg_subjectivity,
        "takeaways": takeaways,
        "metrics": metrics,
        "url": url,
        "total_items": len(df),
        "wordcloud_image": wordcloud_image,
        "entities": entities,
        "graph_data": graph_data,
        "audio_file": audio_file,
        "csv_string": csv_string,
        "cleaned_text": df["cleaned"].tolist() if "cleaned" in df.columns else [],
        "processing_time_sec": processing_time_sec,
        "human_time_mins": human_time_mins
    }


@app.route("/process", methods=["POST"])
def process_url():
    url = request.form.get("url")
    if not url:
        return redirect(url_for("home"))

    _cleanup_old_results()
    result = _process_url(url, cache_result=True)
    if "error" in result:
        return render_template("index.html", error=result["error"])

    _add_time_series(result, url)
    
    # Save context for RAG Groq chat
    user_key = current_user.id if current_user.is_authenticated else request.remote_addr
    texts = result.get("cleaned_text", [])
    faiss_index = build_faiss_index(texts)
    chat_contexts[user_key] = {
        "texts": texts,
        "index": faiss_index
    }
    
    return render_template("index.html", cached=False, **result)

def _add_time_series(result, url):
    """Helper to add subscription and time series data to a probe result."""
    result["time_series_data"] = []
    result["is_subscribed"] = False
    if current_user and current_user.is_authenticated:
        sub = Subscription.query.filter_by(user_id=current_user.id, url=url).first()
        result["is_subscribed"] = sub is not None
        history = ProbeResult.query.filter_by(user_id=current_user.id, url=url).order_by(ProbeResult.timestamp.asc()).all()
        for h in history:
            result["time_series_data"].append({
                "timestamp": h.timestamp.strftime('%m-%d %H:%M'),
                "subjectivity": h.avg_subjectivity
            })

@app.route("/subscribe", methods=["POST"])
@login_required
def subscribe():
    url = request.form.get("url")
    if not url:
        return redirect(url_for("home"))
        
    sub = Subscription.query.filter_by(user_id=current_user.id, url=url).first()
    if sub:
        db.session.delete(sub)
        db.session.commit()
    else:
        new_sub = Subscription(user_id=current_user.id, url=url)
        db.session.add(new_sub)
        db.session.commit()
        
    return redirect(request.referrer or url_for("home"))

@app.route("/api/chat", methods=["POST"])
def chat_api():
    if not groq_client:
        return jsonify({"error": "Groq API key not configured."}), 500
        
    data = request.json
    question = data.get("question", "").strip()
    
    if not question:
        return jsonify({"error": "Empty question."}), 400
        
    user_key = current_user.id if current_user.is_authenticated else request.remote_addr
    context_data = chat_contexts.get(user_key, {})
    
    texts = context_data.get("texts", [])
    index = context_data.get("index", None)
    
    if not texts or not index:
        return jsonify({"error": "No website context found. Please probe a URL first."}), 400
        
    # RAG Retrieval: Extract exactly the top 7 most semantically relevant chunks for the question
    context_text = retrieve_context(question, index, texts, top_k=7)
    
    fact_check = data.get("fact_check", False)
    
    if fact_check:
        prompt = f"You are a rigorous Fact-Checking and Media Analysis AI. Carefully evaluate the following user question against the provided webpage context. CRITICAL: Do NOT blindly trust the context. Use your extensive real-world knowledge to independently verify the claims made in the context. Highlight any factual inaccuracies, missing nuance, misleading framing, or strong biases present in the article while answering the user's question.\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
    else:
        prompt = f"You are GistProbe AI, a friendly and intelligent assistant. If the user greets you casually, respond naturally and offer to help analyze the webpage. For informational questions, use the provided webpage context as your foundation, but feel free to seamlessly weave in relevant real-world knowledge and background details to provide a richer, more comprehensive answer.\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=500
        )
        answer = completion.choices[0].message.content
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Groq API Error: {e}")
        return jsonify({"error": "An error occurred while generating the answer."}), 500


@app.route("/compare", methods=["GET"])
def compare_page():
    return render_template("compare.html")


@app.route("/compare", methods=["POST"])
def compare():
    try:
        url1 = request.form.get("url1", "").strip()
        url2 = request.form.get("url2", "").strip()

        if not url1 or not url2:
            return render_template("compare.html", error="Please enter two valid URLs to compare.")

        _cleanup_old_results()
        
        result1 = _process_url(url1)
        result2 = _process_url(url2)
        
        if "error" in result1:
            return render_template("compare.html", error=f"Error in URL 1 ({url1}): {result1['error']}")
        if "error" in result2:
            return render_template("compare.html", error=f"Error in URL 2 ({url2}): {result2['error']}")
        similarity_score = compute_similarity(result1.get("cleaned_text", []), result2.get("cleaned_text", []))
        
        # Calculate Shared Vocabulary (Entities)
        shared_vocab = []
        if result1.get("entities") and result2.get("entities"):
            ents1 = set(item[0].lower() for items in result1["entities"].values() for item in items)
            ents2 = set(item[0].lower() for items in result2["entities"].values() for item in items)
            shared = ents1.intersection(ents2)
            
            # Keep original casing from result1
            for items in result1["entities"].values():
                for item in items:
                    if item[0].lower() in shared and item[0] not in shared_vocab:
                        shared_vocab.append(item[0])

        # Generate AI Comparative Summary (Debate Mode)
        debate_summary = "AI Summary generation failed or Groq API key is missing."
        if groq_client:
            # Truncate text to fit in prompt limits (~12k chars each)
            text1 = " ".join(result1.get("cleaned_text", []))[:12000]
            text2 = " ".join(result2.get("cleaned_text", []))[:12000]
            
            prompt = f"""You are an expert media analyst. Compare how the following two webpages framed the same event or topic.

Analyze differences in tone, focus, biases, and what facts each omitted. Provide your analysis strictly in HTML format using exactly this structure:

<div style="margin-bottom: 20px;">
    <h5 style="color: var(--accent-primary); margin-bottom: 10px; font-weight: 700;">Tone & Framing</h5>
    <p style="margin-left: 15px; line-height: 1.6;">[Your analysis here]</p>
</div>
<div style="margin-bottom: 20px;">
    <h5 style="color: var(--accent-primary); margin-bottom: 10px; font-weight: 700;">Biases & Omissions</h5>
    <p style="margin-left: 15px; line-height: 1.6;">[Your analysis here]</p>
</div>
<div style="margin-bottom: 20px;">
    <h5 style="color: var(--accent-primary); margin-bottom: 10px; font-weight: 700;">Key Divergence</h5>
    <p style="margin-left: 15px; line-height: 1.6;">[Your analysis here]</p>
</div>

Use <b> tags to highlight key entities or stark differences. Do NOT use markdown code blocks (like ```html). Output ONLY the raw HTML.

Webpage 1 ({url1}):
{text1}

Webpage 2 ({url2}):
{text2}

Executive Comparison Summary (HTML only):"""

            try:
                completion = groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.6,
                    max_tokens=800
                )
                debate_summary = completion.choices[0].message.content
            except Exception as e:
                print(f"Groq API Error in Debate Mode: {e}")
                debate_summary = f"Error generating comparison: {str(e)}"
            
        return render_template("compare.html", result1=result1, result2=result2, similarity_score=similarity_score, shared_vocab=shared_vocab, ai_summary=debate_summary)

    except Exception as e:
        print(f"Compare Pipeline error: {e}")
        return render_template(
            "compare.html",
            error=f"Something went wrong: {str(e)}. Please check the URLs and try again."
        )


@app.route("/download")
def download():
    """Serves the most recent results CSV from memory or database."""
    if current_user.is_authenticated:
        latest_probe = ProbeResult.query.filter_by(user_id=current_user.id).order_by(ProbeResult.timestamp.desc()).first()
        if latest_probe and latest_probe.csv_data:
            return Response(
                latest_probe.csv_data,
                mimetype="text/csv",
                headers={"Content-disposition": "attachment; filename=gistprobe_results.csv"}
            )
    else:
        ip = request.remote_addr
        if ip in latest_anon_csvs:
            return Response(
                latest_anon_csvs[ip],
                mimetype="text/csv",
                headers={"Content-disposition": "attachment; filename=gistprobe_results.csv"}
            )
    return redirect(url_for("home"))


@app.route("/api/v1/analyze", methods=["POST"])
def api_analyze():
    """REST API endpoint for Power Automate and Enterprise integrations."""
    data = request.json or {}
    url = data.get("url")
    if not url:
        return jsonify({"error": "URL parameter is required"}), 400
        
    _cleanup_old_results()
    result = _process_url(url, cache_result=False)
    
    if "error" in result:
        return jsonify(result), 400
        
    # Return a JSON payload suitable for automated workflows
    return jsonify({
        "url": result["url"],
        "total_items": result["total_items"],
        "processing_time_sec": result.get("processing_time_sec"),
        "human_time_mins": result.get("human_time_mins"),
        "takeaways": result["takeaways"],
        "cluster_counts": result["cluster_counts"],
        "metrics": result["metrics"],
        "avg_subjectivity": result["avg_subjectivity"]
    })


@app.route("/export/excel")
def export_excel():
    """Serves the most recent results as an Excel BA Report."""
    latest_csv_string = None
    if current_user.is_authenticated:
        latest_probe = ProbeResult.query.filter_by(user_id=current_user.id).order_by(ProbeResult.timestamp.desc()).first()
        if latest_probe and latest_probe.csv_data:
            latest_csv_string = latest_probe.csv_data
    else:
        ip = request.remote_addr
        if ip in latest_anon_csvs:
            latest_csv_string = latest_anon_csvs[ip]
            
    if not latest_csv_string:
        return redirect(url_for("home"))
        
    df = pd.read_csv(io.StringIO(latest_csv_string))
    
    # Try to load takeaways from cache if authenticated
    takeaways = []
    if current_user.is_authenticated:
        latest_probe = ProbeResult.query.filter_by(user_id=current_user.id).order_by(ProbeResult.timestamp.desc()).first()
        if latest_probe and latest_probe.results_json:
            cache = json.loads(latest_probe.results_json)
            takeaways = cache.get("takeaways", [])

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if takeaways:
            summary_df = pd.DataFrame({"Executive Summary (AI Generated)": takeaways})
            summary_df.to_excel(writer, sheet_name="AI Summary", index=False)
        df.to_excel(writer, sheet_name="Clustered Data", index=False)
        
    output.seek(0)
    return send_file(
        output,
        download_name="GistProbe_BA_Report.xlsx",
        as_attachment=True,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.route("/demo")
def demo():
    """Load the last successful probe result from cache — instant demo mode."""
    cache = _load_cache()
    if not cache:
        return redirect(url_for("home"))

    result = {
        "table": cache["table"],
        "cluster_counts": cache["cluster_counts"],
        "sentiment_counts": cache["sentiment_counts"],
        "avg_subjectivity": cache["avg_subjectivity"],
        "takeaways": cache["takeaways"],
        "metrics": cache["metrics"],
        "url": cache["url"],
        "total_items": cache["total_items"],
        "wordcloud_image": cache.get("wordcloud_image"),
        "entities": cache.get("entities"),
        "graph_data": cache.get("graph_data"),
        "audio_file": cache.get("audio_file"),
        "processing_time_sec": cache.get("processing_time_sec"),
        "human_time_mins": cache.get("human_time_mins"),
    }
    _add_time_series(result, cache["url"])
    return render_template("index.html", cached=True, **result)



@app.route('/history')
@login_required
def history():
    probes = ProbeResult.query.filter_by(user_id=current_user.id).order_by(ProbeResult.timestamp.desc()).all()
    return render_template('history.html', probes=probes)

@app.route('/history/<int:probe_id>')
@login_required
def load_history(probe_id):
    probe = ProbeResult.query.get_or_404(probe_id)
    if probe.user_id != current_user.id:
        return redirect(url_for('home'))
    cache = json.loads(probe.results_json)
    cache['url'] = probe.url
    cache['total_items'] = probe.total_items
    cache['avg_subjectivity'] = probe.avg_subjectivity
    # Check if files still exist on disk to prevent broken images/audio
    wc_img = cache.get("wordcloud_image")
    if wc_img and not os.path.exists(os.path.join("static", wc_img)):
        wc_img = None
        
    audio = cache.get("audio_file")
    if audio and not os.path.exists(os.path.join("static", audio)):
        audio = None

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
        wordcloud_image=wc_img,
        entities=cache.get("entities"),
        graph_data=cache.get("graph_data"),
        audio_file=audio,
        processing_time_sec=cache.get("processing_time_sec"),
        human_time_mins=cache.get("human_time_mins"),
    )

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)