import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from difflib import SequenceMatcher

def _similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def _generate_cluster_name(cluster_id, center, terms):
    """
    Generate a meaningful cluster name from top TF-IDF terms.
    Prefers bigrams/trigrams over single words for more descriptive names.
    """
    top_indices = center.argsort()[-5:]  # Get top 5 terms
    top_words = [terms[t] for t in top_indices]

    # Separate bigrams and unigrams
    bigrams = [w for w in top_words if len(w.split()) > 1]
    unigrams = [w for w in top_words if len(w.split()) == 1]

    if bigrams:
        # Use the best bigram as the cluster name
        topic_name = bigrams[-1].title()
    elif len(unigrams) >= 2:
        # Join top 2 unigrams
        topic_name = f"{unigrams[-1].title()} & {unigrams[-2].title()}"
    else:
        topic_name = unigrams[-1].title() if unigrams else "General"

    return f"{cluster_id}: {topic_name}"


def perform_clustering(df):
    """
    Phase 3: Intelligent Clustering Engine.
    Uses Silhouette Score to find the optimal k.
    
    Improvements over original:
    - Wider k-range (2 to min(10, N)) for better coverage on large datasets
    - Better cluster naming using bigrams/trigrams
    - Graceful error handling for edge cases
    """
    print(f"\n--- Phase 3: Intelligent Clustering (GistProbe) ---")

    if df.empty or len(df) < 3:
        print("Data volume too low for clustering. Defaulting to Cluster 0.")
        df["cluster_name"] = "0: General Topics"
        metrics = {"silhouette_score": "N/A", "optimal_k": 1, "vocab_size": 0}
        return df, {"0: General Topics": len(df)}, [], metrics

    texts = df["cleaned"]

    # NLP: ngram_range=(1, 2) captures phrases like "climate change" or "supreme court"
    # max_features limits noise from rare terms
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        min_df=1,
        max_df=0.95,
    )

    try:
        X = vectorizer.fit_transform(texts)
        vocab_size = X.shape[1]
    except ValueError as e:
        print(f"Vectorization error: {e}")
        df["cluster"] = 0
        df["cluster_name"] = "0: General Topics"
        metrics = {"silhouette_score": "N/A", "optimal_k": 1, "vocab_size": 0}
        return df, {"0: General Topics": len(df)}, [], metrics

    # --- MATHEMATICAL SEARCH FOR OPTIMAL K ---
    best_k = 2
    best_score = -1

    # Extended range: 2 to min(10, N-1) for better granularity on large datasets
    max_k = min(10, len(df))
    print(f"Evaluating optimal k in range [2, {max_k - 1}]...")

    for k in range(2, max_k):
        try:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)

            # Silhouette Score: closer to 1 = better separation
            score = silhouette_score(X, labels)
            print(f"  k={k} → Silhouette Score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k
        except Exception as e:
            print(f"  k={k} → Error: {e}")

    print(f"✓ Optimal k = {best_k} (score: {best_score:.4f})")

    # --- FINAL CLUSTERING ---
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    model.fit(X)

    df["cluster"] = model.labels_
    terms = vectorizer.get_feature_names_out()
    summary_mapping = {}

    print("\nCluster Details:")
    for i in range(best_k):
        center = model.cluster_centers_[i]
        display_name = _generate_cluster_name(i, center, terms)
        summary_mapping[i] = display_name

        # Terminal log: show top 5 keywords for debugging
        raw_keywords = [terms[t] for t in center.argsort()[-5:]]
        count = (model.labels_ == i).sum()
        print(f"  {display_name} ({count} items) → keywords: {raw_keywords}")

    # Map display names to dataframe
    df["cluster_summary_name"] = df["cluster"].map(summary_mapping)

    # Sort by cluster number for consistent UI display
    counts = df["cluster_summary_name"].value_counts().sort_index().to_dict()

    print(f"\n✓ Clustering complete: {best_k} clusters, {len(df)} items")

    # --- EXTRACTIVE SUMMARIZATION (TF-IDF) ---
    print("Extracting Key Takeaways using TF-IDF...")
    takeaways = []
    try:
        doc_scores = X.sum(axis=1).A1  # Convert matrix to 1D array
        df["tfidf_score"] = doc_scores
        
        # Filter meaningful sentences (>50 chars) and sort by TF-IDF density
        valid_sentences = df[df["text"].str.len() > 50].sort_values(by="tfidf_score", ascending=False)
        
        for text in valid_sentences["text"]:
            if len(takeaways) >= 3:
                break
            # Avoid near-duplicates in takeaways
            is_duplicate = any(_similarity(text, t) > 0.6 for t in takeaways) if takeaways else False
            if not is_duplicate:
                takeaways.append(text)
                
    except Exception as e:
        print(f"Summarization error: {e}")

    metrics = {
        "silhouette_score": round(best_score, 3) if best_score > -1 else "N/A",
        "optimal_k": best_k,
        "vocab_size": vocab_size
    }

    return df, counts, takeaways, metrics