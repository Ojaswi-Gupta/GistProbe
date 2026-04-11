import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def perform_clustering(df):
    """
    Phase 3: Intelligent Clustering Engine
    Uses Silhouette Score to find the optimal k and N-grams for better naming.
    """
    print(f"\n--- Phase 3: Intelligent Clustering (GistProbe) ---")
    
    if df.empty or len(df) < 3:
        print("Data volume too low for mathematical optimization. Defaulting to Cluster 0.")
        df["cluster"] = 0
        df["cluster_name"] = "0: General Topics"
        return df, {"0: General Topics": len(df)}

    texts = df["cleaned"]

    # NLP Improvement: ngram_range=(1, 2) captures phrases like "San Francisco" or "Artificial Intelligence"
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    # --- MATHEMATICAL SEARCH FOR OPTIMAL K ---
    best_k = 2
    best_score = -1
    
    # We test a range from 2 to 6 clusters
    limit = min(7, len(df)) 
    print("Evaluating mathematical optimum for k...")

    for k in range(2, limit):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        
        # Calculate Silhouette Score (Closer to 1 is better)
        score = silhouette_score(X, labels)
        print(f"Testing k={k} | Silhouette Score: {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k

    print(f"Mathematical Optimum Found: k={best_k}")

    # --- FINAL CLUSTERING ---
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    model.fit(X)

    df["cluster"] = model.labels_
    terms = vectorizer.get_feature_names_out()
    summary_mapping = {}

    print("\nCluster Keywords:")
    for i in range(best_k):
        # Get the top features for this cluster
        center = model.cluster_centers_[i]
        top_indices = center.argsort()[-2:] # Focus on top 2 most relevant terms
        top_words = [terms[t] for t in top_indices]
        
        # Logic to clean up the name: 
        # If the top word is already a phrase (like "san francisco"), use it.
        # Otherwise, join the top two with an ampersand.
        main_topic = top_words[-1] # The most important word/phrase
        secondary_topic = top_words[-2]
        
        if len(main_topic.split()) > 1:
            # If the top term is already a bigram (e.g., "san francisco")
            topic_name = main_topic.capitalize()
        else:
            # Join two unigrams (e.g., "Robot & Youtube")
            topic_name = f"{secondary_topic} & {main_topic}".capitalize()
            
        display_name = f"{i}: {topic_name}"
        summary_mapping[i] = display_name
        
        # Terminal Log for Keywords
        raw_keywords = [terms[t] for t in center.argsort()[-5:]]
        print(f"Cluster {i} ({display_name}): {raw_keywords}")

    # Map the display names back to the dataframe
    df["cluster_summary_name"] = df["cluster"].map(summary_mapping)
    
    # Sort by cluster number (0, 1, 2...) for the UI summary
    counts = df["cluster_summary_name"].value_counts().sort_index().to_dict()
    
    print("\nCluster Distribution:")
    print(df["cluster_summary_name"].value_counts())
    
    print("\nPreview:")
    print(df[["text", "cluster"]].head())
    
    print("\nClustering done successfully")
    
    return df, counts