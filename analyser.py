import re
import pandas as pd
from difflib import SequenceMatcher
from textblob import TextBlob


def _similarity(a, b):
    """Calculate string similarity ratio between two texts (0.0 to 1.0)."""
    return SequenceMatcher(None, a, b).ratio()


def clean_text_data(df):
    """
    Phase 2: Analysis & Deduplication.
    Standardizes text for the model while preserving meaning.
    
    Improvements over original:
    - Keeps numbers (e.g., ₹30, 2024, 100%) for semantic value
    - Supports Unicode/multilingual characters
    - Removes near-duplicates using similarity matching, not just exact matches
    """
    if df.empty:
        return df

    print(f"\n--- Phase 2: Text Analysis & Deduplication ---")
    original_count = len(df)

    # Stage 1: Remove exact raw duplicates
    df = df.drop_duplicates(subset=["text"]).copy()
    print(f"After exact dedup: {len(df)} rows (removed {original_count - len(df)})")

    def clean_logic(text):
        text = str(text)
        # Keep letters, numbers, and spaces (supports basic multilingual text)
        text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
        # Collapse multiple spaces into one
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()

    df["cleaned"] = df["text"].apply(clean_logic)

    # Stage 2: Remove empty rows
    df = df[df["cleaned"].str.len() > 5].copy()

    # Stage 3: Remove near-duplicates using similarity threshold
    # Two texts that are >85% similar after cleaning are considered duplicates
    to_drop = set()
    cleaned_list = df["cleaned"].tolist()
    indices = df.index.tolist()

    for i in range(len(cleaned_list)):
        if indices[i] in to_drop:
            continue
        for j in range(i + 1, len(cleaned_list)):
            if indices[j] in to_drop:
                continue
            if _similarity(cleaned_list[i], cleaned_list[j]) > 0.85:
                to_drop.add(indices[j])

    df = df.drop(index=to_drop).copy()
    print(f"After near-duplicate removal: {len(df)} rows (removed {len(to_drop)} similar items)")

    # Terminal Logs
    print(f"\nCleaned text preview:")
    print(df[["text", "cleaned"]].head())
    print("✓ Text cleaning complete")

    return df


def compute_sentiment(df):
    """
    Calculate the sentiment of each text row using TextBlob.
    Assigns a label and badge icon for the UI.
    """
    if df.empty:
        return df, {}

    print(f"\n--- Phase 2.5: Sentiment Analysis ---")
    
    def get_sentiment_and_subjectivity(text):
        analysis = TextBlob(str(text))
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        if polarity > 0.1:
            label = "Positive"
        elif polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"
            
        return pd.Series([label, subjectivity])

    def get_badge(sentiment):
        if sentiment == "Positive":
            return "🟩 Positive"
        elif sentiment == "Negative":
            return "🟥 Negative"
        return "⬜ Neutral"

    # Extract sentiment labels and subjectivity scores
    df[["sentiment_label", "subjectivity"]] = df["text"].apply(get_sentiment_and_subjectivity)
    df["Sentiment"] = df["sentiment_label"].apply(get_badge)

    # Calculate Objectivity Index (Average Subjectivity)
    avg_subjectivity = df["subjectivity"].mean()

    counts = df["sentiment_label"].value_counts().to_dict()
    print(f"Sentiment Distribution: {counts}")
    print(f"Average Subjectivity: {avg_subjectivity:.2f}")
    
    # Cleanup intermediate columns
    df = df.drop(columns=["subjectivity"])
    
    return df, counts, avg_subjectivity