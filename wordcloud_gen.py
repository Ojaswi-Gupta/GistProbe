"""
GistProbe Word Cloud Generator.
Creates a word cloud image from TF-IDF term scores.
"""
import os
import glob
from wordcloud import WordCloud


# Color function matching GistProbe's accent palette
COLORS = ["#0ea5e9", "#6366f1", "#8b5cf6", "#38bdf8", "#10b981", "#f59e0b", "#ec4899"]


def _gistprobe_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    """Custom color function using GistProbe's accent palette."""
    import random
    return random.choice(COLORS)


def cleanup_old_wordclouds():
    """Remove old word cloud images (older than 1 hour) to prevent disk clutter."""
    import time
    now = time.time()
    for f in glob.glob("static/wordcloud_*.png"):
        try:
            if os.stat(f).st_mtime < now - 3600:
                os.remove(f)
        except OSError:
            pass


def generate_wordcloud(word_scores, session_id):
    """
    Generate a word cloud PNG from word frequency scores.
    
    Args:
        word_scores: Dictionary of word frequencies/scores
        session_id: Unique session ID for the filename
        
    Returns:
        str: Path to the generated word cloud image (relative to static/), or None on failure.
    """
    try:
        if not word_scores:
            print("No word scores for word cloud generation.")
            return None

        # Generate the word cloud
        wc = WordCloud(
            width=800,
            height=400,
            background_color=None,
            mode="RGBA",
            max_words=80,
            color_func=_gistprobe_color_func,
            prefer_horizontal=0.85,
            min_font_size=10,
            max_font_size=80,
            relative_scaling=0.5,
            margin=10,
        )

        wc.generate_from_frequencies(word_scores)

        # Save to static directory
        os.makedirs("static", exist_ok=True)
        filename = f"wordcloud_{session_id}.png"
        filepath = os.path.join("static", filename)
        wc.to_file(filepath)

        print(f"✓ Word cloud saved: {filepath} ({len(word_scores)} terms)")
        return filename

    except Exception as e:
        print(f"Word cloud generation error: {e}")
        return None
