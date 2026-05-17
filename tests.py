"""
GistProbe Unit Tests
Tests for the core NLP pipeline: analyser, clustering, and crawler modules.

Run with:  python -m pytest tests.py -v
Or:        python tests.py
"""
import unittest
import pandas as pd
from analyser import clean_text_data, compute_sentiment
from clustering import perform_clustering


class TestAnalyser(unittest.TestCase):
    """Tests for the text cleaning and deduplication pipeline."""

    def test_clean_removes_exact_duplicates(self):
        """Exact duplicate rows should be removed."""
        df = pd.DataFrame({"text": [
            "This is a test sentence about machine learning",
            "This is a test sentence about machine learning",
            "Another unique sentence about natural language processing",
        ]})
        result = clean_text_data(df)
        self.assertEqual(len(result), 2)

    def test_clean_removes_near_duplicates(self):
        """Texts with >85% similarity should be treated as near-duplicates."""
        df = pd.DataFrame({"text": [
            "The quick brown fox jumps over the lazy dog in the park",
            "The quick brown fox jumps over the lazy dog in the garden",  # ~85% similar
            "Completely different sentence about artificial intelligence and deep learning",
        ]})
        result = clean_text_data(df)
        # The two fox sentences are near-duplicates; one should be removed
        self.assertLessEqual(len(result), 2)

    def test_clean_removes_short_text(self):
        """Very short text (<=5 chars after cleaning) should be filtered out."""
        df = pd.DataFrame({"text": [
            "Hi",
            "OK",
            "This is a perfectly valid long sentence about technology and innovation",
        ]})
        result = clean_text_data(df)
        self.assertEqual(len(result), 1)

    def test_clean_handles_empty_dataframe(self):
        """Empty DataFrames should be returned as-is without errors."""
        df = pd.DataFrame(columns=["text"])
        result = clean_text_data(df)
        self.assertTrue(result.empty)


class TestSentiment(unittest.TestCase):
    """Tests for sentiment analysis scoring."""

    def test_sentiment_labels_are_valid(self):
        """Sentiment labels should be one of Positive, Negative, or Neutral."""
        df = pd.DataFrame({"text": [
            "This is absolutely wonderful and amazing",
            "This is terrible and awful and disgusting",
            "The meeting is at 3pm in the conference room",
        ]})
        df["cleaned"] = df["text"].str.lower()
        result_df, counts, avg_sub = compute_sentiment(df)
        valid_labels = {"Positive", "Negative", "Neutral"}
        for label in result_df["sentiment_label"]:
            self.assertIn(label, valid_labels)

    def test_sentiment_counts_sum_to_total(self):
        """Sentiment counts should sum to the total number of rows."""
        df = pd.DataFrame({"text": [
            "I love this product so much",
            "I hate this terrible experience",
            "The table is made of wood",
        ]})
        df["cleaned"] = df["text"].str.lower()
        _, counts, _ = compute_sentiment(df)
        self.assertEqual(sum(counts.values()), 3)

    def test_sentiment_handles_empty(self):
        """Empty DataFrames should return empty counts."""
        df = pd.DataFrame(columns=["text"])
        result_df, counts = compute_sentiment(df)
        self.assertTrue(result_df.empty)


class TestClustering(unittest.TestCase):
    """Tests for TF-IDF + K-Means clustering pipeline."""

    def test_clustering_assigns_cluster_labels(self):
        """Every row should receive a cluster assignment."""
        df = pd.DataFrame({
            "text": [
                "Machine learning is transforming healthcare",
                "Deep learning and neural networks",
                "Artificial intelligence in medical diagnosis",
                "Soccer world cup finals results",
                "Football match scores and highlights",
                "Premier league transfer news today",
            ],
            "cleaned": [
                "machine learning is transforming healthcare",
                "deep learning and neural networks",
                "artificial intelligence in medical diagnosis",
                "soccer world cup finals results",
                "football match scores and highlights",
                "premier league transfer news today",
            ],
        })
        result_df, counts, takeaways, metrics, _ = perform_clustering(df)
        self.assertIn("cluster", result_df.columns)
        self.assertTrue(all(result_df["cluster"].notna()))

    def test_clustering_returns_valid_metrics(self):
        """Metrics should contain silhouette_score, optimal_k, and vocab_size."""
        df = pd.DataFrame({
            "text": [
                "Python programming language tutorial",
                "Java development best practices",
                "Web scraping with beautiful soup",
                "Data science and machine learning",
                "Natural language processing techniques",
            ],
            "cleaned": [
                "python programming language tutorial",
                "java development best practices",
                "web scraping with beautiful soup",
                "data science and machine learning",
                "natural language processing techniques",
            ],
        })
        _, _, _, metrics, _ = perform_clustering(df)
        self.assertIn("silhouette_score", metrics)
        self.assertIn("optimal_k", metrics)
        self.assertIn("vocab_size", metrics)
        self.assertGreaterEqual(metrics["optimal_k"], 2)

    def test_clustering_handles_tiny_dataset(self):
        """Datasets with fewer than 3 items should default to a single cluster."""
        df = pd.DataFrame({
            "text": ["One sentence only", "Two sentences here"],
            "cleaned": ["one sentence only", "two sentences here"],
        })
        result_df, counts, takeaways, metrics, _ = perform_clustering(df)
        self.assertEqual(metrics["optimal_k"], 1)

    def test_takeaways_are_unique(self):
        """Extractive summary takeaways should not be near-duplicates of each other."""
        df = pd.DataFrame({
            "text": [
                "The global economy is experiencing significant growth in the technology sector this quarter",
                "Climate change continues to affect agricultural productivity across developing nations worldwide",
                "New artificial intelligence breakthroughs are reshaping how businesses operate and compete globally",
                "International trade agreements are being renegotiated to address modern digital commerce challenges",
                "Healthcare systems around the world are adopting telemedicine solutions at an unprecedented rate",
                "Education reform initiatives are focusing on STEM programs and computational thinking skills",
            ],
            "cleaned": [
                "global economy experiencing significant growth technology sector quarter",
                "climate change continues affect agricultural productivity developing nations worldwide",
                "new artificial intelligence breakthroughs reshaping businesses operate compete globally",
                "international trade agreements renegotiated address modern digital commerce challenges",
                "healthcare systems around world adopting telemedicine solutions unprecedented rate",
                "education reform initiatives focusing stem programs computational thinking skills",
            ],
        })
        _, _, takeaways, _, _ = perform_clustering(df)
        # Each takeaway should be unique
        self.assertEqual(len(takeaways), len(set(takeaways)))


if __name__ == "__main__":
    unittest.main()
