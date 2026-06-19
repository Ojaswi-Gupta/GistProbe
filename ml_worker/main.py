from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import pandas as pd

from crawler import scrape_url
from analyser import clean_text_data, compute_sentiment
from clustering import perform_clustering
from ner import extract_entities
from rag import build_faiss_index, retrieve_context

app = FastAPI(title="GistProbe ML Worker")

class ProbeRequest(BaseModel):
    url: str

class RetrieveRequest(BaseModel):
    question: str
    texts: List[str]
    top_k: int = 7

class SimilarityRequest(BaseModel):
    texts1: List[str]
    texts2: List[str]

@app.post("/probe")
async def probe_url(req: ProbeRequest):
    url = req.url
    start_time = time.time()
    
    # 1. Crawl
    df = scrape_url(url)
    if df.empty:
        raise HTTPException(status_code=400, detail="No content could be extracted from this URL.")
        
    # 2. Analyse
    df = clean_text_data(df)
    if df.empty or len(df) < 2:
        raise HTTPException(status_code=400, detail="Not enough unique text found after cleaning.")
        
    # 3. Sentiment
    df, sentiment_counts, avg_subjectivity = compute_sentiment(df)
    
    # 4. NER
    entities, graph_data = extract_entities(df)
    
    # 5. Clustering
    df, cluster_counts, takeaways, metrics, word_scores = perform_clustering(df)
    
    processing_time_sec = round(time.time() - start_time, 2)
    human_time_mins = round(len(df) * 5 / 60, 1)
    
    # Convert DataFrame to list of dicts for JSON serialization
    # Include 'cleaned' text for RAG
    if "Sentiment" in df.columns:
        display_data = df[["text", "cluster", "Sentiment", "cleaned"]].to_dict(orient="records")
    else:
        display_data = df[["text", "cluster", "cleaned"]].to_dict(orient="records")
        
    return {
        "url": url,
        "total_items": len(df),
        "cluster_counts": cluster_counts,
        "sentiment_counts": sentiment_counts,
        "avg_subjectivity": avg_subjectivity,
        "takeaways": takeaways,
        "metrics": metrics,
        "word_scores": word_scores,
        "entities": entities,
        "graph_data": graph_data,
        "data": display_data,
        "processing_time_sec": processing_time_sec,
        "human_time_mins": human_time_mins
    }

@app.post("/retrieve")
async def retrieve(req: RetrieveRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="No texts provided for RAG.")
    
    index = build_faiss_index(req.texts)
    context_text = retrieve_context(req.question, index, req.texts, top_k=req.top_k)
    return {"context_text": context_text}

@app.post("/similarity")
async def get_similarity(req: SimilarityRequest):
    from clustering import compute_similarity
    score = compute_similarity(req.texts1, req.texts2)
    return {"score": score}

@app.get("/health")
async def health_check():
    return {"status": "ok"}
