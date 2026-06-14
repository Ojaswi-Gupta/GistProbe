import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the model lazily so it doesn't block app startup
_model = None

def get_embedding_model():
    global _model
    if _model is None:
        print("[RAG] Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def build_faiss_index(texts):
    """
    Given a list of text strings, compute their embeddings and build a FAISS index.
    Returns the built index.
    """
    if not texts:
        return None
        
    model = get_embedding_model()
    print(f"[RAG] Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    # Initialize FAISS Index (L2 distance)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Add vectors to index
    index.add(embeddings)
    print(f"[RAG] Built FAISS index with {index.ntotal} vectors.")
    return index

def retrieve_context(question, index, texts, top_k=5):
    """
    Given a user question, search the FAISS index for the top_k most relevant text chunks.
    """
    if not index or not texts:
        return ""
        
    model = get_embedding_model()
    q_embedding = model.encode([question], convert_to_numpy=True)
    
    # Search FAISS
    distances, indices = index.search(q_embedding, min(top_k, len(texts)))
    
    retrieved_chunks = []
    for idx in indices[0]:
        if idx != -1 and idx < len(texts):
            retrieved_chunks.append(texts[idx])
            
    print(f"[RAG] Retrieved {len(retrieved_chunks)} relevant chunks for question: '{question}'")
    return "\n\n".join(retrieved_chunks)
