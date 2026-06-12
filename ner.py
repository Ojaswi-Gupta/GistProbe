"""
Named Entity Recognition (NER) module for GistProbe.
Uses spaCy to extract and categorize entities (People, Organizations, Locations, etc.)
"""
import spacy
from collections import Counter

# Load the small English model
# Will fallback gracefully if model isn't installed
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: spaCy model 'en_core_web_sm' not found. NER will be disabled.")
    nlp = None

# Mapping spaCy labels to user-friendly categories
ENTITY_MAPPING = {
    "PERSON": "People",
    "ORG": "Organizations",
    "GPE": "Locations",
    "LOC": "Locations",
    "DATE": "Dates",
    "MONEY": "Money"
}

def extract_entities(df):
    """
    Extract Named Entities from a DataFrame of text.
    
    Args:
        df: DataFrame containing a 'text' column with raw text.
        
    Returns:
        dict: Categorized entities with counts, e.g.,
              {"Organizations": [("Google", 5), ("Apple", 3)], "People": [...]}
    """
    if nlp is None or df.empty or "text" not in df.columns:
        return {}, {"nodes": [], "edges": []}
        
    print("\n--- Phase 2.8: Named Entity Recognition ---")
    
    # Process all text together
    # We use raw text (not cleaned) because capitalization helps NER
    full_text = " ".join(df["text"].dropna().astype(str).tolist())
    
    # Process with spaCy (increase max_length if text is very long, but usually standard is fine)
    # We truncate to 1,000,000 chars just to be safe
    doc = nlp(full_text[:1000000])
    
    entities_by_type = {v: [] for v in set(ENTITY_MAPPING.values())}
    
    nodes_dict = {}
    edges_dict = {}

    # Extract and categorize, plus build graph
    for sent in doc.sents:
        sent_entities = []
        for ent in sent.ents:
            category = ENTITY_MAPPING.get(ent.label_)
            if category:
                ent_text = ent.text.strip().replace('\n', ' ')
                if len(ent_text) > 1:
                    entities_by_type[category].append(ent_text)
                    sent_entities.append((ent_text, category))
                    
                    if ent_text not in nodes_dict:
                        nodes_dict[ent_text] = {
                            "id": ent_text, 
                            "label": ent_text, 
                            "group": category, 
                            "value": 1,
                            "title": f"<b>{ent_text}</b><br>Type: {category}<br>Mentions: 1"
                        }
                    else:
                        nodes_dict[ent_text]["value"] += 1
                        nodes_dict[ent_text]["title"] = f"<b>{ent_text}</b><br>Type: {category}<br>Mentions: {nodes_dict[ent_text]['value']}"
                        
        # Create edges for co-occurring entities in this sentence
        unique_ents = list(set([e[0] for e in sent_entities]))
        for i in range(len(unique_ents)):
            for j in range(i + 1, len(unique_ents)):
                ent1, ent2 = sorted([unique_ents[i], unique_ents[j]])
                edge = (ent1, ent2)
                edges_dict[edge] = edges_dict.get(edge, 0) + 1
                
    # Count frequencies and get top 5 per category
    result = {}
    for category, entities in entities_by_type.items():
        if entities:
            # Count and get top 5
            top_entities = Counter(entities).most_common(5)
            result[category] = top_entities
            
    # Print summary
    for cat, items in result.items():
        print(f"  {cat}: {len(items)} top items")
        
    # To prevent graph clutter, dynamically limit nodes based on density
    max_nodes = 20 # Reduced from 30 for better clarity
    top_nodes = sorted(nodes_dict.values(), key=lambda x: x["value"], reverse=True)[:max_nodes]
    top_node_ids = {n["id"] for n in top_nodes}
    
    # Filter edges: Only keep edges where both nodes are in the top 20
    candidate_edges = [{"from": edge[0], "to": edge[1], "value": weight, "title": f"Co-occurred {weight} times in same sentences"} 
                       for edge, weight in edges_dict.items() 
                       if edge[0] in top_node_ids and edge[1] in top_node_ids]
                       
    # Sort edges by strength and limit to top 30 to prevent clutter while guaranteeing lines
    candidate_edges.sort(key=lambda x: x["value"], reverse=True)
    if len(candidate_edges) > 30:
        candidate_edges = candidate_edges[:30]
        
    graph_data = {"nodes": top_nodes, "edges": candidate_edges}
        
    return result, graph_data
