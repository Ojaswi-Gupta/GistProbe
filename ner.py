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
        return {}
        
    print("\n--- Phase 2.8: Named Entity Recognition ---")
    
    # Process all text together
    # We use raw text (not cleaned) because capitalization helps NER
    full_text = " ".join(df["text"].dropna().astype(str).tolist())
    
    # Process with spaCy (increase max_length if text is very long, but usually standard is fine)
    # We truncate to 1,000,000 chars just to be safe
    doc = nlp(full_text[:1000000])
    
    entities_by_type = {v: [] for v in set(ENTITY_MAPPING.values())}
    
    # Extract and categorize
    for ent in doc.ents:
        # Map label, default to None if we don't care about it
        category = ENTITY_MAPPING.get(ent.label_)
        if category:
            # Clean up the entity text
            ent_text = ent.text.strip().replace('\n', ' ')
            if len(ent_text) > 1:  # Skip single characters
                entities_by_type[category].append(ent_text)
                
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
        
    return result
