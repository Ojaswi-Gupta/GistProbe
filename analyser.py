# import pandas as pd
# import re

# df = pd.read_csv("data.csv")
# df = df.drop_duplicates()
# cleaned = []

# for text in df["text"]:
#     text = str(text)
#     text = re.sub(r'[^a-zA-Z ]', '', text)
#     text = text.lower()
#     cleaned.append(text)

# df["cleaned"] = cleaned

# # remove empty rows
# df = df[df["cleaned"].str.strip() != ""]

# print(df.head())

# df.to_csv("data.csv", index=False)

# print("Text cleaned successfully")



import re
import pandas as pd

def clean_text_data(df):
    """Phase 2: Analysis & Deduplication. Standardizes text for the model."""
    if df.empty:
        return df

    # Stage 1: Remove exact raw duplicates
    df = df.drop_duplicates(subset=["text"]).copy()

    def clean_logic(text):
        text = str(text)
        # Regex: Keep only letters and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.lower().strip()

    df["cleaned"] = df["text"].apply(clean_logic)
    
    # Stage 2: Remove empty rows and semantic duplicates (same text after cleaning)
    df = df[df["cleaned"] != ""].copy()
    df = df.drop_duplicates(subset=["cleaned"]).copy()
    
    # Terminal Logs
    print(df.head())
    print("Text cleaned successfully")
    
    return df