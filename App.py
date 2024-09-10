import streamlit as st
from rank_bm25 import BM25Okapi
import pandas as pd
import numpy as np

# Load your dataset
filepath = '/content/mtsamples.csv'
data = pd.read_csv(filepath)

# Tokenize function
def tokenize(text):
    return text.split(', ')

# Prepare the documents
# Check if 'rake_keywords' exists, if not try 'keywords'
if 'rake_keywords' in data.columns:
    data['rake_keywords'] = data['rake_keywords'].fillna('')
    documents = data['rake_keywords'].apply(tokenize).tolist()
elif 'keywords' in data.columns: 
    data['keywords'] = data['keywords'].fillna('')
    documents = data['keywords'].apply(tokenize).tolist()
else:
    # Handle the case where neither column exists
    st.write("Error: Could not find 'rake_keywords' or 'keywords' column in the dataset.") 
    st.stop()

# Initialize BM25
bm25 = BM25Okapi(documents)

# Define the search function
def search(query, bm25_model):
    query_tokens = tokenize(query)
    scores = bm25_model.get_scores(query_tokens)
    return scores

# Streamlit app
st.title('Medical Transcription Search Engine')

st.write("Enter keywords to search for relevant medical transcriptions.")

# User input
query = st.text_input("Search Query")

if query:
    # Perform search
    scores = search(query, bm25)
    ranked_indices = np.argsort(scores)[::-1]
    top_documents = data.iloc[ranked_indices].head(10)  # Get top 10 documents

    # Display results
    st.write("Top results:")
    for idx, row in top_documents.iterrows():
        # Check for the correct column name to display
        if 'rake_keywords' in data.columns:
            st.write(f"**Sample Name:** {row['sample_name']}")
            st.write(f"**Keywords:** {row['rake_keywords']}")
        elif 'keywords' in data.columns:
            st.write(f"**Sample Name:** {row['sample_name']}")
            st.write(f"**Keywords:** {row['keywords']}")
        st.write("---")