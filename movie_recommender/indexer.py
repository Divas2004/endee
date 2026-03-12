import os
import time
import json
from typing import List, Dict
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from endee import Endee, Precision

# Initialize models and clients
print("Loading semantic model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Connecting to Endee Vector Database...")
# Assuming Endee runs locally via docker compose on 8080
endee_client = Endee()

INDEX_NAME = "movies"
VECTOR_DIMENSION = 384 # Dimension for all-MiniLM-L6-v2

def setup_endee():
    """Create the index in Endee if it doesn't exist."""
    print(f"Ensuring index '{INDEX_NAME}' exists...")
    try:
        endee_client.create_index(
            name=INDEX_NAME, 
            dimension=VECTOR_DIMENSION, 
            space_type="cosine", 
            precision=Precision.INT8
        )
        print(f"Created new index '{INDEX_NAME}'.")
    except Exception as e:
        # Assuming error means it exists or similar (SDK behavior may vary)
        print(f"Index creation note (likely exists): {e}")

    return endee_client.get_index(name=INDEX_NAME)


def process_and_upload(index):
    """Load dataset, generate embeddings, and upload to Endee."""
    print("Downloading TMDB movie dataset from HuggingFace...")
    # Using a popular and clean 5k movies dataset
    dataset = load_dataset("alejandrowallace/tmdb-5000", split="train")
    
    # Take a subset for the demo to ensure fast ingestion (e.g., 2000 movies)
    # Filter out movies without an overview
    movies = [m for m in dataset if m.get("overview") and isinstance(m["overview"], str) and len(m["overview"].strip()) > 10]
    movies = movies[:2000] # Limit to 2000 for quick demonstration
    
    print(f"Preparing to embed and insert {len(movies)} movies...")
    
    batch_size = 100
    for i in range(0, len(movies), batch_size):
        batch = movies[i:i + batch_size]
        
        # Prepare texts to embed
        texts_to_embed = [
            f"Title: {row.get('original_title', '')}. Overview: {row.get('overview', '')}. Genres: {row.get('genres', '')}"
            for row in batch
        ]
        
        # Generate embeddings
        embeddings = model.encode(texts_to_embed)
        
        # Prepare vectors for Endee
        endee_vectors = []
        for j, row in enumerate(batch):
            doc_id = str(row.get("id", f"movie_{i+j}"))
            
            raw_genres = str(row.get("genres", ""))
            try:
                parsed_genres = json.loads(raw_genres)
                genre_str = "|".join([g.get("name", "") for g in parsed_genres])
            except:
                genre_str = raw_genres

            # Clean metadata
            meta = {
                "title": str(row.get("original_title", "Unknown")),
                "overview": str(row.get("overview", "")),
                "genres": genre_str,
                "release_date": str(row.get("release_date", "")),
                "vote_average": float(row.get("vote_average", 0.0)) if row.get("vote_average") else 0.0,
            }
            
            endee_vectors.append({
                "id": doc_id,
                "vector": embeddings[j].tolist(),
                "meta": meta
            })
            
        # Upsert to Endee
        print(f"Upserting batch {i//batch_size + 1}/{len(movies)//batch_size + (1 if len(movies)%batch_size!=0 else 0)}...")
        try:
            index.upsert(endee_vectors)
        except Exception as e:
            print(f"Warning on upsert: {e}")
            
    print("Ingestion complete!")

if __name__ == "__main__":
    time.sleep(2) # Give Endee DB a moment to be fully ready if just started
    index = setup_endee()
    process_and_upload(index)
