"""
Music Discovery Engine — API Server
FastAPI backend serving mood-based music search via Endee Vector DB.
"""
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from endee import Endee
from typing import Optional

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Music Discovery Engine",
    description="Discover music by mood, vibe, and feeling — powered by Endee Vector DB",
    version="1.0.0",
)

# ── Models & Clients ──────────────────────────────────────────────────────────
print("🎵 Music Discovery Engine — Starting up...")
print("📦 Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("🔌 Connecting to Endee Vector Database...")
try:
    endee_client = Endee()
    index = endee_client.get_index(name="music")
    print("   ✅ Connected to Endee, index 'music' loaded.")
except Exception as e:
    print(f"   ⚠️  Could not connect to Endee: {e}")
    index = None

# ── Ensure frontend directory exists ──────────────────────────────────────────
os.makedirs("frontend", exist_ok=True)

# ── Static files ──────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ── Request / Response Models ─────────────────────────────────────────────────
class SearchRequest(BaseModel):
    query: str
    top_k: int = 8
    filters: Optional[dict] = None


class SpotifySearchRequest(BaseModel):
    query: str


# ── Logic Helpers ─────────────────────────────────────────────────────────────
def get_match_insights(query: str, meta: dict) -> list:
    """Extract keywords from query that appear in metadata for explainability."""
    common_words = {"a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "with", "for", "is", "of"}
    keywords = [w.lower().strip(",.!?") for w in query.split() if w.lower() not in common_words]
    
    insights = []
    meta_text = f"{meta.get('title', '')} {meta.get('genre', '')} {meta.get('mood', '')} {meta.get('description', '')}".lower()
    
    for kw in keywords:
        if len(kw) > 2 and kw in meta_text:
            insights.append(kw)
    
    return list(set(insights))[:4]


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    return FileResponse("frontend/index.html")


@app.post("/api/search")
async def search_music(req: SearchRequest):
    """Search for music with optional hybrid filters."""
    if not index:
        raise HTTPException(status_code=500, detail="Endee DB not connected. Run the indexer first.")

    query = req.query.strip()
    if not query:
        return {"results": []}

    # Embed the query
    query_vector = model.encode([query])[0].tolist()

    try:
        # Prepare filters for Endee (expects a list of dicts with operators like $eq)
        endee_filters = []
        if req.filters:
            for key, value in req.filters.items():
                if key == "year":
                    # Filter by explicit 'era' field in metadata
                    endee_filters.append({"era": {"$eq": value}})
                else:
                    endee_filters.append({key: {"$eq": value}})

        # Search Endee with vector and optional filters
        results = index.query(
            vector=query_vector, 
            top_k=req.top_k,
            filter=endee_filters if endee_filters else None
        )

        formatted = []
        for r in results:
            meta = r.get("meta", {})
            similarity = r.get("similarity", 0)
            similarity_pct = round(similarity * 100, 1)

            # Generate explainability insights
            insights = get_match_insights(query, meta)

            formatted.append({
                "id": r.get("id"),
                "similarity": similarity_pct,
                "title": meta.get("title", ""),
                "artist": meta.get("artist", ""),
                "album": meta.get("album", ""),
                "genre": meta.get("genre", ""),
                "mood": meta.get("mood", ""),
                "year": meta.get("year", ""),
                "preview_url": meta.get("preview_url", ""),
                "description": meta.get("description", ""),
                "insights": insights
            })

        return {"results": formatted, "query": query, "filters_applied": req.filters}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/spotify-search")
async def spotify_search(req: SpotifySearchRequest):
    """
    Generate a direct Spotify search URL.
    """
    from urllib.parse import quote
    encoded_query = quote(req.query)
    spotify_url = f"https://open.spotify.com/search/{encoded_query}"
    
    return {
        "query": req.query,
        "spotify_url": spotify_url
    }


@app.get("/api/similar/{track_id}")
async def find_similar(track_id: str, top_k: int = 6):
    """Find tracks similar to a given track by re-querying with its vector."""
    if not index:
        raise HTTPException(status_code=500, detail="Endee DB not connected.")

    try:
        # First, get the track's data by searching for it by ID
        # We'll search using the track's metadata to find similar ones
        # Since Endee might not have a direct get-by-id, we search with a
        # text composed from the track's known metadata
        all_results = index.query(
            vector=model.encode(["placeholder"]).tolist()[0],  # dummy
            top_k=200
        )

        # Find the target track
        target = None
        for r in all_results:
            if r.get("id") == track_id:
                target = r
                break

        if not target:
            raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found.")

        # Build a query text from the target track's metadata
        meta = target.get("meta", {})
        query_text = (
            f"Title: {meta.get('title', '')}. "
            f"Artist: {meta.get('artist', '')}. "
            f"Genre: {meta.get('genre', '')}. "
            f"Mood: {meta.get('mood', '')}. "
            f"Description: {meta.get('description', '')}"
        )

        # Re-embed and search
        query_vector = model.encode([query_text])[0].tolist()
        results = index.query(vector=query_vector, top_k=top_k + 1)  # +1 to exclude self

        formatted = []
        for r in results:
            if r.get("id") == track_id:
                continue  # Skip the track itself
            m = r.get("meta", {})
            similarity = r.get("similarity", 0)
            similarity_pct = round(similarity * 100, 1)

            formatted.append({
                "id": r.get("id"),
                "similarity": similarity_pct,
                "title": m.get("title", ""),
                "artist": m.get("artist", ""),
                "album": m.get("album", ""),
                "genre": m.get("genre", ""),
                "mood": m.get("mood", ""),
                "year": m.get("year", ""),
                "preview_url": m.get("preview_url", ""),
                "description": m.get("description", ""),
            })

        # Only return top_k results
        return {
            "source_track": {
                "id": track_id,
                "title": meta.get("title", ""),
                "artist": meta.get("artist", ""),
            },
            "results": formatted[:top_k]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Return basic index statistics."""
    if not index:
        return {"status": "disconnected", "total_tracks": 0}

    return {
        "status": "connected",
        "index_name": "music",
        "total_tracks": 100,
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_dimension": 384,
    }


# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("\n🚀 Starting Music Discovery Engine on http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
