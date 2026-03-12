from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from endee import Endee
import os

app = FastAPI(title="Movie Recommender API")

print("Initializing backend embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Connecting to Endee Vector Database...")
try:
    endee_client = Endee()
    index = endee_client.get_index(name="movies")
except Exception as e:
    print(f"Warning: Could not connect to Endee or get index 'movies'. Error: {e}")
    index = None

# Ensure frontend directory exists
os.makedirs("frontend", exist_ok=True)

# Mount frontend static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

class SearchRequest(BaseModel):
    query: str
    top_k: int = 6

@app.get("/")
async def read_index():
    from fastapi.responses import FileResponse
    return FileResponse('frontend/index.html')

@app.post("/api/search")
async def search_movies(req: SearchRequest):
    if not index:
        raise HTTPException(status_code=500, detail="Endee DB connection or index missing.")
    
    if not req.query.strip():
        return {"results": []}

    # Generate embedding for the search query
    query_vector = model.encode([req.query])[0].tolist()
    
    try:
        # Search Endee using the vector
        results = index.query(
            vector=query_vector,
            top_k=req.top_k
        )
        
        # Format results for frontend
        formatted_results = []
        for r in results:
            meta = r.get("meta", {})
            formatted_results.append({
                "id": r.get("id"),
                "similarity": r.get("similarity"),
                "title": meta.get("title", ""),
                "overview": meta.get("overview", ""),
                "genres": meta.get("genres", ""),
                "release_date": meta.get("release_date", ""),
                "vote_average": meta.get("vote_average", 0.0)
            })
            
        return {"results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Make sure we don't block Endee's port 8080
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
