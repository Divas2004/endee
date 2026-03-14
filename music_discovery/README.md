# 🎵 Music Discovery Engine

> **Discover music by mood, vibe, and feeling** — powered by semantic AI search and [Endee Vector Database](https://github.com/endee-io/endee).

Describe a scene like *"driving through rain at night"* or *"cozy morning coffee"* and instantly find the perfect tracks through vector similarity search. Build playlists, explore similar tracks, and discover music the way you feel it.

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [How Endee is Used](#-how-endee-is-used)
- [Tech Stack](#-tech-stack)
- [Setup & Installation](#-setup--installation)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)

---

## 🎯 Problem Statement

Traditional music search relies on exact keywords — song titles, artist names, or rigid genre categories. But people *think* about music in terms of **moods, scenes, and feelings**:

- *"Something for a rainy night with a book"*
- *"High energy pump-up for the gym"*
- *"Chill lo-fi for late-night coding"*

This project bridges that gap using **semantic vector search**. Each track's metadata (title, artist, genre, mood tags, and description) is embedded into a high-dimensional vector space. User queries are embedded the same way, and Endee finds the closest matches by **cosine similarity** — no exact keyword matching needed.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔍 **Mood-Based Search** | Describe any vibe in natural language and get matching tracks |
| 🎛️ **Hybrid Filter Engine** | Combine vector search with precise Genre and Era metadata filtering |
| 💡 **AI Match Insights** | Real-time highlights explaining *why* the AI chose each track |
| 🎨 **Mood Presets** | One-click buttons: Rainy Evening, Night Drive, Workout Energy, etc. |
| 🔗 **Find Similar** | Click any track to discover semantically related music in a discovery drawer |
| 🎧 **Playlist Builder** | Add/remove tracks to a persistent session playlist |
| 📊 **Similarity Scores** | See how closely each result matches your query (percentage) |
| 🌙 **Premium Dark UI** | Glassmorphism, animated gradients, and high-quality micro-animations |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User Browser                        │
│  ┌─────────────────────────────────────────────────────┐ │
│  │   Frontend (HTML/CSS/JS)                            │ │
│  │   • Search bar + Mood presets                       │ │
│  │   • Results grid with track cards                   │ │
│  │   • Playlist panel                                  │ │
│  └──────────────────────┬──────────────────────────────┘ │
└─────────────────────────┼────────────────────────────────┘
                          │ HTTP (JSON)
                          ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend (:8000)                      │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Sentence Transformers (all-MiniLM-L6-v2)           │ │
│  │  • Embeds user queries → 384-dim vectors            │ │
│  └──────────────────────┬──────────────────────────────┘ │
│                         │                                │
│  Endpoints:             │                                │
│  POST /api/search       │ query → vector → Endee search  │
│  GET  /api/similar/:id  │ track → re-embed → Endee       │
│  GET  /api/stats        │ index metadata                 │
└─────────────────────────┼────────────────────────────────┘
                          │ HTTP API
                          ▼
┌─────────────────────────────────────────────────────────┐
│            Endee Vector Database (:8080)                  │
│                                                          │
│  Index: "music"                                          │
│  • 100 tracks embedded as 384-dim vectors                │
│  • Cosine similarity search                              │
│  • Payload filtering on metadata (genre, mood, year)     │
│  • INT8 precision for fast retrieval                     │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Indexing** (one-time): `indexer.py` combines each track's title, artist, genre, mood, and description into text → embeds with Sentence Transformers → upserts vectors + metadata to Endee
2. **Search**: User types a mood → backend embeds the query → Endee returns top-K nearest vectors → frontend displays results with similarity %
3. **Similar**: User clicks "Similar" on a track → backend re-embeds that track's metadata → Endee finds neighbors → displayed below

---

## 🗄️ How Endee is Used

Endee is the **core vector storage and retrieval engine** for this project:

### Index Creation
```python
from endee import Endee, Precision

client = Endee()
client.create_index(
    name="music",
    dimension=384,           # Matches all-MiniLM-L6-v2 output
    space_type="cosine",     # Cosine similarity for text embeddings
    precision=Precision.INT8 # Fast, memory-efficient retrieval
)
```

### Vector Upsert (Indexing)
```python
index = client.get_index(name="music")
index.upsert([{
    "id": "t001",
    "vector": [0.012, -0.034, ...],  # 384-dim embedding
    "meta": {
        "title": "Midnight Rain",
        "artist": "Lofi Dreamer",
        "genre": "Lo-fi",
        "mood": "Chill, Rainy, Mellow",
        "description": "Soft piano loops over gentle rain..."
    }
}])
```

### Vector Search (Query)
```python
query_vector = model.encode(["chill rainy evening"])[0].tolist()
results = index.query(vector=query_vector, top_k=8)
# Returns tracks ranked by cosine similarity
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Vector Database | [Endee](https://github.com/endee-io/endee) |
| Embeddings | [Sentence Transformers](https://www.sbert.net/) (all-MiniLM-L6-v2) |
| Backend | Python + FastAPI |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Containerization | Docker + Docker Compose |

---

## 🚀 Setup & Installation

### Prerequisites

- **Docker** & **Docker Compose** installed
- **Python 3.9+** installed
- **pip** package manager

### Step 1: Start Endee

```bash
cd music_discovery
docker compose up -d
```

Verify Endee is running:
```bash
curl http://localhost:8080
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Index the Music Data

```bash
python indexer.py
```

This will:
- Load the embedding model (~90MB download on first run)
- Create the `music` index in Endee
- Embed and upsert 100 curated tracks

### Step 4: Start the API Server

```bash
python main.py
```

### Step 5: Open the App

Navigate to **http://localhost:8000** in your browser.

---

## 🎮 Usage

1. **Type a mood** in the search bar — e.g., *"peaceful nature walk in the forest"*
2. **Click a mood preset** for quick discovery — Rainy Evening, Night Drive, Workout, etc.
3. **Browse results** — each card shows the track, artist, genre, mood tags, and similarity score
4. **Click "Similar"** on any track to find related music
5. **Build a playlist** — click "➕ Playlist" to add tracks, click the 🎧 button to view your playlist

---

## 📡 API Reference

### `POST /api/search`
Search for tracks by mood/vibe text.

**Request:**
```json
{
    "query": "chill rainy evening",
    "top_k": 8,
    "filters": {
        "genre": "Lo-fi",
        "year": "2020s"
    }
}
```

**Response:**
```json
{
    "query": "chill rainy evening",
    "results": [
        {
            "id": "t001",
            "similarity": 89.3,
            "title": "Midnight Rain",
            "artist": "Lofi Dreamer",
            "genre": "Lo-fi",
            "mood": "Chill, Rainy, Mellow",
            "year": "2023",
            "description": "Soft piano loops over gentle rain sounds..."
        }
    ]
}
```

### `GET /api/similar/{track_id}?top_k=6`
Find tracks similar to a given track.

### `GET /api/stats`
Get index statistics (total tracks, model info).

---

## 📁 Project Structure

```
music_discovery/
├── docker-compose.yml   # Endee server configuration
├── requirements.txt     # Python dependencies
├── indexer.py           # Data embedding & indexing pipeline
├── main.py              # FastAPI backend server
├── README.md            # This file
└── frontend/
    ├── index.html       # Single-page application
    └── style.css        # Premium dark-mode styling
```

---

## 📄 License

This project is built on top of the [Endee](https://github.com/endee-io/endee) open-source vector database (Apache License 2.0).
