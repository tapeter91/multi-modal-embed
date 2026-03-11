import json
import subprocess
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from google import genai
from google.genai import types

SOURCES_DIR = Path("sources")
EMBEDDINGS_FILE = Path("embeddings.json")
CONFIG_FILE = Path("config.json")
MODEL = "gemini-embedding-2-preview"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if SOURCES_DIR.exists():
    app.mount("/sources", StaticFiles(directory=str(SOURCES_DIR)), name="sources")


def load_config() -> dict:
    with open(CONFIG_FILE) as f:
        return json.load(f)


def load_embeddings() -> dict:
    if EMBEDDINGS_FILE.exists():
        with open(EMBEDDINGS_FILE) as f:
            return json.load(f)
    return {}


def cosine_similarity(a: list, b: list) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


config = load_config()
_client = genai.Client(api_key=config["api_key"])


class SearchRequest(BaseModel):
    prompt: str
    top_k: int = 5


@app.post("/api/search")
async def search(req: SearchRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    result = _client.models.embed_content(
        model=MODEL,
        contents=[types.Part.from_text(text=req.prompt)],
    )
    query_vec = result.embeddings[0].values

    embeddings = load_embeddings()
    if not embeddings:
        raise HTTPException(status_code=404, detail="No embeddings found. Run main.py first.")

    scores = [
        {"name": name, "score": cosine_similarity(query_vec, entry["embedding"])}
        for name, entry in embeddings.items()
    ]
    scores.sort(key=lambda x: x["score"], reverse=True)
    return {"results": scores[: req.top_k]}


@app.get("/api/video-duration/{filename}")
async def video_duration(filename: str):
    video_path = SOURCES_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    return {"duration": float(result.stdout.strip())}


@app.get("/api/frame/{filename}")
async def get_frame(filename: str, t: float = 0):
    video_path = SOURCES_DIR / filename
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    result = subprocess.run(
        [
            "ffmpeg", "-ss", str(t),
            "-i", str(video_path),
            "-frames:v", "1",
            "-f", "image2",
            "-c:v", "mjpeg",
            "-q:v", "3",
            "pipe:1",
        ],
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        raise HTTPException(status_code=500, detail="Failed to extract frame")
    return Response(content=result.stdout, media_type="image/jpeg")
