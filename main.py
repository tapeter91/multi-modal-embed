import json
import os
import hashlib
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from google import genai
from google.genai import types

# --- Config ---
CONFIG_FILE = Path("config.json")
SOURCES_DIR = Path("sources")
EMBEDDINGS_FILE = Path("embeddings.json")
MODEL = "gemini-embedding-2-preview"
VIDEO_MAX_SECONDS = 120
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".mov"}
MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".mp4": "video/mp4",
    ".mov": "video/mp4",  # MOV files served as mp4 container
}
VIDEO_EXTENSIONS = {".mp4", ".mov"}


def load_config() -> dict:
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Config file '{CONFIG_FILE}' not found.")
    with open(CONFIG_FILE) as f:
        return json.load(f)


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_embeddings() -> dict:
    if EMBEDDINGS_FILE.exists():
        with open(EMBEDDINGS_FILE) as f:
            return json.load(f)
    return {}


def save_embeddings(data: dict) -> None:
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(data, f)


def get_video_duration(video_path: Path) -> float:
    """Return video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def read_video_bytes(video_path: Path) -> bytes:
    """Read video bytes, trimming to VIDEO_MAX_SECONDS if needed."""
    duration = get_video_duration(video_path)
    if duration <= VIDEO_MAX_SECONDS:
        with open(video_path, "rb") as f:
            return f.read()

    print(f" (trimming {duration:.1f}s -> {VIDEO_MAX_SECONDS}s)", end="", flush=True)
    with tempfile.NamedTemporaryFile(suffix=video_path.suffix, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-t", str(VIDEO_MAX_SECONDS),
                "-c", "copy",
                tmp_path,
            ],
            capture_output=True, check=True,
        )
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def embed_file(client: genai.Client, file_path: Path) -> list[float]:
    ext = file_path.suffix.lower()
    mime = MIME_TYPES[ext]
    if ext in VIDEO_EXTENSIONS:
        data = read_video_bytes(file_path)
    else:
        with open(file_path, "rb") as f:
            data = f.read()
    result = client.models.embed_content(
        model=MODEL,
        contents=[types.Part.from_bytes(data=data, mime_type=mime)],
    )
    return result.embeddings[0].values


def embed_text(client: genai.Client, text: str) -> list[float]:
    result = client.models.embed_content(
        model=MODEL,
        contents=[types.Part.from_text(text=text)],
    )
    return result.embeddings[0].values


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


def scan_sources() -> list[Path]:
    if not SOURCES_DIR.exists():
        SOURCES_DIR.mkdir()
        return []
    return [
        p for p in SOURCES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def sync_embeddings(client: genai.Client) -> dict:
    stored = load_embeddings()
    files = scan_sources()

    if not files:
        print("No supported files found in 'sources/' folder.")
        return stored

    # Remove entries for deleted files
    existing_names = {p.name for p in files}
    removed = [name for name in stored if name not in existing_names]
    for name in removed:
        print(f"  Removed stale entry: {name}")
        del stored[name]

    # Embed new or changed files
    to_embed = []
    for f in files:
        h = file_hash(f)
        entry = stored.get(f.name)
        if entry is None or entry.get("hash") != h:
            to_embed.append((f, h))

    if not to_embed:
        print(f"All {len(files)} file(s) already embedded. Nothing to update.")
        return stored

    print(f"Embedding {len(to_embed)} file(s)...")
    for f, h in to_embed:
        print(f"  Embedding: {f.name}", end="", flush=True)
        embedding = embed_file(client, f)
        stored[f.name] = {"hash": h, "embedding": embedding}
        save_embeddings(stored)
        print(" done")

    return stored


class VectorDB:
    def __init__(self, embeddings: dict):
        self.names: list[str] = []
        self.vectors: list[list[float]] = []
        for name, entry in embeddings.items():
            self.names.append(name)
            self.vectors.append(entry["embedding"])

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        scores = [(name, cosine_similarity(query_vector, vec))
                  for name, vec in zip(self.names, self.vectors)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


def main():
    config = load_config()
    api_key = config.get("api_key", "")
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError("Set your API key in config.json")

    client = genai.Client(api_key=api_key)

    print("=== Gemini Embedding Search ===")
    print("Checking embeddings...")
    embeddings = sync_embeddings(client)

    if not embeddings:
        print("No embeddings available. Add images or videos to the 'sources/' folder and restart.")
        return

    db = VectorDB(embeddings)
    print(f"\nLoaded {len(db.names)} file(s) into memory.\n")

    while True:
        try:
            prompt = input("Enter a text prompt to search (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Bye.")
            break

        print("Embedding prompt...", end="", flush=True)
        query_vec = embed_text(client, prompt)
        print(" done\n")

        results = db.search(query_vec, top_k=min(5, len(db.names)))
        print("Top matches:")
        for i, (name, score) in enumerate(results, 1):
            print(f"  {i}. {name}  (similarity: {score:.4f})")
        print()


if __name__ == "__main__":
    main()
