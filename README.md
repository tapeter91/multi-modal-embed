# Gemini Multimodal Embedding Search

Search images and videos using Google's `gemini-embedding-2-preview` model. Drop media files into `sources/`, embed them once, then find them by describing what you see in natural language.

## What it does

1. **Embeds** images and videos in `sources/` into semantic vectors via the Gemini API
2. **Stores** those vectors in `embeddings.json` (only re-embeds files that changed)
3. **Searches** by embedding a text prompt and ranking all files by cosine similarity
4. **Displays** results in a React UI with a search history grid — videos show a seekable frame preview (frames are extracted server-side via ffmpeg)

## Components

| Component | File | Description |
|-----------|------|-------------|
| Embedder CLI | `main.py` | Scans `sources/`, embeds new/changed files, runs an interactive terminal search |
| API backend | `server.py` | FastAPI server — handles search, serves source files, extracts video frames |
| Web frontend | `frontend/` | React + Vite app — search input, results grid, video frame scrubber |

## Prerequisites

- Python 3.10+
- Node.js 18+
- `ffmpeg` / `ffprobe` on PATH (for video trimming and frame extraction)
- A Gemini API key

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
pip install fastapi uvicorn
```

### 2. Configure API key

Edit `config.json`:

```json
{
  "api_key": "YOUR_GEMINI_API_KEY_HERE"
}
```

### 3. Add media files

Copy images (`.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`) or videos (`.mp4`, `.mov`) into the `sources/` folder.

### 4. Embed the files

Run once (or whenever you add/change files):

```bash
python main.py
```

This scans `sources/`, embeds any new or changed files, saves to `embeddings.json`, then drops into an interactive terminal search you can use to test.

### 5. Install frontend dependencies

```bash
cd frontend
npm install
```

## Running the web app

Open **two terminals**:

**Terminal 1 — API backend:**

```bash
uvicorn server:app --reload
```

Runs on `http://localhost:8000`

**Terminal 2 — React frontend:**

```bash
cd frontend
npm run dev
```

Runs on `http://localhost:5173` — open this in your browser.

## Usage

1. Type a description in the search box (e.g. *"someone cooking in a kitchen"*)
2. Press **Search** or `Ctrl+Enter`
3. Results appear as cards in a grid — the top matching file is shown with its similarity score
4. For videos, drag the seek slider to scrub through frames and find the exact moment that matched
5. Each new search adds a card to the history — useful for screenshots showing how multimodal embedding maps prompts to specific media

## Supported file types

| Extension | Type |
|-----------|------|
| `.jpg` `.jpeg` `.png` `.gif` `.webp` | Image |
| `.mp4` `.mov` | Video (trimmed to 120s for embedding if longer) |
