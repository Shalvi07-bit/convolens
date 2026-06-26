# ConvoLens — RAG Chatbot for Conversation Intelligence

A full end-to-end RAG system that ingests a multi-conversation CSV, builds topic checkpoints, extracts a user persona, and powers a chatbot. 

---

## L2 Assignment Additions

This repo now includes the required L2 modules:

| Part | File | Output |
|------|------|--------|
| Adaptive persona drift | `core/persona_drift.py` | `data/persona_drift.json` with `Day N -> mood & tone` timeline and drift triggers |
| Offline intent classifier | `core/intent_classifier.py` | `data/index/intent_classifier.pkl` and `data/intent_benchmark.json` |
| Conflict-aware RAG resolver | `core/conflict_resolver.py` | `/resolve-conflict` API with recency + emotional weighting and contradiction flags |
| System design doc | `SYSTEM_DESIGN.md` | One-page sync architecture |
| Self evaluation | `SELF_EVALUATION.md` | Constraints, limitations, next steps |

### Quick L2 Test

```bash
python build_pipeline.py --csv data/conversations.csv --skip-summarize --force
python app.py
```

Then test:

```bash
curl -X POST http://localhost:5000/intent -H "Content-Type: application/json" -d "{\"message\":\"remind me to call my sister tomorrow\"}"
curl -X POST http://localhost:5000/resolve-conflict -H "Content-Type: application/json" -d "{\"message\":\"Did I mention anything about my sister?\"}"
curl http://localhost:5000/persona-drift
```

The intent module is fully offline and uses no OpenAI/Gemini/Groq calls. It is a small TF-IDF + LogisticRegression model designed to stay well under 50 MB and under 200 ms per message on CPU.

---

## Architecture Overview

```
CSV  ──► ingest.py  ──► flat chronological message list
                              │
              ┌───────────────┼──────────────────┐
              ▼               ▼                  ▼
    100-msg checkpoints   topic_detector.py   persona_extractor.py
    (every 100 msgs)      (sliding-window     (Claude API over
    → Claude summary       cosine sim)         sampled User 1 msgs)
                              │
                         topic segments
                         → Claude summary
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
              FAISS topic index    FAISS message index
                    └─────────┬──────────┘
                              ▼
                         app.py (Flask)
                              │
                         frontend/index.html
```

---

## How Topic Detection Works

**Algorithm: Sliding-Window Centroid Cosine Similarity**

1. Every message is encoded with `all-MiniLM-L6-v2` (a 384-dim sentence transformer, runs entirely locally — no API needed for detection).
2. A window of `W=10` messages slides forward in steps of `S=5`.
3. At each step, the mean embedding (centroid) of the current window is compared to the centroid of the previous window using cosine similarity.
4. If similarity drops below `THRESHOLD=0.70`, a **topic boundary** is declared.
5. Each resulting segment is independently summarised by Claude.

**Why this works:**
- Conversations naturally shift vocabulary and semantic content when topics change.
- Windowed averaging smooths out noise from single off-topic messages.
- No LLM is needed for detection — only for generating summaries.

**Output:**
```
Topic 0 → msgs 0–127   → "Users discuss moving to Portland and culinary dreams…"
Topic 1 → msgs 128–310 → "Conversation shifts to music, radiology studies, and yoga…"
...
```

---

## How Retrieval Works

Two separate FAISS `IndexFlatIP` (inner-product) indices are built after normalising embeddings, making it equivalent to cosine similarity search:

| Index | What's stored | Vector = |
|-------|--------------|---------|
| **Topic index** | One entry per topic segment | Embedding of the segment's summary |
| **Message index** | One entry per raw message | Embedding of the message text |

**At query time:**
1. Query is embedded with the same `all-MiniLM-L6-v2` model.
2. Top-3 topic summaries are retrieved from the topic index.
3. Top-5 individual messages are retrieved from the message index.
4. Both are combined into a context window.
5. Claude generates the final answer grounded in that context.

---

## How Persona is Built

1. All messages attributed to `User 1` are collected.
2. A random sample of up to 500 messages is taken (keeping chronological order).
3. A single Claude API call analyses this sample with a structured prompt requesting JSON output in this schema:

```json
{
  "habits": [...],
  "personal_facts": [...],
  "personality_traits": [...],
  "communication_style": {
    "tone": "...",
    "message_length": "...",
    "emoji_usage": "...",
    "notable_patterns": [...]
  },
  "interests": [...]
}
```

**Key constraint:** The prompt explicitly instructs Claude to *only include things directly inferable from the text*, avoiding hallucination.

---

## Setup & Run

### Prerequisites
- Python 3.10+
- An `ANTHROPIC_API_KEY`

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Place the CSV
```bash
mkdir data
cp conversations.csv data/conversations.csv
```

### 3. Build the pipeline
```bash
export GROQ_API_KEY=gsk_...

# Full build (may take a while — makes many Claude API calls for summaries)
python build_pipeline.py --csv data/conversations.csv

# Quick test with first 1000 messages, skip Claude summarisation
python build_pipeline.py --csv data/conversations.csv --max-msgs 1000 --skip-summarize
```

This produces:
```
data/
  checkpoints.json       # 100-message checkpoint summaries
  topic_segments.json    # per-topic summaries
  persona.json           # extracted user persona
  index/
    topic.index          # FAISS index for topics
    messages.index       # FAISS index for messages
    topic_meta.json
    msg_meta.json
```

### 4. Start the server
```bash
python app.py
# → http://localhost:5000
```

---

## Cloud Deployment (Render / Railway / Fly.io)

### Render (recommended — free tier)

1. Push your repo to GitHub.
2. Go to [render.com](https://render.com) → New Web Service.
3. Connect your repo, set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `python app.py`
   - **Environment variable:** `GROQ_API_KEY = gsk_...`
4. Upload the `data/` folder outputs (indices + JSON) as part of the repo, **or** run `build_pipeline.py` as a one-time job before the main service starts.

> ⚠️ The FAISS indices and JSON files in `data/` must be committed to the repo (or stored in persistent cloud storage) since cloud VMs don't persist in-memory state.

### Docker
```bash
docker build -t convolens .
docker run -p 5000:5000 -e GROQ_API_KEY=gsk_... \
  -v $(pwd)/data:/app/data convolens
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Chatbot frontend |
| GET | `/health` | Health check |
| GET | `/persona` | Full persona JSON |
| GET | `/topics` | All topic segment summaries |
| GET | `/checkpoints` | All 100-message checkpoint summaries |
| POST | `/chat` | `{"message":"..."}` → `{"answer":"...", "topic_sources":[...], "message_sources":[...]}` |

---

## Tuning Parameters

| File | Parameter | Default | Effect |
|------|-----------|---------|--------|
| `topic_detector.py` | `WINDOW_SIZE` | 10 | Larger = smoother, fewer topics |
| `topic_detector.py` | `THRESHOLD` | 0.70 | Lower = fewer boundaries |
| `persona_extractor.py` | `sample_size` | 500 | More msgs = richer persona |
| `retriever.py` | `top_k_topics` | 3 | More retrieved contexts |
| `retriever.py` | `top_k_msgs` | 5 | More retrieved messages |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) — **local** |
| Vector search | `faiss-cpu` — **local** |
| Summarisation / answers | Groq API — **free** (`llama-3.1-8b-instant`) |
| Backend | Flask + flask-cors |
| Frontend | Vanilla JS + CSS (no build step) |
