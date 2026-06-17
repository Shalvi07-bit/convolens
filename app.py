"""
app.py — Flask REST API + static frontend.
Run: ANTHROPIC_API_KEY=sk-ant-... python app.py
"""
import json, sys, os
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent))
from core.retriever import RAGRetriever
from core.summarizer import generate_answer

DATA_DIR  = Path("data")
INDEX_DIR = DATA_DIR / "index"
FRONTEND  = Path("frontend")

def load_artefacts():
    retriever = RAGRetriever()
    retriever.load(str(INDEX_DIR))
    persona     = json.loads((DATA_DIR/"persona.json").read_text())        if (DATA_DIR/"persona.json").exists()        else {}
    topics      = json.loads((DATA_DIR/"topic_segments.json").read_text()) if (DATA_DIR/"topic_segments.json").exists() else []
    checkpoints = json.loads((DATA_DIR/"checkpoints.json").read_text())    if (DATA_DIR/"checkpoints.json").exists()    else []
    print("✅ Artefacts loaded.")
    return retriever, persona, topics, checkpoints

retriever, PERSONA, TOPICS, CHECKPOINTS = load_artefacts()
app = Flask(__name__, static_folder=str(FRONTEND))
CORS(app)

@app.get("/")
def index(): return send_from_directory(str(FRONTEND), "index.html")

@app.get("/health")
def health(): return jsonify({"status": "ok"})

@app.get("/persona")
def get_persona(): return jsonify(PERSONA)

@app.get("/topics")
def get_topics(): return jsonify(TOPICS)

@app.get("/checkpoints")
def get_checkpoints(): return jsonify(CHECKPOINTS)

@app.post("/chat")
def chat():
    body  = request.get_json(force=True)
    query = (body.get("message") or "").strip()
    if not query:
        return jsonify({"error": "message required"}), 400
    results         = retriever.retrieve(query, top_k_topics=3, top_k_msgs=5)
    topic_summaries = [r["summary"] for r in results["topic_results"]]
    message_chunks  = [f"[msg {r['msg_id']} | {r['speaker']}]: {r['text']}" for r in results["message_results"]]
    answer          = generate_answer(query, topic_summaries, message_chunks, PERSONA)
    return jsonify({"answer": answer, "topic_sources": results["topic_results"], "message_sources": results["message_results"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 ConvoLens at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
