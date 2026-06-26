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
from core.intent_classifier import OfflineIntentClassifier
from core.conflict_resolver import ConflictResolver

DATA_DIR  = Path("data")
INDEX_DIR = DATA_DIR / "index"
FRONTEND  = Path("frontend")

def load_artefacts():
    retriever = RAGRetriever()
    retriever.load(str(INDEX_DIR))
    persona     = json.loads((DATA_DIR/"persona.json").read_text())        if (DATA_DIR/"persona.json").exists()        else {}
    drift       = json.loads((DATA_DIR/"persona_drift.json").read_text())  if (DATA_DIR/"persona_drift.json").exists()  else {}
    topics      = json.loads((DATA_DIR/"topic_segments.json").read_text()) if (DATA_DIR/"topic_segments.json").exists() else []
    checkpoints = json.loads((DATA_DIR/"checkpoints.json").read_text())    if (DATA_DIR/"checkpoints.json").exists()    else []
    intent_path = INDEX_DIR / "intent_classifier.pkl"
    intent_clf  = OfflineIntentClassifier(intent_path) if intent_path.exists() else None
    print("Artefacts loaded.")
    return retriever, persona, drift, topics, checkpoints, intent_clf

retriever, PERSONA, PERSONA_DRIFT, TOPICS, CHECKPOINTS, INTENT_CLF = load_artefacts()
resolver = ConflictResolver(retriever)
app = Flask(__name__, static_folder=str(FRONTEND))
CORS(app)

@app.get("/")
def index(): return send_from_directory(str(FRONTEND), "index.html")

@app.get("/health")
def health(): return jsonify({"status": "ok"})

@app.get("/persona")
def get_persona(): return jsonify(PERSONA)

@app.get("/persona-drift")
def get_persona_drift(): return jsonify(PERSONA_DRIFT)

@app.get("/topics")
def get_topics(): return jsonify(TOPICS)

@app.get("/checkpoints")
def get_checkpoints(): return jsonify(CHECKPOINTS)

@app.post("/intent")
def classify_intent():
    if INTENT_CLF is None:
        return jsonify({"error": "intent classifier not built; run build_pipeline.py"}), 503
    body = request.get_json(force=True)
    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message required"}), 400
    return jsonify(INTENT_CLF.classify(message))

@app.post("/resolve-conflict")
def resolve_conflict():
    body = request.get_json(force=True)
    query = (body.get("message") or body.get("query") or "").strip()
    if not query:
        return jsonify({"error": "message required"}), 400
    return jsonify(resolver.resolve(query))

@app.post("/chat")
def chat():
    body  = request.get_json(force=True)
    query = (body.get("message") or "").strip()
    if not query:
        return jsonify({"error": "message required"}), 400
    if any(term in query.lower() for term in ["sister", "brother", "mother", "mom", "father", "dad", "friend"]):
        resolved = resolver.resolve(query)
        if resolved["ranked_chunks"]:
            return jsonify({
                "answer": resolved["answer"],
                "conflict_resolution": resolved,
                "topic_sources": resolved["topic_sources"],
                "message_sources": resolved["ranked_chunks"],
            })
    results         = retriever.retrieve(query, top_k_topics=3, top_k_msgs=5)
    topic_summaries = [r["summary"] for r in results["topic_results"]]
    message_chunks  = [f"[msg {r['msg_id']} | {r['speaker']}]: {r['text']}" for r in results["message_results"]]
    answer          = generate_answer(query, topic_summaries, message_chunks, PERSONA)
    return jsonify({"answer": answer, "topic_sources": results["topic_results"], "message_sources": results["message_results"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nConvoLens at http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
