"""
build_pipeline.py — RAG pipeline with rate-limit-safe summarisation.
"""

import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.ingest import load_messages
from core.topic_detector import detect_topics
from core.summarizer import summarize_topic, summarize_checkpoint
from core.persona_extractor import extract_persona
from core.retriever import RAGRetriever

DATA_DIR  = Path("data")
INDEX_DIR = DATA_DIR / "index"
CHECKPOINT_FILE = DATA_DIR / "checkpoints.json"
TOPICS_FILE     = DATA_DIR / "topic_segments.json"
PERSONA_FILE    = DATA_DIR / "persona.json"

# ── Limits to stay within Groq free tier ────────────────────────────────────
MAX_TOPICS      = 60   # summarise at most 60 topics
MAX_CHECKPOINTS = 30   # summarise at most 30 checkpoints


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",            default="data/conversations.csv")
    ap.add_argument("--force",          action="store_true")
    ap.add_argument("--skip-summarize", action="store_true")
    ap.add_argument("--max-msgs",       type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    DATA_DIR.mkdir(exist_ok=True)
    INDEX_DIR.mkdir(exist_ok=True)

    # 1. Ingest
    print("\n[1/5] Loading messages …")
    messages = load_messages(args.csv)
    if args.max_msgs:
        messages = messages[:args.max_msgs]
    print(f"  {len(messages)} messages loaded")

    # 2. 100-message checkpoints
    print("\n[2/5] Building 100-message checkpoints …")
    if CHECKPOINT_FILE.exists() and not args.force:
        print("  [cached]"); checkpoints = json.loads(CHECKPOINT_FILE.read_text())
    else:
        all_chunks = []
        for i in range(0, len(messages), 100):
            chunk = messages[i:i+100]
            if chunk:
                all_chunks.append((i // 100, chunk))

        # Evenly sample up to MAX_CHECKPOINTS
        step = max(1, len(all_chunks) // MAX_CHECKPOINTS)
        selected = all_chunks[::step][:MAX_CHECKPOINTS]
        print(f"  Summarising {len(selected)} of {len(all_chunks)} checkpoints …")

        checkpoints = []
        for cp_num, chunk in all_chunks:  # store ALL, summarise only selected
            if any(cp_num == s[0] for s in selected):
                print(f"  Checkpoint {cp_num} (msgs {chunk[0]['msg_id']}–{chunk[-1]['msg_id']}) …")
                summary = (f"[placeholder] Checkpoint {cp_num}"
                           if args.skip_summarize else summarize_checkpoint(chunk, cp_num))
            else:
                summary = f"[skipped] Checkpoint {cp_num}: msgs {chunk[0]['msg_id']}–{chunk[-1]['msg_id']}"
            checkpoints.append({"checkpoint_num": cp_num,
                                 "start_msg_id": chunk[0]["msg_id"],
                                 "end_msg_id": chunk[-1]["msg_id"],
                                 "summary": summary})

        CHECKPOINT_FILE.write_text(json.dumps(checkpoints, indent=2))
        print(f"  Saved {len(checkpoints)} checkpoints")

    # 3. Topic detection
    print("\n[3/5] Detecting topic segments …")
    if TOPICS_FILE.exists() and not args.force:
        print("  [cached]"); topic_segments = json.loads(TOPICS_FILE.read_text())
    else:
        raw_segments = detect_topics(messages, model=None)
        print(f"  Detected {len(raw_segments)} topic segments")

        # Evenly sample up to MAX_TOPICS for summarisation
        step = max(1, len(raw_segments) // MAX_TOPICS)
        to_summarise = set(s["topic_id"] for s in raw_segments[::step][:MAX_TOPICS])
        print(f"  Summarising {len(to_summarise)} of {len(raw_segments)} topics …")

        topic_segments = []
        for seg in raw_segments:
            if seg["topic_id"] in to_summarise:
                print(f"  Topic {seg['topic_id']} (msgs {seg['start_idx']}–{seg['end_idx']}, {len(seg['messages'])} msgs) …")
                summary = (f"[placeholder] Topic {seg['topic_id']}"
                           if args.skip_summarize else summarize_topic(seg))
            else:
                summary = f"[skipped] Topic {seg['topic_id']}: msgs {seg['start_idx']}–{seg['end_idx']}"
            topic_segments.append({
                "topic_id":  seg["topic_id"],
                "start_idx": seg["start_idx"],
                "end_idx":   seg["end_idx"],
                "msg_count": len(seg["messages"]),
                "summary":   summary,
            })

        TOPICS_FILE.write_text(json.dumps(topic_segments, indent=2))
        print(f"  Saved {len(topic_segments)} topic segments")

    # 4. Persona
    print("\n[4/5] Extracting user persona …")
    if PERSONA_FILE.exists() and not args.force:
        print("  [cached]"); persona = json.loads(PERSONA_FILE.read_text())
    else:
        persona = ({"note": "skipped"} if args.skip_summarize else extract_persona(messages))
        PERSONA_FILE.write_text(json.dumps(persona, indent=2))
        print(f"  Persona saved")

    # 5. FAISS indices
    print("\n[5/5] Building FAISS vector indices …")
    retriever = RAGRetriever()
    retriever.build_topic_index(topic_segments)
    retriever.build_message_index(messages)
    retriever.save(str(INDEX_DIR))

    print("\n✅ Pipeline complete!")
    print(f"   • {len(checkpoints)} checkpoints  • {len(topic_segments)} topics")

if __name__ == "__main__":
    main()
