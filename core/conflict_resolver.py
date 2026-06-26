"""
Conflict-aware retrieval resolver.

Designed for questions such as "Did I mention anything about my sister?" where
the same entity appears in several checkpoints/topics with conflicting context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


EMOTION_WORDS = {
    "love", "happy", "excited", "miss", "sad", "cry", "hurt", "angry", "mad",
    "worried", "anxious", "scared", "hate", "frustrated", "proud", "upset",
}
NEGATIONS = {"not", "never", "no", "none", "without", "don't", "didn't", "isn't", "wasn't"}
POSITIVE = {"love", "like", "close", "help", "helped", "support", "proud", "happy", "excited"}
NEGATIVE = {"hate", "fight", "fought", "angry", "upset", "hurt", "annoyed", "frustrated", "distant"}


def extract_focus_terms(query: str) -> list[str]:
    lower = query.lower()
    relationship_terms = ["sister", "brother", "mother", "mom", "father", "dad", "friend", "boss"]
    hits = [term for term in relationship_terms if term in lower]
    if hits:
        return hits
    words = re.findall(r"[a-zA-Z][a-zA-Z']+", lower)
    stop = {"did", "mention", "anything", "about", "what", "tell", "me", "my", "the"}
    return [w for w in words if len(w) > 2 and w not in stop][:3]


def emotional_weight(text: str) -> float:
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    if not tokens:
        return 0.0
    emotion_hits = sum(1 for t in tokens if t in EMOTION_WORDS)
    punctuation = min(3, text.count("!") + text.count("?")) * 0.15
    return min(1.0, emotion_hits / 4 + punctuation)


def stance(text: str) -> str:
    tokens = set(re.findall(r"[a-zA-Z']+", text.lower()))
    if tokens & NEGATIVE:
        return "negative"
    if tokens & POSITIVE:
        return "positive"
    if tokens & NEGATIONS:
        return "negated"
    return "neutral"


def has_contradiction(ranked: list[dict]) -> bool:
    stances = {r["stance"] for r in ranked if r["stance"] != "neutral"}
    return ("positive" in stances and "negative" in stances) or ("negated" in stances and len(stances) > 1)


def _snippet(text: str, terms: list[str]) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for s in sentences:
        if any(t in s.lower() for t in terms):
            return s[:260]
    return text[:260]


class ConflictResolver:
    def __init__(self, retriever):
        self.retriever = retriever

    def resolve(self, query: str, top_k: int = 12) -> dict:
        terms = extract_focus_terms(query)
        expanded_query = query + " " + " ".join(terms)
        results = self.retriever.retrieve(expanded_query, top_k_topics=6, top_k_msgs=max(top_k, 12))
        candidates = []
        all_msg_ids = [m.get("msg_id", 0) for m in self.retriever.msg_meta] or [0]
        newest = max(all_msg_ids)
        seen_ids = set()

        vector_hits = list(results["message_results"])
        lexical_hits = []
        if terms:
            for m in self.retriever.msg_meta:
                text = m.get("text", "")
                if any(t in text.lower() for t in terms):
                    lexical_hits.append({**m, "score": 0.25})

        for m in vector_hits + lexical_hits:
            if m.get("msg_id") in seen_ids:
                continue
            seen_ids.add(m.get("msg_id"))
            text = m["text"]
            if terms and not any(t in text.lower() for t in terms):
                continue
            recency = m["msg_id"] / max(1, newest)
            emotion = emotional_weight(text)
            lexical = float(m.get("score", 0.0))
            final_score = 0.45 * lexical + 0.35 * recency + 0.20 * emotion
            candidates.append({
                **m,
                "recency": round(recency, 4),
                "emotional_weight": round(emotion, 4),
                "stance": stance(text),
                "resolver_score": round(final_score, 4),
                "snippet": _snippet(text, terms),
            })

        ranked = sorted(candidates, key=lambda r: r["resolver_score"], reverse=True)[:top_k]
        contradiction = has_contradiction(ranked)
        answer = self._merged_answer(query, terms, ranked, contradiction)
        return {
            "query": query,
            "focus_terms": terms,
            "answer": answer,
            "contradiction_flag": contradiction,
            "ranked_chunks": ranked,
            "topic_sources": results["topic_results"],
            "ranking_policy": "0.45 lexical + 0.35 recency + 0.20 emotional_weight",
        }

    def _merged_answer(self, query: str, terms: list[str], ranked: list[dict], contradiction: bool) -> str:
        if not ranked:
            focus = ", ".join(terms) or "that"
            return f"I could not find a clear mention of {focus} in the retrieved conversation chunks."

        focus = ", ".join(terms) or "the topic"
        lines = [f"Yes, there are mentions related to {focus}."]
        if contradiction:
            lines.append("The retrieved chunks contain mixed context, so I would treat this as a contradicted memory rather than one clean fact.")
        else:
            lines.append("The retrieved chunks are broadly consistent.")

        evidence = []
        for r in ranked[:4]:
            evidence.append(f"msg {r['msg_id']} ({r['stance']}, recent={r['recency']}, emotion={r['emotional_weight']}): {r['snippet']}")
        lines.append("Merged evidence: " + " | ".join(evidence))
        return " ".join(lines)
