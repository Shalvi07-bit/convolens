"""
Persona drift detector.

Tracks User 1's day-by-day tone/mood instead of producing only one global
persona. In this dataset each CSV row is treated as one chronological day.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


MOOD_TERMS = {
    "frustrated": {"annoyed", "angry", "mad", "frustrated", "irritated", "ugh", "hate", "tired"},
    "sad": {"sad", "cry", "lonely", "hurt", "upset", "miss", "depressed"},
    "anxious": {"anxious", "worried", "nervous", "scared", "panic", "stress", "stressed"},
    "happy": {"happy", "glad", "excited", "great", "amazing", "love", "yay"},
    "curious": {"why", "how", "what", "when", "where", "wonder", "curious", "?"},
    "playful": {"lol", "haha", "lmao", "funny", "joke", "tease"},
}

FORMAL_TERMS = {"please", "thanks", "thank you", "could", "would", "kindly"}
CASUAL_TERMS = {"lol", "haha", "hey", "yeah", "yep", "nah", "bro", "btw", "idk"}
PERSON_RE = re.compile(r"\b(?:mom|mother|dad|father|sister|brother|friend|boss|teacher|sir|ma'am|[A-Z][a-z]{2,})\b")
STOP_TOPICS = {
    "this", "that", "with", "from", "have", "about", "your", "just", "like",
    "really", "would", "could", "there", "their", "what", "when", "where",
}


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+|\?", text.lower())


def _keywords(text: str, limit: int = 5) -> list[str]:
    words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z']+", text)]
    counts = Counter(w for w in words if len(w) > 3 and w not in STOP_TOPICS)
    return [w for w, _ in counts.most_common(limit)]


def _dominant_mood(tokens: list[str]) -> tuple[str, dict[str, int]]:
    scores = {}
    token_set = set(tokens)
    for mood, terms in MOOD_TERMS.items():
        scores[mood] = sum(1 for t in tokens if t in terms) + sum(1 for t in terms if " " in t and t in " ".join(tokens))
    if not any(scores.values()):
        return "neutral", scores
    return max(scores, key=scores.get), scores


def _tone(text: str, tokens: list[str]) -> str:
    lower = text.lower()
    formal = sum(1 for t in FORMAL_TERMS if t in lower)
    casual = sum(1 for t in CASUAL_TERMS if t in lower) + text.count("!")
    avg_len = sum(len(t) for t in tokens if t != "?") / max(1, len([t for t in tokens if t != "?"]))
    if casual > formal:
        return "casual"
    if formal > casual or avg_len > 5.2:
        return "formal"
    return "plain"


def _trigger_for(prev: dict, curr: dict) -> dict:
    new_topics = [t for t in curr["keywords"] if t not in prev["keywords"]]
    new_people = [p for p in curr["people"] if p not in prev["people"]]
    if new_people:
        return {"type": "person", "value": new_people[0]}
    if new_topics:
        return {"type": "topic", "value": new_topics[0]}
    return {"type": "tone-shift", "value": f"{prev['label']} -> {curr['label']}"}


def detect_persona_drift(messages: list[dict], persona: dict | None = None) -> dict:
    by_day: dict[int, list[dict]] = defaultdict(list)
    for m in messages:
        if "1" in str(m.get("speaker", "")):
            by_day[int(m.get("conv_id", 0)) + 1].append(m)

    timeline = []
    for day in sorted(by_day):
        day_msgs = by_day[day]
        text = " ".join(m["text"] for m in day_msgs)
        tokens = _tokens(text)
        mood, mood_scores = _dominant_mood(tokens)
        tone = _tone(text, tokens)
        label = f"{mood} & {tone}" if mood != "neutral" else tone
        people = sorted(set(PERSON_RE.findall(text)))
        questions = text.count("?")
        exclaims = text.count("!")
        timeline.append({
            "day": day,
            "message": f"Day {day} -> {label}",
            "label": label,
            "mood": mood,
            "tone": tone,
            "msg_count": len(day_msgs),
            "keywords": _keywords(text),
            "people": people[:8],
            "signals": {
                "questions": questions,
                "exclamations": exclaims,
                "mood_scores": mood_scores,
            },
            "start_msg_id": day_msgs[0]["msg_id"],
            "end_msg_id": day_msgs[-1]["msg_id"],
        })

    drifts = []
    for prev, curr in zip(timeline, timeline[1:]):
        changed = prev["mood"] != curr["mood"] or prev["tone"] != curr["tone"]
        prev_vec = Counter(prev["keywords"])
        curr_vec = Counter(curr["keywords"])
        overlap = len(set(prev_vec) & set(curr_vec))
        novelty = 1 - overlap / max(1, len(set(prev_vec) | set(curr_vec)))
        if changed or novelty >= 0.75:
            drifts.append({
                "from_day": prev["day"],
                "to_day": curr["day"],
                "from": prev["label"],
                "to": curr["label"],
                "trigger": _trigger_for(prev, curr),
                "novelty": round(novelty, 3),
            })

    baseline = {}
    if persona:
        baseline = {
            "persona_tone": persona.get("communication_style", {}).get("tone"),
            "persona_traits": persona.get("personality_traits", [])[:5],
        }

    return {
        "baseline": baseline,
        "timeline": timeline,
        "drifts": drifts,
        "summary": [item["message"] for item in timeline],
    }


def save_persona_drift(messages: list[dict], persona: dict | None, out_path: str | Path) -> dict:
    result = detect_persona_drift(messages, persona)
    Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
