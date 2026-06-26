"""
Offline intent classifier for user messages.

The model is deliberately small: a TF-IDF vectorizer plus LogisticRegression.
It trains from lightweight seed examples and optional conversation-derived
keyword labels, runs on CPU, and avoids all external API calls.
"""

from __future__ import annotations

import json
import pickle
import re
import time
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


INTENTS = ["reminder", "emotional-support", "action-item", "small-talk", "unknown"]

SEED_EXAMPLES = [
    ("remind me to call mom tomorrow", "reminder"),
    ("please remind me about the meeting at 5", "reminder"),
    ("don't let me forget my assignment deadline", "reminder"),
    ("set a reminder for rent payment", "reminder"),
    ("i feel terrible and need someone to listen", "emotional-support"),
    ("i am sad and overwhelmed", "emotional-support"),
    ("can you comfort me", "emotional-support"),
    ("i'm anxious about everything", "emotional-support"),
    ("i need to finish the report today", "action-item"),
    ("todo buy groceries and send the email", "action-item"),
    ("we should book tickets tonight", "action-item"),
    ("help me plan next steps", "action-item"),
    ("hey how are you", "small-talk"),
    ("good morning", "small-talk"),
    ("lol that's funny", "small-talk"),
    ("what's up", "small-talk"),
    ("blue window seven paper", "unknown"),
    ("asdf random unrelated text", "unknown"),
    ("i saw a table near the station", "unknown"),
    ("nothing specific", "unknown"),
]

KEYWORD_RULES = {
    "reminder": re.compile(r"\b(remind|reminder|forget|alarm|deadline|due|tomorrow|tonight)\b", re.I),
    "emotional-support": re.compile(r"\b(sad|anxious|lonely|hurt|cry|upset|depressed|overwhelmed|comfort|support)\b", re.I),
    "action-item": re.compile(r"\b(todo|need to|have to|should|plan|finish|submit|send|call|book|buy|schedule)\b", re.I),
    "small-talk": re.compile(r"\b(hi|hey|hello|good morning|good night|how are you|what'?s up|lol|haha)\b", re.I),
}


def weak_label(text: str) -> str | None:
    matches = [label for label, pattern in KEYWORD_RULES.items() if pattern.search(text)]
    return matches[0] if len(matches) == 1 else None


def build_training_rows(messages: list[dict] | None = None) -> list[tuple[str, str]]:
    rows = list(SEED_EXAMPLES)
    for m in messages or []:
        if "1" not in str(m.get("speaker", "")):
            continue
        text = str(m.get("text", "")).strip()
        if not text:
            continue
        label = weak_label(text)
        if label:
            rows.append((text, label))
    return rows


def train_intent_classifier(messages: list[dict] | None = None) -> Pipeline:
    rows = build_training_rows(messages)
    texts = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=4000, ngram_range=(1, 2), lowercase=True)),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    clf.fit(texts, labels)
    return clf


def save_intent_classifier(model: Pipeline, out_path: str | Path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_intent_classifier(path: str | Path) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class OfflineIntentClassifier:
    def __init__(self, model_path: str | Path):
        self.model = load_intent_classifier(model_path)

    def classify(self, message: str) -> dict:
        start = time.perf_counter()
        probs = self.model.predict_proba([message])[0]
        classes = list(self.model.classes_)
        best_idx = int(probs.argmax())
        label = str(classes[best_idx])
        confidence = float(probs[best_idx])
        if confidence < 0.34:
            label = "unknown"
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "intent": label,
            "confidence": round(confidence, 4),
            "latency_ms": round(elapsed_ms, 3),
            "offline": True,
        }


def benchmark(model_path: str | Path, samples: list[str] | None = None) -> dict:
    clf = OfflineIntentClassifier(model_path)
    samples = samples or [text for text, _ in SEED_EXAMPLES]
    timings = [clf.classify(text)["latency_ms"] for text in samples]
    return {
        "messages": len(samples),
        "avg_latency_ms": round(sum(timings) / max(1, len(timings)), 3),
        "max_latency_ms": round(max(timings or [0]), 3),
        "under_200ms": max(timings or [0]) < 200,
    }


def write_report(model_path: str | Path, out_path: str | Path) -> None:
    Path(out_path).write_text(json.dumps(benchmark(model_path), indent=2), encoding="utf-8")
