"""
persona_extractor.py — Extract structured user persona using Groq's FREE API.
"""

import os
import json
import random
import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"

PERSONA_SYSTEM = """
You are a behavioural analyst. Analyse the following conversation excerpts and extract a structured persona for "User 1" ONLY.

Return ONLY valid JSON (no markdown fences, no explanation) in this exact schema:
{
  "habits": ["list of observed habits, e.g. late sleeper, eats junk food"],
  "personal_facts": ["list of factual details: job, location, relationships, life events"],
  "personality_traits": ["list of personality descriptors backed by evidence"],
  "communication_style": {
    "tone": "friendly / serious / sarcastic / etc.",
    "message_length": "short / medium / long",
    "emoji_usage": "none / occasional / frequent",
    "notable_patterns": ["any recurring phrases or styles"]
  },
  "interests": ["topics User 1 brings up often or seems enthusiastic about"]
}

Only include things directly inferable from the text. Do not guess.
""".strip()


def extract_persona(messages: list[dict], sample_size: int = 500) -> dict:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise EnvironmentError("GROQ_API_KEY not set. Get a free key at https://console.groq.com")

    user1_msgs = [m for m in messages if "1" in m["speaker"]]
    sample = random.sample(user1_msgs, min(sample_size, len(user1_msgs)))
    sample.sort(key=lambda m: m["msg_id"])

    convo_text = "\n".join(f"[msg {m['msg_id']}] {m['text']}" for m in sample)
    prompt = f"Conversation excerpts from User 1:\n\n{convo_text}"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 1000,
        "messages": [
            {"role": "system", "content": PERSONA_SYSTEM},
            {"role": "user",   "content": prompt},
        ],
    }
    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw, "parse_error": True}
