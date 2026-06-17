"""
summarizer.py — Groq FREE API with rate limit handling.
"""

import os, json, time, requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"


def _get_key():
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "GROQ_API_KEY not set.\n"
            "1. Sign up free at https://console.groq.com\n"
            "2. Create an API key\n"
            "3. Run: $env:GROQ_API_KEY = 'gsk_...'\n"
        )
    return key


def _call_groq(system_prompt: str, user_prompt: str, max_tokens: int = 300) -> str:
    headers = {
        "Authorization": f"Bearer {_get_key()}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    # Retry up to 5 times on rate limit
    for attempt in range(5):
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 429:
            wait = 20 * (attempt + 1)  # 20s, 40s, 60s ...
            print(f"  Rate limited. Waiting {wait}s before retry {attempt+1}/5 ...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    raise Exception("Groq rate limit exceeded after 5 retries. Wait a minute and rerun.")


TOPIC_SYSTEM = (
    "Summarize the key topics and important facts from this conversation segment "
    "in 3–5 concise sentences. Mention specific details like names, places, events, habits."
)

def summarize_topic(segment: dict) -> str:
    msgs = segment["messages"]
    # Only use first 40 messages to reduce token usage
    convo_text = "\n".join(f"{m['speaker']}: {m['text']}" for m in msgs[:40])
    prompt = f"Conversation segment (Topic {segment['topic_id']}):\n\n{convo_text}"
    result = _call_groq(TOPIC_SYSTEM, prompt, max_tokens=200)
    time.sleep(2)  # 2s pause between calls to avoid rate limits
    return result


CHECKPOINT_SYSTEM = (
    "Summarize the following conversation messages into 3–5 concise sentences. "
    "Capture the main themes, facts shared, and tone."
)

def summarize_checkpoint(messages: list[dict], checkpoint_num: int) -> str:
    # Only use every 3rd message to reduce tokens (still representative)
    sampled = messages[::3]
    convo_text = "\n".join(f"{m['speaker']}: {m['text']}" for m in sampled)
    prompt = f"Checkpoint #{checkpoint_num}:\n\n{convo_text}"
    result = _call_groq(CHECKPOINT_SYSTEM, prompt, max_tokens=200)
    time.sleep(2)
    return result


ANSWER_SYSTEM = (
    "You are a helpful assistant with access to conversation summaries and "
    "raw message excerpts. Answer the user's question using only the provided "
    "context. Be concise and cite relevant details."
)

def generate_answer(query: str, topic_summaries: list[str],
                    message_chunks: list[str], persona: dict) -> str:
    context_parts = []
    if topic_summaries:
        context_parts.append("TOPIC SUMMARIES:\n" + "\n---\n".join(topic_summaries))
    if message_chunks:
        context_parts.append("RELEVANT MESSAGES:\n" + "\n".join(message_chunks))
    if persona:
        context_parts.append("USER PERSONA:\n" + json.dumps(persona, indent=2))

    context = "\n\n".join(context_parts) or "No relevant context found."
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    return _call_groq(ANSWER_SYSTEM, prompt, max_tokens=400)
