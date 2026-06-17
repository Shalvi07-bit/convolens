"""
topic_detector.py — Detect topic changes using TF-IDF + cosine similarity.

Strategy:
- Group messages by conversation (conv_id) first.
- Compare conversation-level centroids using cosine similarity.
- Merge consecutive similar conversations into the same topic.
- Result: topics that align with natural conversation breaks.

No external model downloads required — pure scikit-learn.
"""

import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

# A new topic starts when similarity between adjacent conversation centroids
# drops below this value (empirically tuned for this dataset).
THRESHOLD = 0.10


def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def detect_topics(messages: list[dict], model=None) -> list[dict]:
    """
    Returns list of topic segment dicts:
    [{"topic_id", "start_idx", "end_idx", "messages"}, ...]
    """
    if not messages:
        return []

    # Group by conversation
    conv_groups: dict[int, list[dict]] = defaultdict(list)
    for m in messages:
        conv_groups[m["conv_id"]].append(m)
    ordered_conv_ids = sorted(conv_groups.keys())

    print(f"  Vectorising {len(messages)} messages across "
          f"{len(ordered_conv_ids)} conversations …")

    # Fit TF-IDF on all messages
    all_texts = [m["text"] for m in messages]
    vectorizer = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(all_texts).toarray()

    # Map msg_id → row index in tfidf matrix
    msg_id_to_row = {m["msg_id"]: i for i, m in enumerate(messages)}

    # Compute per-conversation centroid
    conv_centroids = {}
    for cid in ordered_conv_ids:
        rows = [msg_id_to_row[m["msg_id"]] for m in conv_groups[cid]]
        conv_centroids[cid] = tfidf[rows].mean(axis=0)

    # Detect topic boundaries between consecutive conversations
    topic_id = 0
    segments = []  # list of list[conv_id]
    current_group = [ordered_conv_ids[0]]

    for i in range(1, len(ordered_conv_ids)):
        prev_cid = ordered_conv_ids[i - 1]
        curr_cid = ordered_conv_ids[i]
        sim = cosine(conv_centroids[prev_cid], conv_centroids[curr_cid])
        if sim < THRESHOLD:
            # New topic
            segments.append(current_group)
            current_group = [curr_cid]
        else:
            current_group.append(curr_cid)

    segments.append(current_group)  # last group

    # Build output format
    result = []
    for topic_id, conv_id_list in enumerate(segments):
        seg_msgs = []
        for cid in conv_id_list:
            seg_msgs.extend(conv_groups[cid])
        # Already in order since conv_groups preserves insertion order from sorted conv_ids
        result.append({
            "topic_id":  topic_id,
            "start_idx": seg_msgs[0]["msg_id"],
            "end_idx":   seg_msgs[-1]["msg_id"],
            "messages":  seg_msgs,
        })

    return result
