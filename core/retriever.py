"""
retriever.py — FAISS vector store using TF-IDF embeddings (no external model download).

Two separate FAISS IndexFlatIP indices:
    topic_index   — one vector per topic summary
    message_index — one vector per message
"""

import json
import pickle
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from pathlib import Path


class RAGRetriever:
    def __init__(self, model=None):  # model param kept for API compat
        self.vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.topic_meta: list[dict] = []
        self.topic_index: faiss.Index | None = None
        self.msg_meta: list[dict] = []
        self.msg_index: faiss.Index | None = None

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> np.ndarray:
        sparse_vecs = self.vectorizer.transform(texts)
        if self.svd is not None:
            vecs = self.svd.transform(sparse_vecs).astype("float32")
        else:
            vecs = sparse_vecs.toarray().astype("float32")
        return normalize(vecs).astype("float32")

    # ── Build ────────────────────────────────────────────────────────────────

    def build_topic_index(self, segments_with_summaries: list[dict]):
        self.topic_meta = segments_with_summaries
        texts = [s["summary"] for s in segments_with_summaries]
        if not hasattr(self, '_fitted') or not self._fitted:
            # Fit vectorizer on topic summaries (will be re-fitted properly in build_message_index)
            pass  # defer fitting to after message index build
        self._pending_topic_texts = texts

    def build_message_index(self, messages: list[dict]):
        self.msg_meta = messages
        msg_texts = [m["text"] for m in messages]

        # Fit TF-IDF on ALL text (topics + messages) for consistent vocab
        topic_texts = getattr(self, '_pending_topic_texts', [])
        all_texts = topic_texts + msg_texts

        print(f"  Fitting TF-IDF on {len(all_texts)} texts …")
        self.vectorizer = TfidfVectorizer(
            max_features=12000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        sparse_all = self.vectorizer.fit_transform(all_texts)
        n_components = min(128, max(2, sparse_all.shape[1] - 1), max(2, sparse_all.shape[0] - 1))
        print(f"  Fitting SVD projection to {n_components} dims ...")
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(sparse_all)

        # Build topic index
        if topic_texts:
            t_vecs = self._embed(topic_texts)
            dim = t_vecs.shape[1]
            self.topic_index = faiss.IndexFlatIP(dim)
            self.topic_index.add(t_vecs)
            print(f"  Topic index: {self.topic_index.ntotal} vectors, dim={dim}")

        # Build message index
        print(f"  Encoding {len(msg_texts)} messages …")
        m_vecs = self._embed(msg_texts)
        dim = m_vecs.shape[1]
        self.msg_index = faiss.IndexFlatIP(dim)
        self.msg_index.add(m_vecs)
        print(f"  Message index: {self.msg_index.ntotal} vectors, dim={dim}")

    # ── Query ────────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k_topics: int = 3,
                 top_k_msgs: int = 5) -> dict:
        q_vec = self._embed([query])
        results = {"topic_results": [], "message_results": []}

        if self.topic_index and self.topic_index.ntotal > 0:
            k = min(top_k_topics, self.topic_index.ntotal)
            scores, indices = self.topic_index.search(q_vec, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1: continue
                meta = self.topic_meta[idx]
                results["topic_results"].append({
                    "topic_id": meta["topic_id"],
                    "summary":  meta["summary"],
                    "score":    float(score),
                })

        if self.msg_index and self.msg_index.ntotal > 0:
            k = min(top_k_msgs, self.msg_index.ntotal)
            scores, indices = self.msg_index.search(q_vec, k)
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1: continue
                m = self.msg_meta[idx]
                results["message_results"].append({
                    "msg_id":  m["msg_id"],
                    "speaker": m["speaker"],
                    "text":    m["text"],
                    "score":   float(score),
                })

        return results

    # ── Persist ──────────────────────────────────────────────────────────────

    def save(self, out_dir: str):
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)

        if self.topic_index:
            faiss.write_index(self.topic_index, str(p / "topic.index"))
            (p / "topic_meta.json").write_text(json.dumps(self.topic_meta, indent=2))

        if self.msg_index:
            faiss.write_index(self.msg_index, str(p / "messages.index"))
            (p / "msg_meta.json").write_text(json.dumps(self.msg_meta, indent=2))

        with open(p / "vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(p / "svd.pkl", "wb") as f:
            pickle.dump(self.svd, f)

        print(f"  Saved indices + vectorizer to {out_dir}")

    def load(self, index_dir: str, model=None):
        p = Path(index_dir)

        vpath = p / "vectorizer.pkl"
        if vpath.exists():
            with open(vpath, "rb") as f:
                self.vectorizer = pickle.load(f)
        spath = p / "svd.pkl"
        if spath.exists():
            with open(spath, "rb") as f:
                self.svd = pickle.load(f)

        tpath = p / "topic.index"
        if tpath.exists():
            self.topic_index = faiss.read_index(str(tpath))
            self.topic_meta  = json.loads((p / "topic_meta.json").read_text())

        mpath = p / "messages.index"
        if mpath.exists():
            self.msg_index = faiss.read_index(str(mpath))
            self.msg_meta  = json.loads((p / "msg_meta.json").read_text())

        print(f"  Loaded indices from {index_dir}")
