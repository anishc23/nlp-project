# -*- coding: utf-8 -*-
"""
Extractive summarisation (module E).

Two scorers, both of which rank sentences and then de-duplicate with MMR:

  * `textrank`  -- PageRank over a sentence-similarity graph built from
                   morphologically normalised TF-IDF vectors. No neural model,
                   so it runs on CPU in milliseconds.
  * `embedding` -- the same graph but with MahaBERT-SBERT sentence vectors,
                   which catches paraphrase that shares no surface tokens.

A `lead` baseline (first k sentences) is included because for news it is a
genuinely strong baseline, and the evaluation script reports all three so the
comparison is honest rather than flattering.

Morphology matters here: without it, "सरकारने" and "सरकारच्या" contribute to
different TF-IDF dimensions and two sentences about the same subject look
unrelated.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.morph import normalize_tokens, sent_tokenize  # noqa: E402

DAMPING = 0.85
MAX_ITER = 100
TOL = 1e-6


@dataclass
class Summary:
    sentences: List[str]
    indices: List[int]
    method: str

    @property
    def text(self) -> str:
        return " ".join(self.sentences)


# --------------------------------------------------------------------------
# Similarity graphs
# --------------------------------------------------------------------------
def _tfidf_matrix(sentences: List[str]) -> np.ndarray:
    """Morphologically normalised TF-IDF, built without sklearn's tokenizer."""
    docs = [normalize_tokens(s) for s in sentences]
    vocab = {}
    for d in docs:
        for t in d:
            vocab.setdefault(t, len(vocab))
    if not vocab:
        return np.zeros((len(sentences), 1), dtype="float32")

    tf = np.zeros((len(docs), len(vocab)), dtype="float32")
    for i, d in enumerate(docs):
        for t in d:
            tf[i, vocab[t]] += 1.0
    tf /= np.maximum(tf.sum(1, keepdims=True), 1e-9)

    df = (tf > 0).sum(0)
    idf = np.log((1 + len(docs)) / (1 + df)) + 1.0
    return tf * idf


def _cosine_graph(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.maximum(norms, 1e-9)
    sim = unit @ unit.T
    np.fill_diagonal(sim, 0.0)
    return np.clip(sim, 0.0, None)


def _pagerank(sim: np.ndarray) -> np.ndarray:
    """Power iteration; falls back to uniform scores on a disconnected graph."""
    n = sim.shape[0]
    if n == 0:
        return np.zeros(0)
    row = sim.sum(1, keepdims=True)
    # Dangling rows get a uniform transition so probability mass is conserved.
    transition = np.where(row > 0, sim / np.maximum(row, 1e-9), 1.0 / n)
    scores = np.full(n, 1.0 / n)
    for _ in range(MAX_ITER):
        nxt = (1 - DAMPING) / n + DAMPING * (transition.T @ scores)
        if np.abs(nxt - scores).sum() < TOL:
            return nxt
        scores = nxt
    return scores


def _mmr(order: List[int], sim: np.ndarray, k: int,
         lambda_: float = 0.7) -> List[int]:
    """
    Greedily pick k sentences trading relevance against redundancy.

    Relevance is fixed up front from each sentence's PageRank position, not
    recomputed against the shrinking candidate list -- otherwise a sentence's
    relevance rises simply because better ones were already taken, and the
    redundancy penalty stops being comparable across iterations.
    """
    n = max(len(order), 1)
    relevance = {idx: 1.0 - pos / n for pos, idx in enumerate(order)}

    selected: List[int] = []
    candidates = list(order)
    while candidates and len(selected) < k:
        best, best_score = None, -np.inf
        for idx in candidates:
            redundancy = max((sim[idx, j] for j in selected), default=0.0)
            score = lambda_ * relevance[idx] - (1 - lambda_) * redundancy
            if score > best_score:
                best, best_score = idx, score
        selected.append(best)
        candidates.remove(best)
    return selected


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def summarize(text: str, k: int = 3, method: str = "textrank",
              lambda_: float = 0.7) -> Summary:
    """Return the k most representative sentences, in original order."""
    sentences = [s for s in sent_tokenize(text) if len(s.split()) >= 3]
    if not sentences:
        return Summary(sentences=[], indices=[], method=method)
    if len(sentences) <= k:
        return Summary(sentences=sentences, indices=list(range(len(sentences))),
                       method=method)

    if method == "lead":
        idx = list(range(k))
        return Summary(sentences=[sentences[i] for i in idx], indices=idx,
                       method=method)

    if method == "embedding":
        from src import models
        vectors = models.embed(sentences)
    else:
        vectors = _tfidf_matrix(sentences)

    sim = _cosine_graph(vectors)
    ranked = list(np.argsort(-_pagerank(sim)))
    chosen = sorted(_mmr(ranked, sim, k, lambda_))
    return Summary(sentences=[sentences[i] for i in chosen], indices=chosen,
                   method=method)


def summarize_many(texts: List[str], k: int = 3,
                   method: str = "textrank") -> List[Summary]:
    return [summarize(t, k=k, method=method) for t in texts]


if __name__ == "__main__":
    import json
    sys.stdout.reconfigure(encoding="utf-8")
    import config
    row = json.loads(open(config.XLSUM_TEST, encoding="utf-8").readline())
    print("TITLE  :", row["title"])
    print("GOLD   :", row["summary"][:220])
    for m in ("lead", "textrank"):
        s = summarize(row["text"], k=2, method=m)
        print(f"\n{m.upper():8s}:", s.text[:300])
