# -*- coding: utf-8 -*-
"""
Information retrieval (module C).

Three retrievers over the same corpus:

  * BM25    -- Okapi BM25 over morphologically normalised tokens. Implemented
               here rather than pulled from rank_bm25 so that the tokenisation
               path (and therefore the morphology ablation) is under our
               control and the index can be pickled with its own vocabulary.
  * Dense   -- cosine similarity over MahaBERT-SBERT document embeddings.
  * Hybrid  -- Reciprocal Rank Fusion of the two. RRF is used instead of a
               weighted score sum because BM25 scores and cosine similarities
               live on different, corpus-dependent scales.

The `use_morph` flag on the BM25 index is the knob for the central experiment:
building the same index with and without the analyser and re-running the same
queries is what produces the retrieval numbers in the report.
"""
from __future__ import annotations

import math
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                      # noqa: E402
from src.morph import normalize_tokens  # noqa: E402

K1 = 1.5
B = 0.75
RRF_K = 60


@dataclass
class Hit:
    doc_id: int
    score: float
    rank: int
    source: str = "hybrid"


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------
@dataclass
class BM25Index:
    use_morph: bool = True
    postings: Dict[str, List[tuple]] = field(default_factory=lambda: defaultdict(list))
    doc_len: List[int] = field(default_factory=list)
    doc_ids: List[int] = field(default_factory=list)
    avg_len: float = 0.0

    def build(self, docs: List[str], doc_ids: Optional[List[int]] = None):
        self.postings = defaultdict(list)
        self.doc_len = []
        self.doc_ids = doc_ids if doc_ids is not None else list(range(len(docs)))

        for i, text in enumerate(docs):
            toks = normalize_tokens(text, use_morph=self.use_morph)
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                self.postings[term].append((i, tf))

        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.postings = dict(self.postings)
        return self

    def search(self, query: str, top_k: int = 10) -> List[Hit]:
        n = len(self.doc_len)
        if n == 0:
            return []
        q_terms = normalize_tokens(query, use_morph=self.use_morph)
        scores = np.zeros(n, dtype="float32")

        for term in q_terms:
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            # BM25 idf with the +0.5 smoothing; clamped at 0 so that terms
            # appearing in more than half the corpus cannot subtract score.
            idf = max(math.log((n - df + 0.5) / (df + 0.5) + 1.0), 0.0)
            for doc_i, tf in posting:
                denom = tf + K1 * (1 - B + B * self.doc_len[doc_i] / max(self.avg_len, 1e-9))
                scores[doc_i] += idf * (tf * (K1 + 1)) / max(denom, 1e-9)

        top = np.argsort(-scores)[:top_k]
        return [Hit(doc_id=self.doc_ids[i], score=float(scores[i]), rank=r, source="bm25")
                for r, i in enumerate(top) if scores[i] > 0]

    def save(self, path: Path):
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "BM25Index":
        return pickle.loads(Path(path).read_bytes())


# --------------------------------------------------------------------------
# Dense
# --------------------------------------------------------------------------
@dataclass
class DenseIndex:
    vectors: np.ndarray = None
    doc_ids: List[int] = field(default_factory=list)

    def build(self, docs: List[str], doc_ids: Optional[List[int]] = None,
              show_progress: bool = True):
        from src import models
        self.vectors = models.embed(docs, show_progress=show_progress)
        self.doc_ids = doc_ids if doc_ids is not None else list(range(len(docs)))
        return self

    def search(self, query: str, top_k: int = 10) -> List[Hit]:
        from src import models
        if self.vectors is None or len(self.vectors) == 0:
            return []
        q = models.embed([query])[0]
        sims = self.vectors @ q            # both are L2-normalised
        top = np.argsort(-sims)[:top_k]
        return [Hit(doc_id=self.doc_ids[i], score=float(sims[i]), rank=r, source="dense")
                for r, i in enumerate(top)]

    def save(self, path: Path):
        np.savez_compressed(path, vectors=self.vectors,
                            doc_ids=np.array(self.doc_ids))

    @staticmethod
    def load(path: Path) -> "DenseIndex":
        z = np.load(path)
        return DenseIndex(vectors=z["vectors"], doc_ids=z["doc_ids"].tolist())


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------
def reciprocal_rank_fusion(runs: List[List[Hit]], top_k: int = 10) -> List[Hit]:
    """Combine ranked lists by summing 1/(k + rank)."""
    fused: Dict[int, float] = defaultdict(float)
    for run in runs:
        for hit in run:
            fused[hit.doc_id] += 1.0 / (RRF_K + hit.rank + 1)
    ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
    return [Hit(doc_id=d, score=s, rank=r, source="hybrid")
            for r, (d, s) in enumerate(ordered)]


class HybridRetriever:
    """BM25 + dense with RRF, with either half optional."""

    def __init__(self, bm25: BM25Index = None, dense: DenseIndex = None):
        self.bm25 = bm25
        self.dense = dense

    def search(self, query: str, top_k: int = 10,
               mode: str = "hybrid") -> List[Hit]:
        if mode == "bm25":
            return self.bm25.search(query, top_k)
        if mode == "dense":
            return self.dense.search(query, top_k)
        runs = []
        if self.bm25 is not None:
            runs.append(self.bm25.search(query, top_k * 3))
        if self.dense is not None:
            runs.append(self.dense.search(query, top_k * 3))
        if not runs:
            return []
        return reciprocal_rank_fusion(runs, top_k)

    @staticmethod
    def load_default() -> "HybridRetriever":
        bm25 = BM25Index.load(config.BM25_INDEX) if config.BM25_INDEX.exists() else None
        dense = DenseIndex.load(config.DENSE_INDEX) if config.DENSE_INDEX.exists() else None
        return HybridRetriever(bm25=bm25, dense=dense)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    docs = [
        "पुण्यातील शाळांमध्ये नवीन शैक्षणिक धोरण लागू करण्यात आले आहे.",
        "भारतीय क्रिकेट संघाने कसोटी सामन्यात विजय मिळवला.",
        "मुंबईत मेट्रो प्रकल्पाचे काम वेगाने सुरू आहे.",
        "शाळेत विद्यार्थ्यांसाठी नवीन अभ्यासक्रम सुरू झाला.",
    ]
    # The query says "शाळा"; documents say "शाळांमध्ये" and "शाळेत".
    query = "शाळा"
    for use_morph in (False, True):
        idx = BM25Index(use_morph=use_morph).build(docs)
        hits = idx.search(query, top_k=3)
        tag = "morph" if use_morph else "plain"
        print(f"  [{tag}] query={query!r} -> {len(hits)} hits")
        for h in hits:
            print(f"      {h.score:.3f}  {docs[h.doc_id][:52]}")
