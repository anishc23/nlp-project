# -*- coding: utf-8 -*-
"""
Sentiment analysis (module F).

`l3cube-pune/marathi-sentiment-md` is a 3-way Negative/Neutral/Positive model
trained on tweet-length Marathi text, so it is applied per sentence rather
than to a whole article; document tone is then the length-weighted mean of the
sentence polarities. This also gives the per-entity tone used by the
dashboard: the sentiment of an entity is the mean polarity of the sentences it
appears in.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                      # noqa: E402
from src import models             # noqa: E402
from src.morph import sent_tokenize  # noqa: E402

# Maps the model's classes onto a scalar polarity in [-1, 1].
POLARITY = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}


@dataclass
class SentenceSentiment:
    text: str
    label: str
    score: float
    polarity: float


@dataclass
class DocSentiment:
    label: str
    polarity: float
    sentences: List[SentenceSentiment] = field(default_factory=list)

    @property
    def distribution(self) -> Dict[str, int]:
        d = {"Negative": 0, "Neutral": 0, "Positive": 0}
        for s in self.sentences:
            d[s.label] = d.get(s.label, 0) + 1
        return d


def _label_from_polarity(p: float) -> str:
    if p > 0.15:
        return "Positive"
    if p < -0.15:
        return "Negative"
    return "Neutral"


@torch.no_grad()
def classify_sentences(sentences: List[str],
                       batch_size: int = None) -> List[SentenceSentiment]:
    """Polarity of each sentence."""
    if not sentences:
        return []
    tok, mod = models.load_classifier(config.MODEL_SENTIMENT)
    id2label = mod.config.id2label
    bs = batch_size or config.BATCH_SIZE
    out: List[SentenceSentiment] = []

    for i in range(0, len(sentences), bs):
        chunk = [s if s else " " for s in sentences[i:i + bs]]
        enc = tok(chunk, padding=True, truncation=True,
                  max_length=config.MAX_LEN, return_tensors="pt").to(config.DEVICE)
        probs = mod(**enc).logits.float().softmax(-1).cpu()
        for text, row in zip(chunk, probs):
            idx = int(row.argmax())
            label = id2label[idx]
            # Expected polarity rather than the argmax class: keeps a
            # 0.51/0.49 Negative/Neutral sentence from counting as strongly
            # negative when averaged over a document.
            pol = sum(POLARITY[id2label[j]] * float(row[j]) for j in range(len(row)))
            out.append(SentenceSentiment(text=text, label=label,
                                         score=float(row[idx]), polarity=pol))
    return out


def analyse_document(text: str) -> DocSentiment:
    """Length-weighted document tone built from sentence-level polarity."""
    sents = sent_tokenize(text)
    if not sents:
        return DocSentiment(label="Neutral", polarity=0.0)
    scored = classify_sentences(sents)
    weights = [max(len(s.text), 1) for s in scored]
    total = sum(weights)
    pol = sum(s.polarity * w for s, w in zip(scored, weights)) / total
    return DocSentiment(label=_label_from_polarity(pol), polarity=pol,
                        sentences=scored)


def entity_sentiment(doc: DocSentiment, entities) -> Dict[str, float]:
    """
    Mean polarity of the sentences each entity is mentioned in.

    This is what answers "how is politician X being covered this month" once
    aggregated across the corpus.
    """
    scores: Dict[str, List[float]] = defaultdict(list)
    for sent in doc.sentences:
        for ent in entities:
            if ent.text and ent.text in sent.text:
                scores[ent.text].append(sent.polarity)
    return {k: sum(v) / len(v) for k, v in scores.items() if v}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    demo = ("भारतीय संघाने शानदार विजय मिळवला आणि चाहत्यांनी जल्लोष केला. "
            "मात्र दुखापतीमुळे कर्णधार पुढील सामन्याला मुकणार आहे. "
            "सामना मुंबईत खेळवण्यात आला.")
    d = analyse_document(demo)
    print(f"  document: {d.label}  polarity={d.polarity:+.3f}  {d.distribution}")
    for s in d.sentences:
        print(f"    [{s.label:8s} {s.polarity:+.2f}] {s.text[:60]}")
