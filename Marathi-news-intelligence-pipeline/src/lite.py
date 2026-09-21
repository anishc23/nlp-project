# -*- coding: utf-8 -*-
"""
Lightweight students of the MahaBERT models, small enough to run in a browser.

The demo page is a single HTML file, so it cannot load the 12-class topic
model, MahaNER or the sentiment model (each is a ~700 MB BERT). Instead it
carries three small linear models trained by scripts/train_lite.py to
reproduce what those models say on XL-Sum:

  * topic      -- softmax regression over sublinear TF-IDF of analyser roots
  * sentiment  -- softmax regression over words, roots and word bigrams
  * entities   -- greedy left-to-right word tagger (a maximum-entropy Markov
                  model) over word, root, suffix, neighbour and previous-label
                  features

How often each agrees with the model it imitates is measured on the held-out
XL-Sum test split and shipped with the weights, so the page can say how far
to trust it.

This module is the reference implementation: the page's JavaScript is a line
by line port of the feature functions here, and scripts/verify_demo.py checks
that the two produce identical outputs.
"""
from __future__ import annotations

import base64
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                                        # noqa: E402
from src.morph import (STOPWORDS, normalize_tokens, sent_tokenize,  # noqa: E402
                       stem, tokenize)

LITE_MODELS = config.ROOT / "app" / "lite_models.json"

NER_LABELS = ["O", "Person", "Location", "Organization", "Date", "Time",
              "Measure", "Designation"]
POLARITY = {"Negative": -1.0, "Neutral": 0.0, "Positive": 1.0}

# Same boundary set as src/ner.py: a "word" for the tagger is a maximal run
# of characters not in this set, which is exactly what MahaNER spans snap to.
BOUNDARY = set(" \t\n\r " + "।,.;:!?\"'()[]{}<>%/\\|-–—")

# How much of an article the topic student reads. The teacher truncates at
# 256 wordpieces, so reading the whole article would teach the student to use
# evidence the teacher never saw.
TOPIC_MAX_TOKENS = 200


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def topic_terms(title: str, text: str) -> List[str]:
    joined = f"{title}. {text}" if title else text
    return normalize_tokens(joined)[:TOPIC_MAX_TOKENS]


def sentiment_features(sentence: str) -> List[str]:
    toks = tokenize(sentence)
    feats = []
    for t in toks:
        feats.append("w:" + t)
        r = stem(t)
        if r != t:
            feats.append("r:" + r)
    for a, b in zip(toks, toks[1:]):
        feats.append("b:" + a + "_" + b)
    return feats


def words(text: str) -> List[Tuple[int, int]]:
    """(start, end) offsets of every word, splitting on BOUNDARY characters."""
    out, start = [], None
    for i, ch in enumerate(text):
        if ch in BOUNDARY:
            if start is not None:
                out.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        out.append((start, len(text)))
    return out


_DIGITS_RE = re.compile(r"[0-9\u0966-\u096F]+")
_ASCII_RE = re.compile(r"[\x00-\x7F]+")


def _shape(w: str) -> str:
    # Explicit character classes rather than str.isdigit()/isascii(), whose
    # Unicode rules differ from JavaScript's and would break parity.
    if _DIGITS_RE.fullmatch(w):
        return "digit"
    if _ASCII_RE.fullmatch(w):
        return "latin"
    return "deva"


def ner_features(toks: List[str], i: int, prev: str, prev2: str) -> List[str]:
    """Features for word i; prev/prev2 are the labels already assigned."""
    w = toks[i]
    r = stem(w)
    p = toks[i - 1] if i > 0 else "<s>"
    n = toks[i + 1] if i + 1 < len(toks) else "</s>"
    n2 = toks[i + 2] if i + 2 < len(toks) else "</s>"
    feats = [
        "bias", "w:" + w, "r:" + r, "s1:" + w[-1:], "s2:" + w[-2:], "s3:" + w[-3:],
        "p3:" + w[:3], "sh:" + _shape(w), "sw:" + str(w in STOPWORDS),
        "pw:" + p, "pr:" + stem(p), "nw:" + n, "nr:" + stem(n), "nw2:" + n2,
        "pl:" + prev, "pl2:" + prev + "|" + prev2, "plw:" + prev + "|" + r,
        "pwn:" + p + "|" + w, "wn:" + w + "|" + n,
    ]
    if i == 0:
        feats.append("first")
    return feats


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
def _softmax(z: List[float]) -> List[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def unpack(b64: str, cols: int) -> List[List[int]]:
    """Rows of int8 weights from the base64 blob written by train_lite.py."""
    raw = base64.b64decode(b64)
    vals = [v - 256 if v > 127 else v for v in raw]
    return [vals[i:i + cols] for i in range(0, len(vals), cols)]


class Lite:
    """Loaded student models. Weights are int8 values times one scale."""

    def __init__(self, data: dict):
        self.data = data
        t = data["topic"]
        self.topic_labels = t["labels"]
        self.topic_idf = dict(zip(t["terms"], t["idf"]))
        self.topic_w = dict(zip(t["terms"], unpack(t["W"], len(t["labels"]))))
        self.topic_b = t["b"]
        self.topic_scale = t["scale"]

        s = data["sentiment"]
        self.sent_labels = s["labels"]
        self.sent_w = dict(zip(s["feats"], unpack(s["W"], len(s["labels"]))))
        self.sent_b = s["b"]
        self.sent_scale = s["scale"]

        n = data["ner"]
        self.ner_labels = n["labels"]
        self.ner_w = dict(zip(n["feats"], unpack(n["W"], len(n["labels"]))))
        self.ner_b = n["b"]
        self.ner_scale = n["scale"]

    @classmethod
    def load(cls, path: Path = LITE_MODELS) -> "Lite":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # ---- topic ----------------------------------------------------------
    def topic(self, title: str, text: str) -> Dict[str, float]:
        tf = Counter(t for t in topic_terms(title, text) if t in self.topic_w)
        vec = {t: (1 + math.log(c)) * self.topic_idf[t] for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        k = len(self.topic_labels)
        z = [0.0] * k
        for t, v in vec.items():
            row = self.topic_w[t]
            for j in range(k):
                z[j] += row[j] * v / norm
        z = [z[j] * self.topic_scale + self.topic_b[j] for j in range(k)]
        return dict(zip(self.topic_labels, _softmax(z)))

    # ---- sentiment ------------------------------------------------------
    def sentence_sentiment(self, sentence: str) -> Dict[str, float]:
        feats = [f for f in set(sentiment_features(sentence)) if f in self.sent_w]
        k = len(self.sent_labels)
        z = [0.0] * k
        inv = 1.0 / math.sqrt(len(feats)) if feats else 0.0
        for f in sorted(feats):
            row = self.sent_w[f]
            for j in range(k):
                z[j] += row[j] * inv
        z = [z[j] * self.sent_scale + self.sent_b[j] for j in range(k)]
        return dict(zip(self.sent_labels, _softmax(z)))

    def document_sentiment(self, text: str) -> dict:
        """Same aggregation as src/sentiment.analyse_document."""
        sents = sent_tokenize(text)
        if not sents:
            return {"label": "Neutral", "polarity": 0.0,
                    "distribution": {"Negative": 0, "Neutral": 0, "Positive": 0},
                    "sentences": []}
        scored = []
        for s in sents:
            p = self.sentence_sentiment(s)
            label = max(p, key=p.get)
            pol = sum(POLARITY[k] * v for k, v in p.items())
            scored.append({"text": s, "label": label, "polarity": pol})
        weights = [max(len(s["text"]), 1) for s in scored]
        pol = sum(s["polarity"] * w for s, w in zip(scored, weights)) / sum(weights)
        label = "Positive" if pol > 0.15 else "Negative" if pol < -0.15 else "Neutral"
        dist = {"Negative": 0, "Neutral": 0, "Positive": 0}
        for s in scored:
            dist[s["label"]] += 1
        return {"label": label, "polarity": pol, "distribution": dist,
                "sentences": scored}

    # ---- entities -------------------------------------------------------
    def tag(self, text: str) -> List[Tuple[int, int, str]]:
        """Word-level labels, as (start, end, label) for every word."""
        offs = words(text)
        toks = [text[s:e] for s, e in offs]
        k = len(self.ner_labels)
        out, prev, prev2 = [], "O", "O"
        for i in range(len(toks)):
            z = list(self.ner_b)
            for f in ner_features(toks, i, prev, prev2):
                row = self.ner_w.get(f)
                if row is not None:
                    for j in range(k):
                        z[j] += row[j] * self.ner_scale
            best = max(range(k), key=lambda j: z[j])
            lab = self.ner_labels[best]
            out.append((offs[i][0], offs[i][1], lab))
            prev2, prev = prev, lab
        return out

    def entities(self, text: str) -> List[Tuple[int, int, str]]:
        """Merge adjacent same-label words separated by one character."""
        spans: List[list] = []
        for s, e, lab in self.tag(text):
            if lab == "O":
                continue
            if spans and spans[-1][2] == lab and s - spans[-1][1] <= 1:
                spans[-1][1] = e
            else:
                spans.append([s, e, lab])
        return [tuple(x) for x in spans]


def group_entities(text: str, spans) -> Dict[str, List[str]]:
    """Same grouping as src/ner.summarize_entities: root key, surface label."""
    counts: Counter = Counter()
    surfaces: Dict[tuple, Counter] = defaultdict(Counter)
    for s, e, lab in spans:
        surf = text[s:e].strip()
        if not surf:
            continue
        parts = surf.split()
        canon = " ".join(parts[:-1] + [stem(parts[-1])])
        counts[(lab, canon)] += 1
        surfaces[(lab, canon)][surf] += 1
    out: Dict[str, List[str]] = defaultdict(list)
    for (lab, canon), _ in counts.most_common():
        out[lab].append(surfaces[(lab, canon)].most_common(1)[0][0])
    return dict(out)


def analyse(model: "Lite", title: str, text: str) -> dict:
    """
    The whole in-browser pipeline, mirroring src/pipeline.process with the
    students in place of the MahaBERT models. The page's JavaScript pipeline()
    must return the same thing; scripts/verify_demo.py checks that it does.
    """
    from src.summarize import summarize
    title, text = (title or "").strip(), (text or "").strip()
    summ = summarize(text, k=3)
    summary = summ.text
    ent_source = f"{title}. {summary}".strip()
    sent = model.document_sentiment(summary or text)
    return {
        "topic_scores": model.topic(title, text),
        "summary": summary,
        "entities": group_entities(ent_source, model.entities(ent_source)),
        "sentiment": sent["label"],
        "polarity": sent["polarity"],
        "sentiment_distribution": sent["distribution"],
    }
