# -*- coding: utf-8 -*-
"""
Named-entity / information extraction (module G).

Uses `l3cube-pune/marathi-ner` (MahaNER). Note that this model emits *flat*
labels -- Person, Location, Organization, Date, Time, Measure, Designation,
Other -- with no B-/I- prefixes. Consecutive wordpieces carrying the same
label therefore have to be merged into spans here, and because there is no
B- tag two adjacent distinct entities of the same type are indistinguishable;
we accept that and merge them, which is the standard reading of this tagset.

Spans are recovered through the tokenizer's offset mapping so the surface text
is taken verbatim from the input rather than rebuilt from wordpieces (which
would lose the original spacing and mangle Devanagari conjuncts).
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                      # noqa: E402
from src import models             # noqa: E402
from src.morph import stem as morph_stem  # noqa: E402

# Types worth surfacing in the dashboard; Measure/Time/Other are kept out of
# the entity summary but still returned by extract().
HEADLINE_TYPES = ("Person", "Organization", "Location", "Date", "Designation")


@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    score: float

    @property
    def canonical(self) -> str:
        """
        Case-stripped form used to aggregate mentions across the corpus.

        Marathi declines proper nouns like any other noun -- मुंबई / मुंबईत /
        मुंबईतील are the same place, and भारतीय जनता पक्षाच्या is the same
        party as भारतीय जनता पक्ष. Counting surface forms would scatter one
        entity across half a dozen rows, so the analyser (module I) is applied
        to the final token, which is the one that carries the case suffix.
        """
        parts = self.text.split()
        if not parts:
            return self.text
        return " ".join(parts[:-1] + [morph_stem(parts[-1])]).strip()

    def __str__(self) -> str:
        return f"{self.text} ({self.label}, {self.score:.2f})"


# Characters that end a word. Anything else (including Devanagari combining
# marks) is treated as word-internal.
_BOUNDARY = set(" \t\n\r " + "।,.;:!?\"'()[]{}<>%/\\|-–—")


def _snap_to_word(text: str, start: int, end: int) -> tuple:
    """
    Widen a span to whole-word boundaries.

    Wordpiece tokenisers happily split a Devanagari word between a consonant
    and its vowel sign, so a span can begin at a combining mark -- which is how
    पंतप्रधान came back as ंतप्रधान. Snapping outwards to the nearest
    boundary character restores the full orthographic word.
    """
    while start > 0 and text[start - 1] not in _BOUNDARY:
        start -= 1
    while end < len(text) and text[end] not in _BOUNDARY:
        end += 1
    return start, end


@torch.no_grad()
def extract(text: str, max_length: int = None) -> List[Entity]:
    """Extract entities from a single document."""
    if not text or not text.strip():
        return []

    tok, mod = models.load_token_classifier(config.MODEL_NER)
    id2label = mod.config.id2label
    max_length = max_length or config.MAX_LEN

    enc = tok(text, return_offsets_mapping=True, truncation=True,
              max_length=max_length, return_tensors="pt")
    offsets = enc.pop("offset_mapping")[0].tolist()
    enc.pop("special_tokens_mask", None)
    enc = {k: v.to(config.DEVICE) for k, v in enc.items()}

    probs = mod(**enc).logits.float().softmax(-1)[0].cpu()
    label_ids = probs.argmax(-1).tolist()
    scores = probs.max(-1).values.tolist()

    spans: List[Entity] = []
    cur = None
    for (start, end), lid, sc in zip(offsets, label_ids, scores):
        if start == end:                      # special token
            continue
        label = id2label[lid]
        if label == "Other":
            cur = None
            continue
        # Continue the current span when the label matches and the gap is only
        # whitespace; otherwise start a new one.
        if cur and cur.label == label and start - cur.end <= 1:
            cur.end = end
            cur.score = (cur.score + sc) / 2
        else:
            cur = Entity(text="", label=label, start=start, end=end, score=sc)
            spans.append(cur)

    for sp in spans:
        sp.start, sp.end = _snap_to_word(text, sp.start, sp.end)
        sp.text = text[sp.start:sp.end].strip()

    # Snapping can make two adjacent spans of the same type identical; collapse
    # runs of duplicates while preserving order.
    deduped: List[Entity] = []
    for sp in spans:
        if not sp.text:
            continue
        if deduped and deduped[-1].text == sp.text and deduped[-1].label == sp.label:
            continue
        deduped.append(sp)
    return deduped


def extract_many(texts: List[str]) -> List[List[Entity]]:
    return [extract(t) for t in texts]


def summarize_entities(entities: List[Entity]) -> Dict[str, List[str]]:
    """
    Group entities by type, de-duplicated and ordered by frequency.

    Mentions are *grouped* by their canonical (de-inflected) form so that
    मुंबई and मुंबईत count as one entity, but each group is *displayed* using
    its most frequent surface form -- de-inflection is a good equivalence key
    and a poor label, since it turns शिंदे into शिंद.
    """
    counts: Dict[tuple, int] = Counter()
    surfaces: Dict[tuple, Counter] = defaultdict(Counter)
    for e in entities:
        key = (e.label, e.canonical)
        counts[key] += 1
        surfaces[key][e.text] += 1

    buckets: Dict[str, List[str]] = defaultdict(list)
    for (label, _canon), _n in counts.most_common():
        buckets[label].append(surfaces[(label, _canon)].most_common(1)[0][0])
    return dict(buckets)


def merge_aliases(counts: Counter, min_parts: int = 2,
                  host_ratio: float = 0.5) -> Counter:
    """
    Fold short-form mentions into the full name they abbreviate.

    News text alternates between नरेंद्र मोदी and plain मोदी, and counting
    them separately splits one person across two rows in the corpus-level
    tallies. A single-token name is merged into a multi-token name that ends
    with it, provided the long form is at least `host_ratio` as frequent as
    the short one. That threshold keeps a genuinely common surname from being
    absorbed into some rare full name that happens to share it, while still
    merging the ordinary case where both forms are used about equally often
    (ट्रंप 40 / डोनाल्ड ट्रंप 39).

    Deliberately conservative: it does not touch initials, honorifics or
    nicknames, and it never merges two multi-token names. Real coreference
    would need a dedicated model; this only fixes the one pattern that
    dominates the counts.
    """
    merged = Counter(counts)
    longs = [n for n in counts if len(n.split()) >= min_parts]
    for short, n_short in list(counts.items()):
        if len(short.split()) != 1:
            continue
        hosts = [L for L in longs if L.split()[-1] == short]
        if not hosts:
            continue
        host = max(hosts, key=lambda L: counts[L])
        if counts[host] >= host_ratio * n_short:
            merged[host] += n_short
            del merged[short]
    return merged


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    demo = ("मुख्यमंत्री एकनाथ शिंदे यांनी सोमवारी मुंबईत भारतीय जनता पक्षाच्या "
            "बैठकीत नवीन योजनेची घोषणा केली. ही योजना १ जानेवारी २०२४ पासून "
            "लागू होणार आहे.")
    ents = extract(demo)
    for e in ents:
        print("  ", e)
    print("\n  grouped:", summarize_entities(ents))
