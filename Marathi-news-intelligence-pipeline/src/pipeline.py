# -*- coding: utf-8 -*-
"""
End-to-end document pipeline.

Takes one raw Marathi news article and returns a fully enriched record:

    text -> morphology -> topic (D)
                       -> entities (G)
                       -> summary (E)
                       -> sentiment, incl. per-entity tone (F)
                       -> English rendering (A, optional)

Retrieval (C) is not per-document -- it is built over the enriched corpus by
scripts/build_index.py.

Translation is opt-in because NLLB-600M is by far the heaviest model here and
most batch runs do not need it.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import categorize, ner, sentiment, summarize  # noqa: E402
from src.morph import analyse, tokenize  # noqa: E402


@dataclass
class Document:
    """One fully analysed news article."""
    id: Optional[str] = None
    title: str = ""
    text: str = ""
    url: str = ""

    topic: str = ""
    topic_score: float = 0.0
    topic_scores: Dict[str, float] = field(default_factory=dict)

    summary: str = ""
    summary_sentences: List[str] = field(default_factory=list)

    entities: Dict[str, List[str]] = field(default_factory=dict)
    entity_sentiment: Dict[str, float] = field(default_factory=dict)

    sentiment: str = ""
    polarity: float = 0.0
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)

    summary_en: str = ""
    title_en: str = ""

    tokens: int = 0
    inflected_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def search_text(self) -> str:
        """Field actually indexed: title is repeated to weight it up."""
        return f"{self.title} {self.title} {self.text}"


def _morph_stats(text: str) -> tuple:
    toks = tokenize(text)
    if not toks:
        return 0, 0.0
    inflected = sum(1 for t in toks if analyse(t).is_inflected)
    return len(toks), inflected / len(toks)


def process(title: str, text: str, *, url: str = "", doc_id: str = None,
            summary_k: int = 3, summary_method: str = "textrank",
            translate: bool = False) -> Document:
    """Run every module over a single article."""
    doc = Document(id=doc_id, title=title or "", text=text or "", url=url)

    # --- D: topic ---------------------------------------------------------
    clf_input = f"{doc.title}. {doc.text}" if doc.title else doc.text
    pred = categorize.classify([clf_input])[0]
    doc.topic, doc.topic_score, doc.topic_scores = pred.label, pred.score, pred.all_scores

    # --- E: summary -------------------------------------------------------
    summ = summarize.summarize(doc.text, k=summary_k, method=summary_method)
    doc.summary, doc.summary_sentences = summ.text, summ.sentences

    # --- G: entities ------------------------------------------------------
    # NER is run over title + summary rather than the full article: the model
    # truncates at 256 tokens anyway, and the summary is a denser carrier of
    # the article's principal entities than an arbitrary 256-token prefix.
    ent_source = f"{doc.title}. {doc.summary}".strip()
    ents = ner.extract(ent_source)
    doc.entities = ner.summarize_entities(ents)

    # --- F: sentiment -----------------------------------------------------
    sent_doc = sentiment.analyse_document(doc.summary or doc.text)
    doc.sentiment = sent_doc.label
    doc.polarity = sent_doc.polarity
    doc.sentiment_distribution = sent_doc.distribution
    doc.entity_sentiment = sentiment.entity_sentiment(sent_doc, ents)

    # --- A: translation (optional) ---------------------------------------
    if translate:
        from src import translate as mt
        doc.title_en = mt.translate(doc.title) if doc.title else ""
        doc.summary_en = mt.translate_document(doc.summary) if doc.summary else ""

    # --- I: morphology statistics ----------------------------------------
    doc.tokens, doc.inflected_ratio = _morph_stats(doc.text)
    return doc


def process_many(rows: List[Dict[str, str]], **kwargs) -> List[Document]:
    """Process a list of {title, text, url, id} dicts."""
    out = []
    for r in rows:
        out.append(process(r.get("title", ""), r.get("text", ""),
                           url=r.get("url", ""), doc_id=r.get("id"), **kwargs))
    return out


if __name__ == "__main__":
    import json
    import config
    sys.stdout.reconfigure(encoding="utf-8")

    row = json.loads(open(config.XLSUM_TEST, encoding="utf-8").readline())
    doc = process(row["title"], row["text"], url=row["url"], doc_id=row["id"])

    print("TITLE     :", doc.title)
    print("TOPIC     :", f"{doc.topic} ({doc.topic_score:.3f})")
    print("SENTIMENT :", f"{doc.sentiment} ({doc.polarity:+.3f})", doc.sentiment_distribution)
    print("TOKENS    :", doc.tokens, f"| inflected {doc.inflected_ratio:.1%}")
    print("ENTITIES  :")
    for k, v in doc.entities.items():
        print(f"    {k:14s} {', '.join(v[:6])}")
    print("SUMMARY   :", doc.summary[:300])
    print("\nGOLD      :", row["summary"][:200])
