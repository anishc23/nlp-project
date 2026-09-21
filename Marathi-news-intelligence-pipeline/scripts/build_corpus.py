# -*- coding: utf-8 -*-
"""
Enrich raw articles into the analysed corpus that the index and dashboard use.

Runs the whole pipeline over XL-Sum Marathi (or a live RSS pull) and writes one
JSON object per line to data/processed/corpus.jsonl.

The stages are deliberately run *stage-by-stage over the whole batch* rather
than document-by-document: the GPU is only worth using when it gets a full
batch, and this way each model is resident once instead of being swapped in
and out per article.

    python scripts/build_corpus.py --limit 1500
    python scripts/build_corpus.py --source rss --translate
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config                                    # noqa: E402
from src import categorize, ner, sentiment, summarize  # noqa: E402
from src.morph import analyse, tokenize          # noqa: E402
from src.pipeline import Document                # noqa: E402


def load_xlsum(limit: int, split: str = "train"):
    path = {"train": config.XLSUM_TRAIN, "val": config.XLSUM_VAL,
            "test": config.XLSUM_TEST}[split]
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if limit and len(rows) >= limit:
                break
            r = json.loads(line)
            if len(r.get("text", "")) < 200:      # skip stubs
                continue
            rows.append({"id": r["id"], "title": r["title"], "text": r["text"],
                         "url": r.get("url", ""), "gold_summary": r.get("summary", "")})
    return rows


def load_rss(limit: int):
    from src.ingest import fetch_all
    return fetch_all(limit=limit)


def morph_stats(text: str):
    toks = tokenize(text)
    if not toks:
        return 0, 0.0
    return len(toks), sum(1 for t in toks if analyse(t).is_inflected) / len(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--source", choices=["xlsum", "rss"], default="xlsum")
    ap.add_argument("--split", default="train")
    ap.add_argument("--summary-k", type=int, default=3)
    ap.add_argument("--method", default="textrank")
    ap.add_argument("--translate", action="store_true",
                    help="also render English title/summary (slow: NLLB-600M)")
    ap.add_argument("--out", default=str(config.CORPUS_JSONL))
    args = ap.parse_args()

    t0 = time.time()
    rows = load_rss(args.limit) if args.source == "rss" else load_xlsum(args.limit, args.split)
    print(f"[1/6] loaded {len(rows)} articles from {args.source}")

    docs = [Document(id=str(r.get("id") or i), title=r.get("title", ""),
                     text=r.get("text", ""), url=r.get("url", ""))
            for i, r in enumerate(rows)]

    # --- E: summaries (CPU, no model) ------------------------------------
    for d in docs:
        s = summarize.summarize(d.text, k=args.summary_k, method=args.method)
        d.summary, d.summary_sentences = s.text, s.sentences
    print(f"[2/6] summarised            ({time.time() - t0:.0f}s)")

    # --- D: topic, batched ------------------------------------------------
    preds = categorize.classify([f"{d.title}. {d.text}" for d in docs])
    for d, p in zip(docs, preds):
        d.topic, d.topic_score, d.topic_scores = p.label, p.score, p.all_scores
    print(f"[3/6] categorised           ({time.time() - t0:.0f}s)")

    # --- G + F: entities and sentiment ------------------------------------
    for n, d in enumerate(docs, 1):
        ents = ner.extract(f"{d.title}. {d.summary}".strip())
        d.entities = ner.summarize_entities(ents)
        sd = sentiment.analyse_document(d.summary or d.text)
        d.sentiment, d.polarity = sd.label, sd.polarity
        d.sentiment_distribution = sd.distribution
        d.entity_sentiment = sentiment.entity_sentiment(sd, ents)
        if n % 250 == 0:
            print(f"      ...{n}/{len(docs)}  ({time.time() - t0:.0f}s)", flush=True)
    print(f"[4/6] entities + sentiment  ({time.time() - t0:.0f}s)")

    # --- A: translation (optional) ---------------------------------------
    if args.translate:
        from src import translate as mt
        titles = mt.translate([d.title for d in docs])
        summaries = mt.translate([d.summary[:900] for d in docs])
        for d, t, s in zip(docs, titles, summaries):
            d.title_en, d.summary_en = t, s
        print(f"[5/6] translated            ({time.time() - t0:.0f}s)")
    else:
        print("[5/6] translation skipped (pass --translate to enable)")

    # --- I: morphology stats ---------------------------------------------
    for d in docs:
        d.tokens, d.inflected_ratio = morph_stats(d.text)

    out = Path(args.out)
    with open(out, "w", encoding="utf-8") as fh:
        for d, r in zip(docs, rows):
            rec = d.to_dict()
            if r.get("gold_summary"):
                rec["gold_summary"] = r["gold_summary"]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    avg_infl = sum(d.inflected_ratio for d in docs) / max(len(docs), 1)
    print(f"[6/6] wrote {len(docs)} docs -> {out}")
    print(f"      mean inflected-token ratio: {avg_infl:.1%}")
    print(f"      total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
