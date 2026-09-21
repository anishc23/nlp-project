# -*- coding: utf-8 -*-
"""
Export everything the demo page needs into one JSON file.

    python scripts/export_demo.py --docs 400 --showcase 8

Two things are exported, and the distinction matters:

  * REAL PIPELINE OUTPUT -- topics, entities, sentiment, summaries and
    translations that the actual models produced, copied verbatim out of
    data/processed/corpus.jsonl. The demo page does not re-run any model; it
    displays what the pipeline already computed.

  * THE ANALYSER'S RULE TABLES -- the suffix inventory, stopword list and
    exception map are dumped straight from src/morph.py so the JavaScript
    port on the page runs the *same* rules rather than a hand-retyped copy
    that could silently drift out of sync.

Translation is done here rather than in build_corpus so only the handful of
showcase articles pay the NLLB cost. Run with CUDA_VISIBLE_DEVICES= to keep it
off a GPU that is busy training.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config                       # noqa: E402
from src import morph               # noqa: E402
from src.lite import LITE_MODELS    # noqa: E402
from src.ner import merge_aliases   # noqa: E402


def morph_tables() -> dict:
    """Dump the analyser's rules so the JS port stays in lockstep."""
    def flat(table):
        return [[t[0], t[1], t[2] if len(t) > 2 else ""] for t in table]

    return {
        "clitics": flat(morph.CLITICS),
        "postpositions": flat(morph.POSTPOSITIONS),
        "verb": flat(morph.VERB_SUFFIXES),
        "case": flat(morph.CASE_NUMBER),
        "stopwords": sorted(morph.STOPWORDS),
        "exceptions": morph.EXCEPTIONS,
        "marks": "".join(sorted(morph._MARKS)),
        "finalMatras": "".join(sorted(morph._FINAL_MATRAS)),
        "minStem": morph.MIN_STEM,
        "maxStrips": morph.MAX_STRIPS,
        "virama": morph.VIRAMA,
    }


def pick_showcase(corpus, n):
    """Spread the showcase across topics, preferring entity-rich articles."""
    by_topic = defaultdict(list)
    for r in corpus:
        by_topic[r.get("topic", "?")].append(r)

    def richness(r):
        return sum(len(v) for v in (r.get("entities") or {}).values())

    picks, topics = [], sorted(by_topic, key=lambda t: -len(by_topic[t]))
    while len(picks) < n and topics:
        for t in list(topics):
            pool = [r for r in by_topic[t]
                    if 400 < len(r.get("text", "")) < 6000 and richness(r) >= 4]
            pool.sort(key=richness, reverse=True)
            pool = [r for r in pool if r not in picks]
            if pool:
                picks.append(pool[0])
            else:
                topics.remove(t)
            if len(picks) >= n:
                break
    return picks[:n]


def slim(rec, keep_text=False):
    out = {
        "id": rec.get("id"),
        "title": rec.get("title", ""),
        "summary": rec.get("summary", ""),
        "topic": rec.get("topic", ""),
        "topic_score": round(float(rec.get("topic_score", 0)), 3),
        "sentiment": rec.get("sentiment", ""),
        "polarity": round(float(rec.get("polarity", 0)), 3),
        "entities": rec.get("entities") or {},
        "url": rec.get("url", ""),
        "tokens": rec.get("tokens", 0),
        "inflected_ratio": round(float(rec.get("inflected_ratio", 0)), 3),
    }
    if keep_text:
        out["text"] = rec.get("text", "")
        out["gold_summary"] = rec.get("gold_summary", "")
        out["topic_scores"] = {k: round(float(v), 4)
                               for k, v in (rec.get("topic_scores") or {}).items()}
        out["sentiment_distribution"] = rec.get("sentiment_distribution") or {}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=400, help="docs in the search index")
    ap.add_argument("--showcase", type=int, default=8)
    ap.add_argument("--no-translate", action="store_true")
    ap.add_argument("--out", default=str(config.OUTPUTS / "demo_data.json"))
    args = ap.parse_args()

    if not LITE_MODELS.exists():
        sys.exit(f"{LITE_MODELS} missing -- run scripts/train_lite.py first")
    corpus = [json.loads(l) for l in open(config.CORPUS_JSONL, encoding="utf-8")]
    print(f"corpus: {len(corpus)} docs")

    showcase = pick_showcase(corpus, args.showcase)
    print(f"showcase: {len(showcase)} articles "
          f"({', '.join(r['topic'] for r in showcase)})")

    # Search index: the showcase articles plus a topic-spread sample.
    chosen = {id(r): r for r in showcase}
    per_topic = defaultdict(int)
    cap = max(1, args.docs // max(len(set(r.get('topic') for r in corpus)), 1))
    for r in corpus:
        if len(chosen) >= args.docs:
            break
        t = r.get("topic", "?")
        if id(r) in chosen or per_topic[t] >= cap or len(r.get("text", "")) < 300:
            continue
        chosen[id(r)] = r
        per_topic[t] += 1
    for r in corpus:                      # top up if caps left us short
        if len(chosen) >= args.docs:
            break
        if id(r) not in chosen and len(r.get("text", "")) >= 300:
            chosen[id(r)] = r
    docs = list(chosen.values())
    print(f"search index: {len(docs)} docs")

    # Translation of the showcase only.
    if not args.no_translate:
        from src import translate as mt
        print(f"translating {len(showcase)} showcase articles on {config.DEVICE}...")
        titles = mt.translate([r["title"] for r in showcase])
        summaries = mt.translate([r["summary"][:700] for r in showcase])
        for r, t, s in zip(showcase, titles, summaries):
            r["_title_en"], r["_summary_en"] = t, s
        print("  done")

    show_out = []
    for r in showcase:
        d = slim(r, keep_text=True)
        d["title_en"] = r.get("_title_en", "")
        d["summary_en"] = r.get("_summary_en", "")
        show_out.append(d)

    # Corpus-level aggregates, computed over the whole 3,000 not the sample.
    ent_buckets = defaultdict(Counter)
    for r in corpus:
        for label, vals in (r.get("entities") or {}).items():
            for v in vals:
                ent_buckets[label][v] += 1
    top_entities = {
        label: merge_aliases(c).most_common(12)
        for label, c in ent_buckets.items()
        if label in ("Person", "Organization", "Location")
    }

    tone_by_topic = defaultdict(lambda: Counter())
    for r in corpus:
        tone_by_topic[r.get("topic", "?")][r.get("sentiment", "Neutral")] += 1

    payload = {
        "meta": {
            "corpus_size": len(corpus),
            "index_size": len(docs),
            "mean_inflected_ratio": round(
                sum(r.get("inflected_ratio", 0) for r in corpus) / len(corpus), 4),
            "source": "XL-Sum Marathi (BBC Marathi)",
        },
        "morph": morph_tables(),
        "showcase": show_out,
        "docs": [slim(r) for r in docs],
        "topics": Counter(r.get("topic", "?") for r in corpus).most_common(),
        "sentiment": Counter(r.get("sentiment", "?") for r in corpus).most_common(),
        "tone_by_topic": {k: dict(v) for k, v in tone_by_topic.items()},
        "top_entities": top_entities,
        # The in-browser students (src/lite.py), so pasted text can be
        # analysed with no Python -- with their measured agreement.
        "lite": json.loads(LITE_MODELS.read_text(encoding="utf-8")),
    }

    out = Path(args.out)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
