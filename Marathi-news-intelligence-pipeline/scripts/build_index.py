# -*- coding: utf-8 -*-
"""
Build the retrieval indexes over the enriched corpus.

    python scripts/build_index.py                # BM25 (morph) + dense
    python scripts/build_index.py --no-dense     # BM25 only, no GPU needed
    python scripts/build_index.py --plain-bm25   # ablation index, no morphology

The `--plain-bm25` variant writes a second index built from raw whitespace
tokens. Keeping both on disk is what lets scripts/evaluate.py run the
morphology ablation without rebuilding anything.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config                                     # noqa: E402
from src.retrieve import BM25Index, DenseIndex    # noqa: E402

PLAIN_BM25_INDEX = config.INDEX / "bm25_plain.pkl"


def load_corpus(path: Path):
    if not path.exists():
        sys.exit(f"corpus not found: {path}\nRun scripts/build_corpus.py first.")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def search_text(rec: dict) -> str:
    """Title is repeated so it outweighs the body in BM25 term frequency."""
    return f"{rec.get('title', '')} {rec.get('title', '')} {rec.get('text', '')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(config.CORPUS_JSONL))
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--plain-bm25", action="store_true",
                    help="also build a morphology-free index for the ablation")
    args = ap.parse_args()

    corpus = load_corpus(Path(args.corpus))
    texts = [search_text(r) for r in corpus]
    doc_ids = list(range(len(corpus)))
    print(f"indexing {len(corpus)} documents")

    t0 = time.time()
    bm25 = BM25Index(use_morph=True).build(texts, doc_ids)
    bm25.save(config.BM25_INDEX)
    print(f"  BM25 (morph) : {len(bm25.postings):>7,} terms  "
          f"avg_len={bm25.avg_len:.0f}  ({time.time() - t0:.1f}s)")

    if args.plain_bm25:
        t1 = time.time()
        plain = BM25Index(use_morph=False).build(texts, doc_ids)
        plain.save(PLAIN_BM25_INDEX)
        print(f"  BM25 (plain) : {len(plain.postings):>7,} terms  "
              f"({time.time() - t1:.1f}s)")
        shrink = 1 - len(bm25.postings) / max(len(plain.postings), 1)
        print(f"  vocabulary reduced by {shrink:.1%} through morphological normalisation")

    if not args.no_dense:
        t2 = time.time()
        # Embed title + summary rather than the full article: the encoder
        # truncates at 256 tokens, and the summary is a better 256 tokens than
        # an arbitrary prefix of the body.
        dense_texts = [f"{r.get('title', '')}. {r.get('summary', '')}" for r in corpus]
        dense = DenseIndex().build(dense_texts, doc_ids)
        dense.save(config.DENSE_INDEX)
        print(f"  Dense        : {dense.vectors.shape}  ({time.time() - t2:.1f}s)")

    print(f"done in {time.time() - t0:.1f}s -> {config.INDEX}")


if __name__ == "__main__":
    main()
