# -*- coding: utf-8 -*-
"""
Evaluation suite. Five experiments, each runnable on its own:

    python scripts/evaluate.py --task retrieval       # full-headline queries
    python scripts/evaluate.py --task shortquery      # 1-3 word queries
    python scripts/evaluate.py --task crossinfl       # when morphology decides
    python scripts/evaluate.py --task summarization
    python scripts/evaluate.py --task classification
    python scripts/evaluate.py --task all

RETRIEVAL (1, 1b, 1c) is what the project is built around: the same BM25 index
is constructed with and without the morphological analyser and queried the
same way. The three variants exist because the first one alone is misleading.

  1  full-headline queries     -- morphology gives a small gain (+3.9% MRR)
  1b short, realistic queries  -- morphology gives no gain, and at 1-2 terms
                                  is slightly worse
  1c the subset where a query term occurs in the body only in a different
     inflection -- 68% of articles have one, and there the plain index returns
     an empty result page 14.4% of the time against 1.1% with morphology

Taken together: morphology is recall insurance for Marathi search, not a
ranking improvement. Reporting only experiment 1 would overstate it, and
reporting only the hand-picked शाळा example would overstate it badly.

SUMMARIZATION scores extractive summaries against XL-Sum's gold abstracts with
ROUGE-1/2/L. ROUGE is implemented here rather than taken from `rouge_score`,
whose default tokenizer strips non-ASCII and would score every Marathi summary
as 0. Scores are reported over both raw and morphologically normalised tokens:
the raw numbers are comparable with published work, the normalised ones are
the fairer measurement for a language where the same lemma surfaces in half a
dozen forms.

Note on the ceiling: XL-Sum summaries are abstractive and roughly one
sentence, so an extractive system cannot reach a high ROUGE here. The lead
baseline is included so the comparison is against something honest.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config                                        # noqa: E402
from src.morph import normalize_tokens, tokenize     # noqa: E402

SEED = 13


# ==========================================================================
# ROUGE
# ==========================================================================
def _f1(match: int, pred_n: int, gold_n: int) -> float:
    if match == 0 or pred_n == 0 or gold_n == 0:
        return 0.0
    p, r = match / pred_n, match / gold_n
    return 2 * p * r / (p + r)


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(pred: List[str], gold: List[str], n: int) -> float:
    p, g = _ngrams(pred, n), _ngrams(gold, n)
    overlap = sum((p & g).values())
    return _f1(overlap, max(sum(p.values()), 0), max(sum(g.values()), 0))


def rouge_l(pred: List[str], gold: List[str]) -> float:
    """F1 over the longest common subsequence (O(n*m), quadratic memory-free)."""
    if not pred or not gold:
        return 0.0
    prev = [0] * (len(gold) + 1)
    for a in pred:
        cur = [0]
        for j, b in enumerate(gold):
            cur.append(prev[j] + 1 if a == b else max(cur[j], prev[j + 1]))
        prev = cur
    return _f1(prev[-1], len(pred), len(gold))


def rouge_all(pred_text: str, gold_text: str, use_morph: bool) -> dict:
    if use_morph:
        p = normalize_tokens(pred_text, drop_stopwords=False)
        g = normalize_tokens(gold_text, drop_stopwords=False)
    else:
        p, g = tokenize(pred_text), tokenize(gold_text)
    return {"rouge1": rouge_n(p, g, 1), "rouge2": rouge_n(p, g, 2),
            "rougeL": rouge_l(p, g)}


# ==========================================================================
# Experiment 1 -- retrieval
# ==========================================================================
def eval_retrieval(corpus, n_queries=300, top_k=10, include_dense=True):
    """
    Known-item retrieval: each query is one document's title, and the correct
    answer is that document. Titles are held out of the indexed text so the
    task is not trivially solved by an exact string match.
    """
    from src.retrieve import BM25Index, DenseIndex, HybridRetriever

    random.seed(SEED)
    bodies = [r.get("text", "") for r in corpus]
    doc_ids = list(range(len(corpus)))

    eligible = [i for i, r in enumerate(corpus)
                if len(r.get("title", "").split()) >= 4 and len(r.get("text", "")) > 300]
    queries = random.sample(eligible, min(n_queries, len(eligible)))
    print(f"  {len(corpus)} docs indexed, {len(queries)} known-item queries\n")

    runs = {}
    built = {}

    for use_morph in (False, True):
        tag = "BM25 + morphology" if use_morph else "BM25 (plain tokens)"
        t0 = time.time()
        idx = BM25Index(use_morph=use_morph).build(bodies, doc_ids)
        built[use_morph] = idx
        results = []
        for qi in queries:
            hits = idx.search(corpus[qi]["title"], top_k=top_k)
            rank = next((h.rank for h in hits if h.doc_id == qi), None)
            results.append(rank)
        runs[tag] = results
        runs[tag + "::vocab"] = len(idx.postings)
        runs[tag + "::time"] = time.time() - t0

    if include_dense:
        t0 = time.time()
        dense_texts = [f"{r.get('summary', '') or r.get('text', '')[:600]}" for r in corpus]
        dense = DenseIndex().build(dense_texts, doc_ids, show_progress=False)
        results = []
        for qi in queries:
            hits = dense.search(corpus[qi]["title"], top_k=top_k)
            rank = next((h.rank for h in hits if h.doc_id == qi), None)
            results.append(rank)
        runs["Dense (MahaBERT-SBERT)"] = results
        runs["Dense (MahaBERT-SBERT)::time"] = time.time() - t0

        hybrid = HybridRetriever(bm25=built[True], dense=dense)
        results = []
        for qi in queries:
            hits = hybrid.search(corpus[qi]["title"], top_k=top_k)
            rank = next((h.rank for h in hits if h.doc_id == qi), None)
            results.append(rank)
        runs["Hybrid (RRF)"] = results

    # ---- report ----
    header = f"  {'system':26s} {'R@1':>7s} {'R@5':>7s} {'R@10':>7s} {'MRR@10':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    table = {}
    for name, ranks in runs.items():
        if "::" in name:
            continue
        found = [r for r in ranks if r is not None]
        r1 = sum(1 for r in found if r < 1) / len(ranks)
        r5 = sum(1 for r in found if r < 5) / len(ranks)
        r10 = len(found) / len(ranks)
        mrr = sum(1 / (r + 1) for r in found) / len(ranks)
        table[name] = {"R@1": r1, "R@5": r5, "R@10": r10, "MRR@10": mrr}
        print(f"  {name:26s} {r1:7.3f} {r5:7.3f} {r10:7.3f} {mrr:8.3f}")

    plain = table.get("BM25 (plain tokens)", {}).get("MRR@10", 0)
    morph = table.get("BM25 + morphology", {}).get("MRR@10", 0)
    if plain:
        print(f"\n  morphology lifts BM25 MRR@10 by {(morph - plain) / plain:+.1%}")
    v_plain = runs.get("BM25 (plain tokens)::vocab", 0)
    v_morph = runs.get("BM25 + morphology::vocab", 0)
    if v_plain:
        print(f"  vocabulary: {v_plain:,} -> {v_morph:,} types "
              f"({1 - v_morph / v_plain:.1%} smaller)")
    return table


def eval_short_queries(corpus, n_queries=300, top_k=10, lengths=(1, 2, 3)):
    """
    The same known-item task, but with realistically short queries.

    A full headline contains eight or ten content words, so even if the
    analyser rescues two of them the other six already found the document --
    which is why the full-title experiment understates what morphology does.
    Real search traffic is one to three words, where a single unmatched term
    is the difference between a hit and an empty result page.

    Query terms are taken verbatim from the document's *title* and matched
    against its *body*. Nothing is lemmatised on the way in: the two fields
    simply inflect the same nouns differently, which is the effect being
    measured. Taking roots as queries would bias the test toward the analyser.
    """
    from src.retrieve import BM25Index
    from src.morph import STOPWORDS

    random.seed(SEED)
    bodies = [r.get("text", "") for r in corpus]
    doc_ids = list(range(len(corpus)))

    def content_words(title):
        return [t for t in tokenize(title) if t not in STOPWORDS and len(t) > 2]

    eligible = [i for i, r in enumerate(corpus)
                if len(content_words(r.get("title", ""))) >= max(lengths)
                and len(r.get("text", "")) > 300]
    sample = random.sample(eligible, min(n_queries, len(eligible)))

    indexes = {m: BM25Index(use_morph=m).build(bodies, doc_ids) for m in (False, True)}

    print(f"  {len(sample)} queries, terms taken verbatim from each article's title\n")
    print(f"  {'query len':>10s} {'plain MRR':>11s} {'morph MRR':>11s} "
          f"{'plain R@10':>11s} {'morph R@10':>11s} {'zero-hit plain':>15s}")
    print("  " + "-" * 74)

    table = {}
    for n in lengths:
        stats = {}
        for use_morph, idx in indexes.items():
            ranks, empty = [], 0
            for qi in sample:
                q = " ".join(content_words(corpus[qi]["title"])[:n])
                hits = idx.search(q, top_k=top_k)
                if not hits:
                    empty += 1
                ranks.append(next((h.rank for h in hits if h.doc_id == qi), None))
            found = [r for r in ranks if r is not None]
            stats[use_morph] = {
                "MRR@10": sum(1 / (r + 1) for r in found) / len(ranks),
                "R@10": len(found) / len(ranks),
                "zero_hit_rate": empty / len(ranks),
            }
        table[n] = {"plain": stats[False], "morph": stats[True]}
        print(f"  {n:>10d} {stats[False]['MRR@10']:>11.3f} {stats[True]['MRR@10']:>11.3f} "
              f"{stats[False]['R@10']:>11.3f} {stats[True]['R@10']:>11.3f} "
              f"{stats[False]['zero_hit_rate']:>14.1%}")

    best = max(lengths, key=lambda n: (table[n]["morph"]["MRR@10"]
                                       - table[n]["plain"]["MRR@10"]))
    p, m = table[best]["plain"]["MRR@10"], table[best]["morph"]["MRR@10"]
    if p:
        print(f"\n  largest gain at {best}-term queries: MRR@10 {p:.3f} -> {m:.3f} "
              f"({(m - p) / p:+.1%})")
    return table


def eval_cross_inflection(corpus, n_docs=1200, top_k=10):
    """
    How often does morphology actually decide a query, and how much does it
    help when it does?

    Experiments 1 and 1b measure the average query, and on the average query
    the analyser barely moves the needle -- a title word usually appears in
    the body in the same form, so there is nothing to rescue. The interesting
    quantity is the size and difficulty of the subset where that is *not*
    true.

    A query here is a single title word chosen so that
        - its exact surface form does NOT occur in the article body, but
        - some body token shares its root.
    That is precisely the case a plain index cannot serve and a normalised one
    can. The `coverage` figure below says how often such a word exists at all,
    which is the honest measure of how much this matters in practice.
    """
    from src.retrieve import BM25Index
    from src.morph import STOPWORDS, stem

    random.seed(SEED)
    bodies = [r.get("text", "") for r in corpus]
    doc_ids = list(range(len(corpus)))

    pool = [i for i, r in enumerate(corpus) if len(r.get("text", "")) > 300]
    sample = random.sample(pool, min(n_docs, len(pool)))

    cases = []
    for qi in sample:
        body_tokens = tokenize(corpus[qi]["text"])
        surface = set(body_tokens)
        roots = {stem(t) for t in body_tokens}
        for w in tokenize(corpus[qi].get("title", "")):
            if w in STOPWORDS or len(w) <= 2:
                continue
            if w not in surface and stem(w) in roots:
                cases.append((qi, w))
                break

    coverage = len(cases) / len(sample)
    print(f"  {len(sample)} articles examined")
    print(f"  {len(cases)} ({coverage:.1%}) contain a headline word that is "
          f"absent from the body\n  in its surface form but present as a root "
          f"-- i.e. only a morphological\n  index can match it\n")

    if not cases:
        return {"coverage": coverage}

    indexes = {m: BM25Index(use_morph=m).build(bodies, doc_ids) for m in (False, True)}
    out = {}
    print(f"  {'system':22s} {'R@1':>7s} {'R@10':>7s} {'MRR@10':>8s} {'zero-hit':>10s}")
    print("  " + "-" * 58)
    for use_morph, idx in indexes.items():
        ranks, empty = [], 0
        for qi, w in cases:
            hits = idx.search(w, top_k=top_k)
            if not hits:
                empty += 1
            ranks.append(next((h.rank for h in hits if h.doc_id == qi), None))
        found = [r for r in ranks if r is not None]
        stats = {
            "R@1": sum(1 for r in found if r < 1) / len(ranks),
            "R@10": len(found) / len(ranks),
            "MRR@10": sum(1 / (r + 1) for r in found) / len(ranks),
            "zero_hit_rate": empty / len(ranks),
        }
        tag = "BM25 + morphology" if use_morph else "BM25 (plain tokens)"
        out[tag] = stats
        print(f"  {tag:22s} {stats['R@1']:7.3f} {stats['R@10']:7.3f} "
              f"{stats['MRR@10']:8.3f} {stats['zero_hit_rate']:9.1%}")

    out["coverage"] = coverage
    print(f"\n  On this subset the plain index returns nothing for "
          f"{out['BM25 (plain tokens)']['zero_hit_rate']:.0%} of queries.")
    print(f"  Weighted over all traffic, the subset is {coverage:.1%} of queries.")
    return out


# ==========================================================================
# Experiment 2 -- summarization
# ==========================================================================
def eval_summarization(corpus, n=300, k=2):
    from src import summarize as S

    rows = [r for r in corpus if r.get("gold_summary") and len(r.get("text", "")) > 400]
    random.seed(SEED)
    rows = random.sample(rows, min(n, len(rows)))
    if not rows:
        print("  no gold summaries in corpus (build it from XL-Sum to evaluate)")
        return {}
    print(f"  scoring {len(rows)} articles against XL-Sum gold summaries (k={k})\n")

    methods = ["lead", "textrank"]
    out = {}
    for method in methods:
        for use_morph in (False, True):
            scores = []
            for r in rows:
                pred = S.summarize(r["text"], k=k, method=method).text
                scores.append(rouge_all(pred, r["gold_summary"], use_morph))
            agg = {m: float(np.mean([s[m] for s in scores])) for m in
                   ("rouge1", "rouge2", "rougeL")}
            tag = f"{method}{' +morph' if use_morph else ''}"
            out[tag] = agg

    print(f"  {'system':18s} {'ROUGE-1':>9s} {'ROUGE-2':>9s} {'ROUGE-L':>9s}")
    print("  " + "-" * 48)
    for tag, agg in out.items():
        print(f"  {tag:18s} {agg['rouge1']:9.4f} {agg['rouge2']:9.4f} {agg['rougeL']:9.4f}")
    print("\n  (+morph = ROUGE computed over morphologically normalised tokens)")
    return out


# ==========================================================================
# Experiment 3 -- classification
# ==========================================================================
def eval_classification(limit=0, model_name=None):
    import pandas as pd
    from sklearn.metrics import classification_report

    from src import categorize

    df = pd.read_parquet(config.MAHANEWS_TEST)
    if limit:
        df = df.head(limit)
    texts = df["text"].astype(str).tolist()
    gold = [config.MAHANEWS_LABELS[int(x)] for x in df["label"]]

    model = model_name or categorize.active_model()
    print(f"  model: {model}")
    print(f"  {len(texts)} test headlines\n")

    preds = [p.label for p in categorize.classify(texts, model_name=model)]

    # The 12-class L3Cube model has to be mapped onto the 3-class MahaNews
    # label space before it can be scored against this test set.
    if model == config.MODEL_TOPIC:
        mapping = {"Sports": "sports", "Manoranjan": "entertainment",
                   "Politics": "state", "Crime": "state", "Auto": "state",
                   "Bhakti": "state", "Education": "state", "Fashion": "entertainment",
                   "Health": "state", "International": "state", "Tech": "state",
                   "Travel": "state"}
        preds = [mapping.get(p, "state") for p in preds]
        print("  (12-class zero-shot predictions mapped onto the 3-class"
              " MahaNews label space)\n")

    acc = sum(p == g for p, g in zip(preds, gold)) / len(gold)
    print(classification_report(gold, preds, digits=4, zero_division=0))
    print(f"  accuracy = {acc:.4f}")
    return {"accuracy": acc}


# ==========================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all",
                    choices=["all", "retrieval", "shortquery", "crossinfl",
                             "summarization", "classification"])
    ap.add_argument("--corpus", default=str(config.CORPUS_JSONL))
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--summ-n", type=int, default=300)
    ap.add_argument("--clf-limit", type=int, default=0)
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--out", default=str(config.OUTPUTS / "evaluation.json"))
    args = ap.parse_args()

    corpus = []
    path = Path(args.corpus)
    if path.exists():
        corpus = [json.loads(l) for l in open(path, encoding="utf-8")]

    results = {}
    if args.task in ("all", "retrieval"):
        print("\n" + "=" * 68)
        print("EXPERIMENT 1 -- RETRIEVAL: does morphology help find Marathi news?")
        print("=" * 68)
        if not corpus:
            print("  no corpus; run scripts/build_corpus.py first")
        else:
            results["retrieval"] = eval_retrieval(
                corpus, n_queries=args.queries, include_dense=not args.no_dense)

    if args.task in ("all", "shortquery"):
        print("\n" + "=" * 68)
        print("EXPERIMENT 1b -- RETRIEVAL with short, realistic queries")
        print("=" * 68)
        if not corpus:
            print("  no corpus; run scripts/build_corpus.py first")
        else:
            results["short_queries"] = eval_short_queries(
                corpus, n_queries=args.queries)

    if args.task in ("all", "crossinfl"):
        print("\n" + "=" * 68)
        print("EXPERIMENT 1c -- WHEN does morphology decide the query?")
        print("=" * 68)
        if not corpus:
            print("  no corpus; run scripts/build_corpus.py first")
        else:
            results["cross_inflection"] = eval_cross_inflection(corpus)

    if args.task in ("all", "summarization"):
        print("\n" + "=" * 68)
        print("EXPERIMENT 2 -- SUMMARIZATION vs XL-Sum gold abstracts")
        print("=" * 68)
        if not corpus:
            print("  no corpus; run scripts/build_corpus.py first")
        else:
            results["summarization"] = eval_summarization(corpus, n=args.summ_n)

    if args.task in ("all", "classification"):
        print("\n" + "=" * 68)
        print("EXPERIMENT 3 -- TOPIC CLASSIFICATION on the MahaNews test split")
        print("=" * 68)
        results["classification"] = eval_classification(limit=args.clf_limit)
        from src import categorize as _c
        if _c.custom_model_available():
            print("\n  --- fine-tuned 3-class head ---\n")
            results["classification_finetuned"] = eval_classification(
                limit=args.clf_limit, model_name=str(config.CUSTOM_TOPIC_DIR))

    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
