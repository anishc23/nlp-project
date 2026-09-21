# -*- coding: utf-8 -*-
"""
Train the in-browser student models (src/lite.py) on the full models' output.

    python scripts/distill_label.py      # label XL-Sum with the full models
    python scripts/train_lite.py         # fit, score on test, write app/lite_models.json

Each student is a linear softmax model trained against the teacher's
probabilities (soft targets), using torch EmbeddingBag so the sparse features
never become a dense matrix. Weights are then rounded to small integers times
one scale per model -- that is what the page embeds -- and the *rounded*
models are what get scored, so the agreement figures describe exactly what
ships.

Agreement is measured against the teacher on the XL-Sum test split, which the
students never see:

  * topic      -- top-1 label matches the 12-class topic model
  * sentiment  -- sentence label matches the sentiment model; also the
                  document tone after the same length-weighted aggregation
  * entities   -- span-level precision / recall / F1 against MahaNER
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config                                              # noqa: E402
from src import lite                                       # noqa: E402

torch.manual_seed(0)


def load(split):
    path = config.PROCESSED / f"distill_{split}.jsonl"
    if not path.exists():
        sys.exit(f"{path} missing -- run scripts/distill_label.py first")
    return [json.loads(l) for l in open(path, encoding="utf-8")]


# --------------------------------------------------------------------------
# Generic sparse softmax regression
# --------------------------------------------------------------------------
def fit_softmax(rows, targets, n_feats, n_classes, *, epochs, lr, l2,
                batch=256):
    """
    rows: list of (feature_ids, feature_values); targets: list of prob lists.
    Returns (W [n_feats x n_classes], b [n_classes]) as float tensors.
    """
    emb = torch.nn.EmbeddingBag(n_feats, n_classes, mode="sum")
    torch.nn.init.zeros_(emb.weight)
    bias = torch.nn.Parameter(torch.zeros(n_classes))
    opt = torch.optim.Adam(list(emb.parameters()) + [bias], lr=lr)
    y_all = torch.tensor(targets, dtype=torch.float32)
    order = list(range(len(rows)))

    for ep in range(epochs):
        g = torch.Generator().manual_seed(ep)
        perm = torch.randperm(len(order), generator=g).tolist()
        total = 0.0
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            ids, vals, offs = [], [], []
            for r in idx:
                offs.append(len(ids))
                ids.extend(rows[r][0])
                vals.extend(rows[r][1])
            logits = emb(torch.tensor(ids, dtype=torch.long),
                         torch.tensor(offs, dtype=torch.long),
                         per_sample_weights=torch.tensor(vals, dtype=torch.float32)) + bias
            loss = -(y_all[idx] * logits.log_softmax(-1)).sum(-1).mean()
            loss = loss + l2 * emb.weight.pow(2).sum() / len(rows)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"    epoch {ep + 1}/{epochs}  loss {total / len(rows):.4f}", flush=True)
    return emb.weight.detach(), bias.detach()


def quantize(W, levels=127):
    """
    int8 weights plus one scale, packed row-major and base64-encoded -- about
    a third of the size of a JSON list of numbers.
    """
    scale = float(W.abs().max()) / levels or 1.0
    q = torch.round(W / scale).clamp(-levels, levels).to(torch.int8)
    return base64.b64encode(q.numpy().tobytes()).decode("ascii"), scale


# --------------------------------------------------------------------------
# Topic
# --------------------------------------------------------------------------
def train_topic(train, max_vocab):
    labels = sorted(train[0]["topic"])
    docs = [lite.topic_terms(r["title"], r["text"]) for r in train]
    df = Counter(t for d in docs for t in set(d))
    vocab = [t for t, c in df.most_common(max_vocab) if c >= 3]
    index = {t: i for i, t in enumerate(vocab)}
    n = len(docs)
    idf = [round(math.log((1 + n) / (1 + df[t])) + 1.0, 4) for t in vocab]

    rows = []
    for d in docs:
        tf = Counter(t for t in d if t in index)
        vec = {t: (1 + math.log(c)) * idf[index[t]] for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        rows.append(([index[t] for t in vec], [v / norm for v in vec.values()]))
    targets = [[r["topic"][l] for l in labels] for r in train]
    print(f"  topic: {len(vocab):,} terms, {len(rows):,} articles")
    W, b = fit_softmax(rows, targets, len(vocab), len(labels),
                       epochs=12, lr=0.05, l2=1e-4)
    Wq, scale = quantize(W)
    return {"labels": labels, "terms": vocab, "idf": idf, "W": Wq,
            "b": [round(float(x), 4) for x in b], "scale": scale,
            "max_tokens": lite.TOPIC_MAX_TOKENS}


# --------------------------------------------------------------------------
# Sentiment
# --------------------------------------------------------------------------
def train_sentiment(train, max_feats):
    labels = ["Negative", "Neutral", "Positive"]
    sents = [s for r in train for s in r["sentences"]]
    feats = [set(lite.sentiment_features(s["s"])) for s in sents]
    cnt = Counter(f for fs in feats for f in fs)
    vocab = [f for f, c in cnt.most_common(max_feats) if c >= 3]
    index = {f: i for i, f in enumerate(vocab)}
    rows = []
    for fs in feats:
        known = sorted(f for f in fs if f in index)
        inv = 1.0 / math.sqrt(len(known)) if known else 0.0
        rows.append(([index[f] for f in known], [inv] * len(known)))
    targets = [[s["sent"][l] for l in labels] for s in sents]
    print(f"  sentiment: {len(vocab):,} features, {len(rows):,} sentences")
    W, b = fit_softmax(rows, targets, len(vocab), len(labels),
                       epochs=8, lr=0.03, l2=1e-4, batch=512)
    Wq, scale = quantize(W)
    return {"labels": labels, "feats": vocab, "W": Wq,
            "b": [round(float(x), 4) for x in b], "scale": scale}


# --------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------
def word_labels(text, spans):
    offs = lite.words(text)
    out = []
    for s, e in offs:
        lab = "O"
        for a, z, l in spans:
            if a <= s and e <= z:
                lab = l
                break
        out.append(lab)
    return [text[s:e] for s, e in offs], out


def train_ner(train, max_feats, keep):
    labels = lite.NER_LABELS
    lid = {l: i for i, l in enumerate(labels)}
    seqs = []
    for r in train:
        for s in r["sentences"]:
            toks, labs = word_labels(s["s"], s["ner"])
            if toks:
                seqs.append((toks, labs))

    # Teacher forcing: previous-label features use the teacher's labels.
    examples, cnt = [], Counter()
    for toks, labs in seqs:
        for i in range(len(toks)):
            prev = labs[i - 1] if i > 0 else "O"
            prev2 = labs[i - 2] if i > 1 else "O"
            fs = lite.ner_features(toks, i, prev, prev2)
            examples.append((fs, lid[labs[i]]))
            cnt.update(fs)
    vocab = [f for f, c in cnt.most_common(max_feats) if c >= 2]
    index = {f: i for i, f in enumerate(vocab)}
    rows = []
    for fs, _ in examples:
        known = [index[f] for f in fs if f in index]
        rows.append((known, [1.0] * len(known)))
    targets = [[1.0 if j == y else 0.0 for j in range(len(labels))] for _, y in examples]
    print(f"  entities: {len(vocab):,} features, {len(rows):,} words, "
          f"{sum(1 for _, y in examples if y):,} inside entities")
    W, b = fit_softmax(rows, targets, len(vocab), len(labels),
                       epochs=6, lr=0.05, l2=1e-5, batch=1024)

    # Keep the features that move the most decisions: weight spread times how
    # often the feature fires. Spread alone keeps rare, memorised features
    # and drops common decisive ones (the weights that say "आहे" is not a
    # name), which halves precision at the same size.
    spread = (W.max(1).values - W.min(1).values)
    freq = torch.tensor([cnt[f] for f in vocab], dtype=torch.float32)
    order = torch.argsort(spread * freq, descending=True)[:keep].tolist()
    order.sort()
    Wk = W[order]
    Wq, scale = quantize(Wk)
    return {"labels": labels, "feats": [vocab[i] for i in order], "W": Wq,
            "b": [round(float(x), 4) for x in b], "scale": scale}


# --------------------------------------------------------------------------
# Scoring the shipped (rounded) models
# --------------------------------------------------------------------------
def score(model: lite.Lite, test):
    top = sum(max(model.topic(r["title"], r["text"]).items(), key=lambda kv: kv[1])[0]
              == max(r["topic"], key=r["topic"].get) for r in test)

    s_ok = s_n = d_ok = 0
    for r in test:
        pols, weights = [], []
        for s in r["sentences"]:
            p = model.sentence_sentiment(s["s"])
            s_ok += max(p, key=p.get) == max(s["sent"], key=s["sent"].get)
            s_n += 1
        # Document tone over the article's sentences, both sides aggregated
        # the same way (length-weighted expected polarity).
        def tone(get):
            num = den = 0.0
            for s in r["sentences"]:
                p = get(s)
                w = max(len(s["s"]), 1)
                num += w * sum(lite.POLARITY[k] * v for k, v in p.items())
                den += w
            pol = num / den if den else 0.0
            return "Positive" if pol > 0.15 else "Negative" if pol < -0.15 else "Neutral"
        d_ok += tone(lambda s: model.sentence_sentiment(s["s"])) == tone(lambda s: s["sent"])

    tp = fp = fn = 0
    for r in test:
        for s in r["sentences"]:
            gold = {tuple(x) for x in s["ner"]}
            pred = set(model.entities(s["s"]))
            tp += len(gold & pred)
            fp += len(pred - gold)
            fn += len(gold - pred)
    p = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return {
        "test_articles": len(test),
        "topic_top1_agreement": round(top / len(test), 4),
        "sentiment_sentence_agreement": round(s_ok / max(s_n, 1), 4),
        "sentiment_document_agreement": round(d_ok / len(test), 4),
        "ner_precision": round(p, 4), "ner_recall": round(rc, 4),
        "ner_f1": round(2 * p * rc / max(p + rc, 1e-9), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic-vocab", type=int, default=20000)
    ap.add_argument("--sent-feats", type=int, default=40000)
    ap.add_argument("--ner-feats", type=int, default=400000)
    ap.add_argument("--ner-keep", type=int, default=40000)
    ap.add_argument("--out", default=str(lite.LITE_MODELS))
    args = ap.parse_args()

    t0 = time.time()
    train = load("train") + load("val")
    test = load("test")
    print(f"{len(train):,} training articles, {len(test):,} test articles")

    data = {
        "topic": train_topic(train, args.topic_vocab),
        "sentiment": train_sentiment(train, args.sent_feats),
        "ner": train_ner(train, args.ner_feats, args.ner_keep),
    }
    print(f"trained ({time.time() - t0:.0f}s); scoring the rounded models on test...")
    data["agreement"] = score(lite.Lite(data), test)
    for k, v in data["agreement"].items():
        print(f"  {k:32s} {v}")

    out = Path(args.out)
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
