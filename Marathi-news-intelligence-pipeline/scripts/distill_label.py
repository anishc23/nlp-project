# -*- coding: utf-8 -*-
"""
Label XL-Sum with the full models, as training data for the in-browser models.

    python scripts/distill_label.py                      # train + val -> fit, test -> score
    python scripts/distill_label.py --limit 500          # quick run

The demo page cannot run the MahaBERT models, so it carries small students
(src/lite.py) trained to reproduce them. This script produces the targets:

  * topic      -- the 12-class topic distribution for every article
  * sentiment  -- the 3-class distribution for each sentence
  * entities   -- MahaNER spans for each sentence

Sentences are capped per article so long articles do not dominate. Output is
one JSON object per article in data/processed/distill_{split}.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config                                   # noqa: E402
from src import categorize, models              # noqa: E402
from src.morph import sent_tokenize             # noqa: E402
from src.ner import _snap_to_word               # noqa: E402

SPLITS = {"train": config.XLSUM_TRAIN, "val": config.XLSUM_VAL,
          "test": config.XLSUM_TEST}


def load(path: Path, limit: int):
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if len(r.get("text", "")) < 200:
                continue
            rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def pick_sentences(text: str, cap: int):
    sents = [s for s in sent_tokenize(text)
             if len(s.split()) >= 3 and len(s) <= 600]
    return sents[:cap]


@torch.no_grad()
def ner_batch(texts, batch_size):
    """Batched MahaNER, returning word-snapped [start, end, label] spans."""
    tok, mod = models.load_token_classifier(config.MODEL_NER)
    id2label = mod.config.id2label
    out = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i + batch_size]
        enc = tok(chunk, return_offsets_mapping=True, truncation=True,
                  max_length=config.MAX_LEN, padding=True, return_tensors="pt")
        offsets = enc.pop("offset_mapping").tolist()
        enc.pop("special_tokens_mask", None)
        enc = {k: v.to(config.DEVICE) for k, v in enc.items()}
        labels = mod(**enc).logits.argmax(-1).cpu().tolist()
        for text, offs, labs in zip(chunk, offsets, labels):
            spans, cur = [], None
            for (s, e), lid in zip(offs, labs):
                if s == e:
                    continue
                lab = id2label[lid]
                if lab == "Other":
                    cur = None
                    continue
                if cur and cur[2] == lab and s - cur[1] <= 1:
                    cur[1] = e
                else:
                    cur = [s, e, lab]
                    spans.append(cur)
            snapped = []
            for s, e, lab in spans:
                s, e = _snap_to_word(text, s, e)
                if snapped and snapped[-1][:2] == [s, e]:
                    continue
                snapped.append([s, e, lab])
            out.append(snapped)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--limit", type=int, default=0, help="articles per split (0 = all)")
    ap.add_argument("--sent-cap", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    for split in args.splits:
        t0 = time.time()
        rows = load(SPLITS[split], args.limit)
        print(f"[{split}] {len(rows)} articles on {config.DEVICE}", flush=True)

        topics = categorize.classify([f"{r['title']}. {r['text']}" for r in rows],
                                     batch_size=args.batch)
        print(f"  topics      ({time.time() - t0:.0f}s)", flush=True)

        sents_per_doc = [pick_sentences(r["text"], args.sent_cap) for r in rows]
        # The title is labelled too: pasted input often starts with one.
        flat = [s for r, ss in zip(rows, sents_per_doc) for s in [r["title"], *ss]]

        sent_scores = []
        tok, mod = models.load_classifier(config.MODEL_SENTIMENT)
        id2label = mod.config.id2label
        order = [id2label[j] for j in range(len(id2label))]
        with torch.no_grad():
            for i in range(0, len(flat), args.batch):
                chunk = flat[i:i + args.batch]
                enc = tok(chunk, padding=True, truncation=True,
                          max_length=config.MAX_LEN, return_tensors="pt").to(config.DEVICE)
                probs = mod(**enc).logits.float().softmax(-1).cpu().tolist()
                sent_scores.extend(
                    [{lab: round(p[j], 4) for j, lab in enumerate(order)} for p in probs])
        print(f"  sentiment   {len(flat)} sentences ({time.time() - t0:.0f}s)", flush=True)

        spans = ner_batch(flat, args.batch)
        print(f"  entities    ({time.time() - t0:.0f}s)", flush=True)

        out = config.PROCESSED / f"distill_{split}.jsonl"
        k = 0
        with open(out, "w", encoding="utf-8") as fh:
            for r, pred, ss in zip(rows, topics, sents_per_doc):
                n = len(ss) + 1
                fh.write(json.dumps({
                    "id": r["id"], "title": r["title"], "text": r["text"],
                    "topic": {lab: round(v, 4) for lab, v in pred.all_scores.items()},
                    "sentences": [{"s": s, "sent": sc, "ner": sp} for s, sc, sp in
                                  zip(flat[k:k + n], sent_scores[k:k + n], spans[k:k + n])],
                }, ensure_ascii=False) + "\n")
                k += n
        print(f"  wrote {out}  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
