# -*- coding: utf-8 -*-
"""
Fine-tune MahaBERT for Marathi news topic classification.

Trains `l3cube-pune/marathi-bert-v2` on the MahaNews headline split
(entertainment / sports / state) and writes the result to models/topic-clf,
which src/categorize.py then picks up automatically.

    python scripts/train_classifier.py --epochs 3

Written as a plain PyTorch loop rather than transformers.Trainer: it avoids
the `accelerate` dependency, keeps the mixed-precision and gradient-clipping
behaviour explicit, and is small enough to read. Mixed precision plus a batch
size of 16 keeps a 110M-parameter BERT inside 4 GB of VRAM.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config  # noqa: E402

LABELS = ["entertainment", "sports", "state"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts, self.labels = texts, labels
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len,
                       padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def load_split(path: Path):
    import pandas as pd
    df = pd.read_parquet(path)
    # The parquet ships integer labels; config.MAHANEWS_LABELS maps them to
    # the names used everywhere else in the project.
    texts = df["text"].astype(str).tolist()
    labels = [LABEL2ID[config.MAHANEWS_LABELS[int(x)]] for x in df["label"]]
    return texts, labels


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, golds = [], []
    for batch in loader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device, dtype=torch.float16,
                            enabled=(device == "cuda")):
            logits = model(**batch).logits
        preds.extend(logits.float().argmax(-1).cpu().tolist())
        golds.extend(labels.tolist())
    return np.array(preds), np.array(golds)


def report(preds, golds):
    from sklearn.metrics import classification_report, confusion_matrix
    acc = float((preds == golds).mean())
    rep = classification_report(golds, preds, target_names=LABELS,
                                digits=4, zero_division=0, output_dict=True)
    print(classification_report(golds, preds, target_names=LABELS,
                                digits=4, zero_division=0))
    print("confusion matrix (rows = gold):")
    print(confusion_matrix(golds, preds))
    return acc, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="subsample for a quick run")
    ap.add_argument("--out", default=str(config.CUSTOM_TOPIC_DIR))
    args = ap.parse_args()

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = config.DEVICE
    tok = AutoTokenizer.from_pretrained(config.MODEL_BASE_BERT)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_BASE_BERT, num_labels=len(LABELS),
        id2label={i: l for i, l in enumerate(LABELS)}, label2id=LABEL2ID,
    ).to(device)

    tr_x, tr_y = load_split(config.MAHANEWS_TRAIN)
    te_x, te_y = load_split(config.MAHANEWS_TEST)
    if args.limit:
        tr_x, tr_y = tr_x[:args.limit], tr_y[:args.limit]
    print(f"train={len(tr_x)}  test={len(te_x)}  device={device}")

    train_dl = DataLoader(NewsDataset(tr_x, tr_y, tok, args.max_len),
                          batch_size=args.batch_size, shuffle=True)
    test_dl = DataLoader(NewsDataset(te_x, te_y, tok, args.max_len),
                         batch_size=args.batch_size)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = len(train_dl) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.1, anneal_strategy="linear")
    scaler = torch.amp.GradScaler(device, enabled=(device == "cuda"))

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.float16,
                                enabled=(device == "cuda")):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_dl)} "
                      f"loss={running / step:.4f} ({time.time() - t0:.0f}s)", flush=True)

        preds, golds = evaluate(model, test_dl, device)
        print(f"\nepoch {epoch}: test accuracy = {(preds == golds).mean():.4f}\n")

    print("=" * 60)
    acc, rep = report(*evaluate(model, test_dl, device))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "metrics.json").write_text(
        json.dumps({"accuracy": acc, "report": rep,
                    "epochs": args.epochs, "lr": args.lr,
                    "base_model": config.MODEL_BASE_BERT}, indent=2),
        encoding="utf-8")
    print(f"\nsaved -> {out}  (accuracy {acc:.4f}, {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
