# -*- coding: utf-8 -*-
"""
Topic categorisation (module D).

Two backends:

  * `l3cube-pune/marathi-topic-all-doc` -- 12-way MahaNews document classifier,
    used out of the box so the pipeline works with no training step.
  * a locally fine-tuned MahaBERT head produced by scripts/train_classifier.py,
    selected with MNIP_TOPIC_MODEL=custom. It is NOT the default -- see
    active_model() for why a better benchmark score would make the pipeline
    worse.

Both expose the same `classify()` signature so callers do not care which one
is active.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                      # noqa: E402
from src import models             # noqa: E402


@dataclass
class TopicPrediction:
    label: str
    score: float
    all_scores: dict


def active_model() -> str:
    """
    The model the pipeline uses by default.

    This deliberately stays on the 12-class topic model even after
    scripts/train_classifier.py has produced a fine-tuned checkpoint. The
    fine-tuned head is trained on MahaNews, whose label space is only
    entertainment / sports / state -- it scores better *on that benchmark*
    (0.954 vs 0.847) precisely because it answers an easier question. Letting
    it take over the pipeline would collapse Politics, Tech, Health, Crime and
    the rest into "state" and make every downstream view worse.

    Set MNIP_TOPIC_MODEL=custom to use the fine-tuned head, which is what the
    classification experiment in scripts/evaluate.py does.
    """
    choice = os.environ.get("MNIP_TOPIC_MODEL", "").lower()
    if choice in ("custom", "finetuned", "local"):
        if (config.CUSTOM_TOPIC_DIR / "config.json").exists():
            return str(config.CUSTOM_TOPIC_DIR)
        raise FileNotFoundError(
            f"MNIP_TOPIC_MODEL={choice} but no checkpoint at "
            f"{config.CUSTOM_TOPIC_DIR}. Run scripts/train_classifier.py first.")
    return config.MODEL_TOPIC


def custom_model_available() -> bool:
    return (config.CUSTOM_TOPIC_DIR / "config.json").exists()


@torch.no_grad()
def classify(texts: List[str], model_name: str = None,
             batch_size: int = None) -> List[TopicPrediction]:
    """Classify a batch of documents into news topics."""
    if isinstance(texts, str):
        texts = [texts]
    model_name = model_name or active_model()
    tok, mod = models.load_classifier(model_name)
    id2label = mod.config.id2label
    bs = batch_size or config.BATCH_SIZE
    results: List[TopicPrediction] = []

    for i in range(0, len(texts), bs):
        chunk = [t if t else " " for t in texts[i:i + bs]]
        enc = tok(chunk, padding=True, truncation=True,
                  max_length=config.MAX_LEN, return_tensors="pt").to(config.DEVICE)
        probs = mod(**enc).logits.float().softmax(-1).cpu()
        for row in probs:
            idx = int(row.argmax())
            results.append(TopicPrediction(
                label=id2label[idx],
                score=float(row[idx]),
                all_scores={id2label[j]: float(row[j]) for j in range(len(row))},
            ))
    return results


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    demo = [
        "भारतीय क्रिकेट संघाने दुसऱ्या कसोटी सामन्यात इंग्लंडचा नऊ गडी राखून पराभव केला.",
        "मुख्यमंत्र्यांनी विधानसभेत नवीन शेतकरी कर्जमाफी योजनेची घोषणा केली.",
        "नव्या स्मार्टफोनमध्ये कृत्रिम बुद्धिमत्तेवर आधारित कॅमेरा देण्यात आला आहे.",
    ]
    print("model:", active_model())
    for t, p in zip(demo, classify(demo)):
        print(f"  {p.label:15s} {p.score:.3f}  |  {t[:55]}")
