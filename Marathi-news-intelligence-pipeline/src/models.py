# -*- coding: utf-8 -*-
"""
Lazy, cached model loading.

Every model in this project is loaded through here so that (a) the Streamlit
app and the batch scripts share one copy per process, and (b) a 4 GB laptop
GPU does not end up holding five models at once. Models are moved to the GPU
on first use and kept in fp16 there; CPU stays fp32 because fp16 matmul on CPU
is slower, not faster.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def _prepare(model):
    model.eval()
    if config.DEVICE == "cuda":
        model = model.half().to("cuda")
    elif config.DEVICE != "cpu":
        model = model.to(config.DEVICE)
    return model


@functools.lru_cache(maxsize=None)
def load_classifier(name: str):
    """Sequence-classification model + tokenizer (topic, sentiment)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    mod = _prepare(AutoModelForSequenceClassification.from_pretrained(name))
    return tok, mod


@functools.lru_cache(maxsize=None)
def load_token_classifier(name: str):
    """Token-classification model + tokenizer (NER)."""
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    mod = _prepare(AutoModelForTokenClassification.from_pretrained(name))
    return tok, mod


@functools.lru_cache(maxsize=None)
def load_encoder(name: str = config.MODEL_SBERT):
    """Bare encoder for sentence embeddings (mean pooling applied by caller)."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    mod = _prepare(AutoModel.from_pretrained(name))
    return tok, mod


@functools.lru_cache(maxsize=None)
def load_translator(name: str = config.MODEL_TRANSLATE):
    """Seq2seq translation model (NLLB-200 distilled)."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, src_lang=config.NLLB_SRC_LANG)
    mod = _prepare(AutoModelForSeq2SeqLM.from_pretrained(name))
    # NLLB ships generation_config.max_length=200, which clashes with the
    # max_new_tokens we pass and prints a warning on every batch.
    mod.generation_config.max_length = None
    return tok, mod


def free():
    """Drop every cached model and empty the CUDA allocator."""
    for fn in (load_classifier, load_token_classifier, load_encoder,
               load_translator):
        fn.cache_clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif config.DEVICE == "mps":
        torch.mps.empty_cache()


@torch.no_grad()
def embed(texts, name: str = config.MODEL_SBERT, batch_size: int = None,
          show_progress: bool = False):
    """
    Mean-pooled, L2-normalised sentence embeddings.

    Implemented directly on top of transformers rather than pulling in
    sentence-transformers, which keeps the dependency list short and avoids a
    version pin against transformers 5.x.
    """
    import numpy as np
    tok, mod = load_encoder(name)
    bs = batch_size or config.BATCH_SIZE
    texts = [t if t else " " for t in texts]
    out = []

    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        enc = tok(chunk, padding=True, truncation=True,
                  max_length=config.MAX_LEN, return_tensors="pt").to(config.DEVICE)
        hidden = mod(**enc).last_hidden_state              # (B, T, H)
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        out.append(pooled.float().cpu().numpy())
        if show_progress and (i // bs) % 20 == 0:
            print(f"    embedded {min(i + bs, len(texts))}/{len(texts)}", flush=True)

    return np.vstack(out) if out else np.zeros((0, 768), dtype="float32")
