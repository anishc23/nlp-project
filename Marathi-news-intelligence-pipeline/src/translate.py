# -*- coding: utf-8 -*-
"""
Marathi -> English machine translation (module A).

Uses NLLB-200 (distilled, 600M). Two jobs in this pipeline:

  1. Render Marathi summaries and headlines for a non-Marathi reader.
  2. Provide the cross-lingual query path -- an English query is *not*
     translated at search time; instead documents carry an English rendering
     of their summary, which the retriever can match against. Translating the
     corpus once is far cheaper than translating every query and gives the
     user something readable in the results.

Long inputs are split on sentence boundaries and translated in batches, since
NLLB degrades badly past its training length.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config                      # noqa: E402
from src import models             # noqa: E402
from src.morph import sent_tokenize  # noqa: E402


def _target_lang_id(tok) -> int:
    """
    Resolve the forced BOS token for the target language.

    transformers moved this around across versions: `lang_code_to_id` was
    removed in 5.x, so fall back to a plain vocabulary lookup.
    """
    code = config.NLLB_TGT_LANG
    mapping = getattr(tok, "lang_code_to_id", None)
    if isinstance(mapping, dict) and code in mapping:
        return mapping[code]
    tid = tok.convert_tokens_to_ids(code)
    if tid is None or tid == tok.unk_token_id:
        raise ValueError(f"Could not resolve NLLB language token {code!r}")
    return tid


@torch.no_grad()
def translate(texts, max_new_tokens: int = 256,
              batch_size: int = None, num_beams: int = 4) -> List[str]:
    """Translate Marathi strings to English."""
    single = isinstance(texts, str)
    if single:
        texts = [texts]
    texts = [t if t and t.strip() else " " for t in texts]

    tok, mod = models.load_translator()
    bos = _target_lang_id(tok)
    bs = batch_size or max(1, (config.BATCH_SIZE // 4))
    out: List[str] = []

    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        enc = tok(chunk, padding=True, truncation=True, max_length=512,
                  return_tensors="pt").to(config.DEVICE)
        gen = mod.generate(**enc, forced_bos_token_id=bos,
                           max_new_tokens=max_new_tokens, num_beams=num_beams)
        out.extend(tok.batch_decode(gen, skip_special_tokens=True))

    return out[0] if single else out


def translate_document(text: str, max_sentences: int = 12) -> str:
    """Sentence-split a long document, translate, and re-join."""
    sents = sent_tokenize(text)[:max_sentences]
    if not sents:
        return ""
    return " ".join(translate(sents))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    demo = [
        "मुख्यमंत्र्यांनी विधानसभेत नवीन शेतकरी कर्जमाफी योजनेची घोषणा केली.",
        "भारतीय क्रिकेट संघाने दुसऱ्या कसोटी सामन्यात इंग्लंडचा पराभव केला.",
    ]
    for src, eng in zip(demo, translate(demo)):
        print(f"  MR: {src}\n  EN: {eng}\n")
