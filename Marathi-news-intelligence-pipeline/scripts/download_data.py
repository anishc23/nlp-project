# -*- coding: utf-8 -*-
"""
Fetch the two public corpora this project is built on.

    python scripts/download_data.py

  * XL-Sum (Marathi) -- 10,903 / 1,362 / 1,362 BBC Marathi articles with
    human-written abstractive summaries. Used as the document corpus and as
    the gold standard for the summarisation evaluation.
    https://huggingface.co/datasets/csebuetnlp/xlsum

  * MahaNews (headline split, via MTEB) -- 9,673 / 2,048 Marathi headlines
    labelled entertainment / sports / state. Used to fine-tune the topic
    classifier and to score it.
    https://huggingface.co/datasets/mteb/MarathiNewsClassification

Both are downloaded straight over HTTPS rather than through `datasets`, which
keeps the dependency list short and means no Hub authentication is needed.
Nothing here is uploaded; the requests are read-only downloads.
"""
from __future__ import annotations

import sys
import tarfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config  # noqa: E402

XLSUM_URL = ("https://huggingface.co/datasets/csebuetnlp/xlsum/resolve/main/"
             "data/marathi_XLSum_v2.0.tar.bz2")
MAHANEWS = {
    "mahanews_train.parquet": ("https://huggingface.co/datasets/mteb/"
                               "MarathiNewsClassification/resolve/main/"
                               "data/train-00000-of-00001.parquet"),
    "mahanews_test.parquet": ("https://huggingface.co/datasets/mteb/"
                              "MarathiNewsClassification/resolve/main/"
                              "data/test-00000-of-00001.parquet"),
}


def fetch(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  = {dest.name} (already present, {dest.stat().st_size:,} bytes)")
        return
    print(f"  > {dest.name} ...", end="", flush=True)
    urllib.request.urlretrieve(url, dest)
    print(f" {dest.stat().st_size:,} bytes")


def main():
    print("XL-Sum (Marathi)")
    archive = config.RAW / "marathi_xlsum.tar.bz2"
    fetch(XLSUM_URL, archive)
    if not config.XLSUM_TRAIN.exists():
        with tarfile.open(archive) as tf:
            # Flatten: the archive stores ./marathi_{split}.jsonl
            for member in tf.getmembers():
                if member.isfile() and member.name.endswith(".jsonl"):
                    member.name = Path(member.name).name
                    # filter="data" blocks path traversal; older Pythons
                    # without the argument fall back to plain extract.
                    try:
                        tf.extract(member, config.RAW, filter="data")
                    except TypeError:
                        tf.extract(member, config.RAW)
        print("    extracted train/val/test")

    print("\nMahaNews (headline classification)")
    for name, url in MAHANEWS.items():
        fetch(url, config.RAW / name)

    print("\nReady. Next:")
    print("  python scripts/build_corpus.py --limit 3000")
    print("  python scripts/build_index.py --plain-bm25")
    print("  streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
