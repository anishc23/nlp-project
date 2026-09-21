"""Central configuration for the Marathi News Intelligence Pipeline."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
INDEX = DATA / "index"
OUTPUTS = ROOT / "outputs"
MODELS_LOCAL = ROOT / "models"

for _p in (RAW, PROCESSED, INDEX, OUTPUTS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- models ----
# All verified present on the HuggingFace Hub.
MODEL_TOPIC      = "l3cube-pune/marathi-topic-all-doc"        # 12-class MahaNews
MODEL_NER        = "l3cube-pune/marathi-ner"                  # 8-tag flat NER
MODEL_SENTIMENT  = "l3cube-pune/marathi-sentiment-md"         # Neg/Neu/Pos
MODEL_SBERT      = "l3cube-pune/marathi-sentence-similarity-sbert"
MODEL_BASE_BERT  = "l3cube-pune/marathi-bert-v2"              # backbone we fine-tune
MODEL_TRANSLATE  = "facebook/nllb-200-distilled-600M"         # mar_Deva -> eng_Latn

NLLB_SRC_LANG = "mar_Deva"
NLLB_TGT_LANG = "eng_Latn"

# Our own fine-tuned topic classifier (produced by scripts/train_classifier.py)
CUSTOM_TOPIC_DIR = MODELS_LOCAL / "topic-clf"

# ----------------------------------------------------------------- data ----
XLSUM_TRAIN = RAW / "marathi_train.jsonl"
XLSUM_VAL   = RAW / "marathi_val.jsonl"
XLSUM_TEST  = RAW / "marathi_test.jsonl"
MAHANEWS_TRAIN = RAW / "mahanews_train.parquet"
MAHANEWS_TEST  = RAW / "mahanews_test.parquet"

CORPUS_JSONL = PROCESSED / "corpus.jsonl"        # enriched documents
BM25_INDEX   = INDEX / "bm25.pkl"
DENSE_INDEX  = INDEX / "dense.npz"

# ------------------------------------------------------------- runtime -----
import torch  # noqa: E402
def _pick_device() -> str:
    forced = os.environ.get("MNIP_DEVICE", "").strip().lower()
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon GPU. Kept in fp32 (fp16 is only enabled on CUDA below).
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _pick_device()
# RTX 3050 Laptop = 4 GB VRAM; keep batches small and use fp16 on GPU.
BATCH_SIZE = int(os.environ.get("MNIP_BATCH", 16))
MAX_LEN = 256
FP16 = DEVICE == "cuda"

# The MTEB parquet ships integer labels with no names. This mapping was
# verified by joining the parquet against mlexplorer008/marathi_news_
# classification (same 9,673 rows, but labelled with strings) on the headline
# text: 0->entertainment, 1->sports, 2->state, with only a handful of
# off-diagonal rows caused by duplicate headlines in the source data.
MAHANEWS_LABELS = {0: "entertainment", 1: "sports", 2: "state"}
