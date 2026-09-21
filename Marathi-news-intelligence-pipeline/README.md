# Marathi News Intelligence Pipeline

An end-to-end NLP system for Marathi (मराठी) news. One article goes in; a
categorised, summarised, entity-tagged, sentiment-scored and searchable record
comes out.

The project covers **seven** of the listed NLP concepts, wired into a single
pipeline rather than built as seven disconnected demos:

| # | Concept | Where it lives |
|---|---------|----------------|
| **A** | Machine Translation | [`src/translate.py`](src/translate.py) — Marathi → English via NLLB-200 |
| **C** | Information Retrieval | [`src/retrieve.py`](src/retrieve.py) — BM25 + dense + RRF hybrid |
| **D** | Text Categorization | [`src/categorize.py`](src/categorize.py) — 12-class topic model, plus a fine-tuned 3-class head |
| **E** | Text Summarization | [`src/summarize.py`](src/summarize.py) — TextRank / embedding / lead, with MMR |
| **F** | Sentiment Analysis | [`src/sentiment.py`](src/sentiment.py) — sentence-level, aggregated per document and per entity |
| **G** | Information Extraction | [`src/ner.py`](src/ner.py) — 8-type NER with span repair and alias merging |
| **I** | Morphological Analyser | [`src/morph.py`](src/morph.py) — rule-based, and used by C, E, G |

Module **I** is the part that is written from scratch rather than fine-tuned,
and it feeds three of the others: it normalises the retrieval index, builds the
sentence-similarity graph for summarisation, and de-inflects entity mentions so
they can be counted.

---

## Why morphology is the spine of this project

Marathi is agglutinative. A single noun root takes a number suffix, an oblique
stem change, and one or more postpositions, all written as one word:

```
घर        house
घरात      = घर + आत           in the house
घरातून    = घर + आतून         from inside the house
घरांमधून  = घर + ां + मधून    from inside the houses
घराच्या   = घर + आ + च्या     of the house
```

To a whitespace tokeniser these are five unrelated types. In the 3,000-article
corpus used here, **36% of all tokens carry at least one strippable suffix**,
and normalising them collapses the index vocabulary from 186,593 types to
84,480 — **54.7% smaller**.

`src/morph.py` strips the suffix chain layer by layer (clitic → postposition →
verb → case/number), then undoes the oblique stem alternation. One detail
worth calling out: every length guard counts **aksharas**, not Unicode code
points. Devanagari vowel signs are separate code points, so `len("घरा")` is 3
while the word is two aksharas — using `len()` makes the minimum-stem check
fire inconsistently and sends `शाळा` and `शाळेत` to different roots, which
defeats the entire purpose.

```
घरात      -> घर    [LOC]
घरांमधून  -> घर    [ABL+PL.OBL]
शाळेतून   -> शाळ   [ABL]
नद्या     -> नद    [PL]
सरकारने   -> सरकार [-]
```

---

## Quickstart

```bash
pip install -r requirements.txt

python scripts/download_data.py                  # XL-Sum Marathi + MahaNews
python scripts/build_corpus.py --limit 3000      # ~3 min on a laptop GPU
python scripts/build_index.py --plain-bm25       # ~2 min
streamlit run app/dashboard.py
```

Optional:

```bash
python scripts/train_classifier.py --epochs 3    # fine-tune your own topic head
python scripts/evaluate.py --task all            # reproduce the tables below
python -m pytest tests/ -q                       # 51 tests, no GPU needed
RUN_SLOW_TESTS=1 python -m pytest tests/ -q      # + 8 model-backed tests
```

Everything runs on CPU; a GPU only makes it faster. The device is picked
automatically — CUDA, then Apple Silicon (MPS), then CPU — and can be forced
with `MNIP_DEVICE=cpu`. Developed on an RTX 3050 Laptop (4 GB), which is why
models are held in fp16 on CUDA and batches are small.

---

## The demo page

`Marathi-NLP-Demo.html` is a single self-contained file: open it in any
browser, paste Marathi text into any tab, and it runs.

| Stage | What runs in the page |
|---|---|
| I, C, E | The same algorithms as `src/morph.py`, `src/retrieve.py`, `src/summarize.py`, ported to JavaScript |
| D, F, G | Small students (`src/lite.py`) trained to imitate the MahaBERT models |
| A | NLLB-200 itself, via transformers.js — downloaded on first use (~900 MB), then cached |

The students are linear models trained on the full models' outputs over
12,262 XL-Sum articles (`scripts/distill_label.py`, `scripts/train_lite.py`).
How often they agree with the model they imitate, on the 1,362 held-out
test articles:

| Student | Agreement with the full model |
|---|---|
| Topic (12 classes) | 83.0% top-1 |
| Sentiment | 75.7% of sentences, 73.6% of article tones |
| Entities | span F1 0.72 (P 0.70, R 0.74) |

The page shows these figures next to each result. For the full models, run
the local server and reload the page -- it detects the server and sends
pasted text through `src/pipeline.py` instead:

```bash
python app/server.py        # http://127.0.0.1:8765, this machine only
```

`python scripts/verify_demo.py` extracts the engine from the built page and
checks it against the Python on 1,000 held-out articles: roots, summaries,
topic and sentiment probabilities, entity spans and the whole pipeline are
identical.

---

## Architecture

```
                    raw Marathi article
                            │
              ┌─────────────┴──────────────┐
              │   src/morph.py  (I)        │  ← rule-based, no training data
              │   root + features          │
              └─────────────┬──────────────┘
                            │
   ┌──────────┬─────────────┼─────────────┬──────────────┐
   │          │             │             │              │
  (D)        (E)           (G)           (F)            (A)
categorize  summarize      ner        sentiment      translate
12 topics   TextRank    8 entity    sentence-level   NLLB-200
            + MMR        types      → doc → entity   mr→en
   │          │             │             │              │
   └──────────┴─────────────┼─────────────┴──────────────┘
                            │
                  data/processed/corpus.jsonl
                            │
              ┌─────────────┴──────────────┐
              │   src/retrieve.py  (C)     │
              │  BM25(morph) + dense → RRF │
              └─────────────┬──────────────┘
                            │
                   app/dashboard.py
```

### Models

All are public checkpoints, verified against the Hub before use:

| Role | Checkpoint |
|------|-----------|
| Topic (12-class) | `l3cube-pune/marathi-topic-all-doc` |
| NER (8 types) | `l3cube-pune/marathi-ner` |
| Sentiment | `l3cube-pune/marathi-sentiment-md` |
| Sentence embeddings | `l3cube-pune/marathi-sentence-similarity-sbert` |
| Fine-tuning backbone | `l3cube-pune/marathi-bert-v2` |
| Translation | `facebook/nllb-200-distilled-600M` |

### Data

- **XL-Sum Marathi** — 10,903 / 1,362 / 1,362 BBC Marathi articles with
  human-written abstractive summaries. The document corpus, and the gold
  standard for summarisation.
- **MahaNews (headline split)** — 9,673 / 2,048 headlines labelled
  entertainment / sports / state. Used to train and score the topic classifier.

Both download over plain HTTPS; no Hub token needed.

---

## Results

Corpus: 3,000 XL-Sum articles. Reproduce with `python scripts/evaluate.py --task all`.

### Retrieval (C) — the morphology ablation

Known-item retrieval: the query is an article's headline, the target is that
article, and headlines are held out of the indexed text.

| System | R@1 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|
| BM25 (plain tokens) | 0.600 | 0.770 | 0.817 | 0.675 |
| **BM25 + morphology** | **0.623** | **0.810** | 0.830 | **0.702** |
| Dense (MahaBERT-SBERT) | 0.420 | 0.653 | 0.733 | 0.527 |
| Hybrid (RRF) | 0.550 | 0.783 | **0.873** | 0.651 |

Morphology lifts BM25 MRR@10 by **+3.9%**, and halves the index vocabulary.
The hybrid retriever has the best R@10 but a *worse* MRR than BM25 alone —
RRF trades top-1 precision for coverage, which is the right trade for a
dashboard with filters and the wrong one for a single-answer system.

**This gain is smaller than the project's premise suggested, and the next two
tables are an attempt to say honestly why.**

### Retrieval with short queries

| Query length | plain MRR | morph MRR | plain R@10 | morph R@10 |
|---|---|---|---|---|
| 1 term | 0.212 | 0.202 | 0.357 | 0.343 |
| 2 terms | 0.315 | 0.310 | 0.453 | 0.445 |
| 3 terms | 0.418 | **0.432** | **0.608** | 0.595 |

On short queries morphology provides **no reliable benefit**, and at one and
two terms it is marginally *worse*. Two things are happening: a headline word
usually also occurs in the body in the same form, so there is nothing to
rescue; and over-stemming merges genuinely distinct words, which costs
precision. This contradicts the hand-picked `शाळा` example that motivated the
design, and it is the honest headline result.

### When does morphology actually decide a query?

Restricting to single-word queries where the headline word is absent from the
body in its surface form but present as a root:

| System | R@10 | MRR@10 | empty result page |
|---|---|---|---|
| BM25 (plain tokens) | 0.000 | 0.000 | 14.4% |
| BM25 + morphology | 0.303 | 0.135 | **1.1%** |

**Read this table carefully.** The subset is *constructed* so that the plain
index cannot match the target through that term, so `0.000` is definitional,
not an empirical finding. The two numbers that do mean something are:

- **67.8%** of articles contain at least one such headline word, so the
  situation is common rather than exotic; and
- the plain index returns a **completely empty result page** for 14.4% of
  these queries, against 1.1% with morphology.

So the fair summary is: morphology rarely changes the *ranking* much, but it
substantially reduces how often Marathi search returns nothing at all. For a
language with this much inflection, recall insurance is the real benefit —
not the precision gain the project originally assumed.

### Summarization (E)

300 articles, 2-sentence summaries, scored against XL-Sum gold abstracts.

| System | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| lead-2 | 0.1045 | 0.0192 | 0.0817 |
| **lead-2 + morph tokens** | 0.1566 | **0.0300** | **0.1199** |
| TextRank | 0.0981 | 0.0184 | 0.0757 |
| TextRank + morph tokens | **0.1588** | 0.0296 | 0.1154 |

**TextRank does not beat the lead baseline** on raw ROUGE, and on normalised
tokens it edges ahead only on ROUGE-1 (by 0.002) while trailing on ROUGE-2
and ROUGE-L -- effectively a tie. For news this is a well-known
and unsurprising outcome — journalists front-load the important facts — and
the lead baseline is reported here precisely so the comparison is not
flattering. Absolute ROUGE is low because XL-Sum summaries are abstractive
one-liners and these systems are extractive, so there is a hard ceiling.

`+ morph tokens` means ROUGE was computed over morphologically normalised
tokens rather than raw ones: matching `शाळेत` against `शाळांमध्ये` scores 0
on raw ROUGE-1 and correctly scores a match once normalised. It raises every
system by roughly 50%, which is a statement about **how ROUGE should be
computed for Marathi**, not about summary quality.

### Categorization (D)

MahaNews test split, 2,048 headlines.

| System | Accuracy | Macro-F1 |
|---|---|---|
| `marathi-topic-all-doc`, 12-class zero-shot, mapped to 3 classes | 0.8467 | 0.7994 |
| **Fine-tuned MahaBERT, 3-class** (`scripts/train_classifier.py`) | **0.9536** | **0.9380** |

The zero-shot model's weak point is entertainment recall (0.577): the 12→3
label mapping has to fold `Manoranjan` and `Fashion` into one class and guess
where `Bhakti` and `Travel` belong, and that mapping — not the model — is what
loses the accuracy. Fine-tuning a 3-class head directly on MahaNews lifts that
recall to 0.943 and the overall accuracy to 0.954 (3 epochs, 42 min on a 4 GB
RTX 3050).

**The fine-tuned head is not the pipeline default, and the better score is why.**
It wins on this benchmark because MahaNews only asks a 3-way question
(entertainment / sports / state). Promoting it would collapse Politics, Tech,
Health, Crime and the rest into `state` and make every downstream view worse,
so `categorize.active_model()` stays on the 12-class model unless
`MNIP_TOPIC_MODEL=custom` is set. A higher number on an easier task is not a
better component.

---

## Design decisions worth defending

**Rule-based morphology, not a learned lemmatiser.** No Marathi lemmatisation
training data was needed, it runs at ~100k tokens/sec on CPU, and — critically
for the ablation — its output is inspectable. When retrieval improves you can
point at the exact suffix that was stripped.

**Derivational suffixes are off by default.** Stripping `कार` from `सरकार`
("government") yields `सर`, a different word. Derivation is available behind a
flag but is not in the default chain, and the residual-stem floor is higher for
it. The over-stemming visible in the short-query table is what this guard is
holding back.

**BM25 written by hand.** `rank_bm25` would have worked, but the tokenisation
path *is* the experiment — the ablation needs the same index code with one
switch flipped.

**RRF instead of weighted score fusion.** BM25 scores and cosine similarities
live on different, corpus-dependent scales; rank-based fusion needs no
per-corpus tuning.

**Entities grouped by root, displayed by surface form.** De-inflection is a
good equivalence key and a bad label: it correctly merges मुंबई / मुंबईत but
turns शिंदे into शिंद. Each group is therefore shown using its most frequent
surface form.

**Sentiment per sentence, not per article.** The model is trained on
tweet-length text. Document tone is the length-weighted mean of sentence
polarities, using expected polarity rather than the argmax class so a 0.51/0.49
split does not count as strongly negative.

**NER spans snapped to word boundaries.** Wordpiece tokenisers split Devanagari
between a consonant and its vowel sign, which turned `पंतप्रधान` into
`ंतप्रधान`. There is a regression test for this.

---

## Known limitations

- **The morphology gain is modest**, and negative on 1–2 word queries. Its real
  contribution is reducing empty result pages, not improving ranking.
- **Over-stemming.** `नदी` and `नद्या` both become `नद`, which is not a word.
  Roots are consistent equivalence classes, not dictionary lemmas. A lexicon
  check would fix this and is the obvious next step.
- **No B-/I- tags in MahaNER**, so two adjacent entities of the same type merge
  into one span. This is inherent to the tagset.
- **Alias merging is not coreference.** It folds `मोदी` into `नरेंद्र मोदी` by
  suffix match and frequency; it cannot resolve pronouns or nicknames.
- **Extractive summarisation only.** An abstractive model (IndicBART, mT5)
  would score better but needs fine-tuning beyond a 4 GB GPU.
- **Corpus skew.** XL-Sum is BBC Marathi, so the topic mix leans
  International/Politics and the tone leans negative (1,642 of 3,000 articles).
  Conclusions about "Marathi news" generally should not be drawn from it.
- **Translation is one-directional** (mr→en) and not quantitatively evaluated;
  no Marathi–English test set with references was scored.

---

## Repository layout

```
config.py                  paths, model ids, device/batch settings
src/
  morph.py                 (I) morphological analyser  ← written from scratch
  models.py                lazy, cached model loading; fp16 on GPU
  categorize.py            (D) topic classification
  summarize.py             (E) TextRank / embedding / lead + MMR
  ner.py                   (G) entity extraction, span repair, alias merging
  sentiment.py             (F) sentence → document → entity sentiment
  translate.py             (A) NLLB-200 mr→en
  retrieve.py              (C) BM25 + dense + RRF
  ingest.py                live RSS ingestion with article-body extraction
  pipeline.py              per-document orchestrator
  lite.py                  in-browser student models (D, F, G) for the demo page
scripts/
  download_data.py         fetch XL-Sum + MahaNews
  build_corpus.py          enrich articles → corpus.jsonl
  build_index.py           build BM25 (+ plain ablation) and dense indexes
  train_classifier.py      fine-tune MahaBERT on MahaNews
  evaluate.py              all experiments + Marathi-aware ROUGE
  distill_label.py         label XL-Sum with the full models (student targets)
  train_lite.py            train + score the students -> app/lite_models.json
  export_demo.py           demo page data: corpus sample, examples, weights
  build_demo_page.py       assemble Marathi-NLP-Demo.html
  verify_demo.py           JavaScript vs Python parity check
app/
  dashboard.py             Streamlit UI (search, overview, entity tone, morphology, live)
  server.py                local model server used by the demo page
  demo_template.html       demo page layout
  demo_engine.js           demo page NLP engine (port of I, C, E + students)
  lite_models.json         student weights
tests/test_pipeline.py     51 fast tests + 8 model-backed
```

Every module runs standalone for inspection:

```bash
python -m src.morph        # analyser + conflation check
python -m src.ner          # entity extraction demo
python -m src.retrieve     # the plain-vs-morph BM25 contrast
python -m src.ingest       # pull today's Marathi headlines
python -m src.pipeline     # full pipeline on one XL-Sum article
```

---

## Live ingestion

`src/ingest.py` reads public RSS from BBC Marathi, ABP Majha and Lokmat, and
follows each link to pull the article body (longest run of `<p>` text).
Best-effort by design: a failed body fetch degrades to the RSS description
rather than dropping the article. Requests are read-only, one at a time, with
a polite delay.

```bash
python scripts/build_corpus.py --source rss --limit 30
```

---

## Acknowledgements

- **L3Cube, Pune** — MahaBERT, MahaNER, MahaSent, MahaNews
  ([L3Cube-MahaNLP](https://github.com/l3cube-pune/MarathiNLP))
- **CSEBUETNLP** — [XL-Sum](https://huggingface.co/datasets/csebuetnlp/xlsum)
- **Meta AI** — [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M)
