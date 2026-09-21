# Project Guide — Marathi News Intelligence

Everything you need to understand the project and demo it confidently.

- **Part 1** — What the project does (read this before the viva)
- **Part 2** — How to run the demo UI
- **Part 3** — What to say while demoing, step by step
- **Part 4** — Questions she may ask, with honest answers
- **Part 5** — Running the full live system
- **Part 6** — Where everything lives

---

# Part 1 — What the project does

## The one-line version

A Marathi news article goes in. Out comes its **category**, a **summary**, the
**people/places/organisations** in it, its **emotional tone**, an **English
translation**, and it becomes **searchable** — all automatically.

## The seven NLP tasks

From your assignment list, this project implements seven, connected into one
pipeline rather than seven separate programs:

| Code | Task | What it actually does here |
|---|---|---|
| **I** | Morphological analyser | Breaks a Marathi word into root + grammar suffixes |
| **D** | Text categorization | Sorts articles into 12 topics (Politics, Sports, Tech…) |
| **E** | Text summarization | Picks the 3 most important sentences |
| **G** | Information extraction | Pulls out people, places, organisations, dates |
| **F** | Sentiment analysis | Scores tone, sentence by sentence |
| **A** | Machine translation | Marathi → English |
| **C** | Information retrieval | Search across 3,000 articles |

## The core idea — why morphology matters

This is the part that makes the project original, so understand it properly.

**Marathi joins a whole phrase into a single word.** English says "from inside
the houses" in four words. Marathi says it in one:

```
घर          house              (the root)
घरात        in the house       घर + आत
घरातून      from inside        घर + आतून
घरांमधून    from inside the houses    घर + ां + मधून
घराच्या     of the house       घर + आ + च्या
```

A normal search engine treats those five as **five unrelated words**. So if
someone searches for `घर`, it finds nothing — even though every one of those
articles is about houses.

**The analyser fixes this** by stripping the suffixes back to the root, so all
five forms get stored under one entry: `घर`.

Two measured results:

- **36%** of all words in the corpus carry at least one strippable suffix
- Normalising them shrinks the search index from **186,593 → 84,480** words
  (**54.7% smaller**)

## Be honest about what it does and doesn't do

Don't oversell this. If she probes and you've overclaimed, it looks bad. The
measured truth:

- On **full-sentence queries**, morphology improves ranking by only **+3.9%**
- On **short 1–2 word queries**, it gives **no improvement at all**
- **Where it genuinely matters:** in **67.8%** of articles, at least one
  headline word appears in the body only in a *different* form. For those
  searches, the plain index returns a **completely empty page 14.4%** of the
  time, versus **1.1%** with morphology

**The honest summary:** morphology is *insurance against finding nothing*, not
a ranking booster. Saying this yourself is much stronger than being caught.

---

# Part 2 — How to run the demo UI

## The easy way

Double-click **`Marathi-NLP-Demo.html`** in the project folder. It opens in
your browser. That's it.

- No Python, no installation, no internet needed (except the first English
  translation — see below)
- Works on any laptop — you can email it to yourself or carry it on a pendrive
- It is **one single file** (about 4 MB) containing everything

> **Tip:** with internet, the page loads proper Marathi typefaces. Offline it
> falls back to your system's Marathi font — still perfectly readable, just
> slightly different. Either is fine for a demo.

## Everything is live — paste any Marathi text

Every tab computes on whatever you type or paste, in your browser:

- **Morphology** — one word shows the root + suffix blocks; paste a sentence
  or a whole article and you get every word's root in a table.
- **Search** — any query; you can also paste your own articles into the index
  ("Add your own articles").
- **Full pipeline** — paste any article (or pick an example) and press
  **Analyse**. Category, summary, entities, sentiment all run on the spot;
  **Translate to English** runs NLLB-200 in the browser.

Be accurate about *which models* run, if asked:

- **Summary (E) and morphology (I)** run the exact same algorithms as Python.
- **Category (D), sentiment (F) and entities (G)** use small in-browser
  models trained to imitate the MahaBERT models (the real ones are ~700 MB
  each). Each stage shows how often it agrees with the real model on 1,362
  held-out articles: topic 83%, article tone 74%, entities F1 0.72.
- **Translation (A)** is the real NLLB-200. The first click downloads it
  (~900 MB, needs internet), then the browser caches it.
- **Want the real models for D, F, G?** Run `python app/server.py` before
  opening the page. The badge next to *Analyse* switches to *Full models ·
  local server* and everything goes through the real pipeline.

`python scripts/verify_demo.py` checks the page against the Python code on
1,000 held-out articles: identical roots, summaries, topics, sentiment and
entities.

---

# Part 3 — What to say while demoing

Three tabs. Aim for about 5 minutes.

## Tab 1 — Morphology (90 seconds)

**Do this:** the word `घरांमधून` is already loaded. Point at the coloured
blocks.

**Say this:**
> "Marathi joins a whole phrase into one word. This is *from inside the
> houses* — a single word. My analyser splits it into the root घर plus a plural
> marker plus an ablative postposition. The teal block is the root."

**Then:** click a few suffix chips. Then **type a Marathi word of your own** —
this is the moment that proves it's real and not hardcoded.

**Point at the bottom cards:**
> "Each group shows different forms of the same word all collapsing to one
> root. That's what makes search work."

## Tab 2 — Search (2 minutes)

**This is your strongest moment. Do it in this exact order.**

**Step 1** — `न्यायालय` (court) is already loaded. Point at both counts.
> "Same query, same 400 articles. Without morphology: 1 result. With it: 14.
> The other 13 articles use the word in a different grammatical form, so a
> normal search can't see them."

**Step 2** — click `पाऊस`. **Both sides show 3.**
> "And here it makes no difference at all. The articles happen to use this word
> in the same form as my query. This happens often, which is why my measured
> average improvement is small — about 4%."

**Why do this deliberately:** you've just shown you tested your own work
honestly. If you only showed `न्यायालय`, and she clicked `पाऊस` herself, it
would look like you were hiding something. Volunteering the weakness is what a
researcher does.

**Step 3** — close it out:
> "So the real benefit isn't better ranking — it's that searches stop returning
> nothing. Empty results drop from 14% to 1%."

## Tab 3 — Full pipeline (2 minutes)

**Do this:** the first example is already analysed. Walk down the stages.
Then **paste an article from today's Marathi news** (copy one from
lokmat.com or bbc.com/marathi) and press **Analyse** — this proves it isn't
stored output.

**Say this:**
> "One article, every stage. **D** — categorised as Politics with 96%
> confidence. **E** — summarised down to the key sentences. **G** — it pulled
> out the people, places and organisations by itself. **F** — tone scored per
> sentence, this one is negative. **A** — translated to English so a
> non-Marathi reader can follow it."

**Point at the small grey label on each stage** ("in-browser · 83%
agreement"): it tells the audience exactly how far the in-browser model can
be trusted. If `python app/server.py` is running, the labels say "full model".

---

# Part 4 — Questions she may ask

**"Did you train any model yourself, or only use ready-made ones?"**
> Both. The morphological analyser is written entirely from scratch — rules,
> no training data. I also fine-tuned MahaBERT for topic classification and got
> **95.4% accuracy**, up from 84.7% without training. The NER, sentiment and
> translation models are pre-trained ones from L3Cube and Meta.

**"Why is your fine-tuned model not used in the pipeline?"**
> Because it scores higher on an *easier* question. It only classifies 3
> categories; the pipeline model handles 12. Switching would collapse Politics,
> Tech and Health all into one label and make the output worse. A better number
> on a simpler task isn't a better component.

*(This is a genuinely strong answer — it shows you understand evaluation.)*

**"How much does the morphological analyser actually help?"**
> Less than I first expected, and I measured it three different ways rather
> than reporting only the flattering one. Ranking improves about 4%. On short
> queries, not at all. Its real value is cutting empty result pages from 14% to
> 1%, plus halving the index size.

**"Is your summarizer better than just taking the first sentences?"**
> No — and I reported that. For news, the opening sentences are a very strong
> baseline because journalists put key facts first. My TextRank scores 0.098
> ROUGE-1 against the baseline's 0.105. I included the baseline specifically so
> the comparison would be honest.

**"Is the web page running the real models?"**
> The morphology, search and summary are the exact same algorithms as the
> Python — checked on 1,000 articles with zero differences. For category,
> sentiment and entities the page uses small models I trained to imitate the
> MahaBERT models, because those are 700 MB each. I measured how often they
> agree: 83% on topic, 74% on article tone, F1 0.72 on entities — and the page
> shows those numbers. With the local server running, it uses the real models.

**"What data did you use?"**
> XL-Sum Marathi — 10,903 BBC Marathi articles with human-written summaries —
> and MahaNews for classification. I processed 3,000 articles. Both are public
> research datasets.

**"What are the weaknesses?"**
> Four real ones. Over-stemming: `नदी` and `नद्या` both become `नद`, which
> isn't a real word — consistent, but not a dictionary form. The NER tagset has
> no B-/I- markers, so two adjacent names of the same type merge. My
> summarization is extractive, not abstractive. And the corpus is all BBC
> Marathi, so the topic mix leans international and the tone leans negative.

**"Can it work on today's news?"**
> Yes — there's a live RSS module that pulls from BBC Marathi, ABP Majha and
> Lokmat and runs the full pipeline on them. *(Only say this if you can
> actually run Part 5.)*

---

# Part 5 — Running the full live system

Only needed if you want to process **new** text live. Requires Python and
about 6 GB of model downloads on first run.

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/build_corpus.py --limit 3000
python scripts/build_index.py --plain-bm25
streamlit run app/dashboard.py
```

The Streamlit dashboard has an **Analyse** tab where you can paste any Marathi
text, or fetch today's live news, and watch the models process it.

**Or keep the HTML page and give it the real models:**

```bash
python app/server.py        # leave running, then open Marathi-NLP-Demo.html
```

**Recommended demo setup:** lead with the HTML file (fast, reliable, can't
break), then start the server or open Streamlit to show the full models.

To reproduce the numbers:

```bash
python scripts/evaluate.py --task all     # all five experiments
python -m pytest tests/ -q                # 51 tests
python scripts/verify_demo.py             # browser/Python parity check
```

---

# Part 6 — Where everything lives

```
Marathi-NLP-Demo.html      ← the demo — just open this
GUIDE.md                   ← this file
README.md                  ← full technical report with all results

src/
  morph.py                 (I) the analyser — written from scratch
  categorize.py            (D) topic classification
  summarize.py             (E) summarization
  ner.py                   (G) entity extraction
  sentiment.py             (F) sentiment
  translate.py             (A) Marathi → English
  retrieve.py              (C) search
  ingest.py                live news from RSS
  pipeline.py              runs all stages on one article
  lite.py                  small in-browser models (D, F, G) for the demo page

scripts/
  download_data.py         fetch the datasets
  build_corpus.py          process articles
  build_index.py           build the search index
  train_classifier.py      fine-tune the classifier (95.4%)
  evaluate.py              all experiments
  distill_label.py         label XL-Sum with the full models
  train_lite.py            train the in-browser models on those labels
  export_demo.py           export data for the HTML page
  build_demo_page.py       build the HTML page
  verify_demo.py           check browser matches Python

app/
  dashboard.py             the live Streamlit app
  server.py                local model server the demo page can use
  demo_template.html       source of the demo page (layout)
  demo_engine.js           source of the demo page (NLP code, run in the browser)
  lite_models.json         weights of the in-browser models

tests/test_pipeline.py     51 tests
```

## If you change the demo page

Edit `app/demo_template.html` or `app/demo_engine.js`, then:

```bash
python scripts/build_demo_page.py     # writes Marathi-NLP-Demo.html
python scripts/verify_demo.py         # must still say PARITY OK
```

To retrain the in-browser models (about 40 minutes on a laptop GPU):

```bash
python scripts/distill_label.py
python scripts/train_lite.py
python scripts/export_demo.py
python scripts/build_demo_page.py
```

---

## Quick reference card

| | |
|---|---|
| **Open the demo** | double-click `Marathi-NLP-Demo.html` |
| **Best moment** | Search tab: `न्यायालय` → 1 vs 14 |
| **Honesty moment** | Search tab: `पाऊस` → 3 vs 3, say so out loud |
| **Prove it's live** | paste today's Marathi news into Full pipeline → Analyse |
| **Real models on the page** | `python app/server.py`, then open the page |
| **Numbers to remember** | 7 tasks · 3,000 articles · 36% inflected · 54.7% smaller index · 95.4% classifier |
