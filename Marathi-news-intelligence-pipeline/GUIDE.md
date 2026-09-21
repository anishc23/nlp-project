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

- No Python, no installation, no internet needed
- Works on any laptop — you can email it to yourself or carry it on a pendrive
- It is **one single file** (about 700 KB) containing everything

> **Tip:** with internet, the page loads proper Marathi typefaces. Offline it
> falls back to your system's Marathi font — still perfectly readable, just
> slightly different. Either is fine for a demo.

## What's genuinely live vs. pre-computed

Be accurate about this if asked:

- **Morphology tab and Search tab are computing live.** Type anything and it
  runs in your browser — real code, not a recording.
- **Full pipeline tab shows stored results.** The neural models (BERT, NLLB)
  are far too large to run in a web page, so those outputs were computed by
  the Python pipeline and saved.

The browser analyser was verified against the Python one on **8,772 words with
zero mismatches** (`python scripts/verify_demo.py`), so it is the same
algorithm, not a simplified imitation.

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

**Do this:** pick an article from the dropdown. Walk down the stages.

**Say this:**
> "One article, every stage. **D** — categorised as Politics with 96%
> confidence. **E** — summarised down to the key sentences. **G** — it pulled
> out the people, places and organisations by itself. **F** — tone scored per
> sentence, this one is negative. **A** — translated to English so a
> non-Marathi reader can follow it."

**Switch to a different article** to show it isn't one lucky example.

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
> baseline because journalists put key facts first. My TextRank scores 0.096
> ROUGE-1 against the baseline's 0.105. I included the baseline specifically so
> the comparison would be honest.

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

**Recommended demo setup:** lead with the HTML file (fast, reliable, can't
break), then open Streamlit to prove the models really run.

To reproduce the numbers:

```bash
python scripts/evaluate.py --task all     # all five experiments
python -m pytest tests/ -q                # 44 tests
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

scripts/
  download_data.py         fetch the datasets
  build_corpus.py          process articles
  build_index.py           build the search index
  train_classifier.py      fine-tune the classifier (95.4%)
  evaluate.py              all experiments
  export_demo.py           export data for the HTML page
  build_demo_page.py       build the HTML page
  verify_demo.py           check browser matches Python

app/
  dashboard.py             the live Streamlit app
  demo_template.html       source of the demo page

tests/test_pipeline.py     44 tests
```

## If you change the demo page

Edit `app/demo_template.html`, then:

```bash
python scripts/build_demo_page.py
copy outputs\demo.html Marathi-NLP-Demo.html
```

---

## Quick reference card

| | |
|---|---|
| **Open the demo** | double-click `Marathi-NLP-Demo.html` |
| **Best moment** | Search tab: `न्यायालय` → 1 vs 14 |
| **Honesty moment** | Search tab: `पाऊस` → 3 vs 3, say so out loud |
| **Prove it's live** | type your own word in the Morphology tab |
| **Numbers to remember** | 7 tasks · 3,000 articles · 36% inflected · 54.7% smaller index · 95.4% classifier |
