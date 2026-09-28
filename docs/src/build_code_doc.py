# -*- coding: utf-8 -*-
"""
Build the source-code listing document (docs/Source-Code.pdf).

    python docs/src/build_code_doc.py     # writes docs/src/code.html
    cd docs/src && node render_code.js    # prints it to ../Source-Code.pdf

The front matter (aim, objectives, outcomes, requirements, theory) comes from
this file; the code is read from the project itself, so the listing can never
drift out of date with what is committed.
"""
from __future__ import annotations

import html
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_for_filename

ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT = ROOT / "Marathi-news-intelligence-pipeline"
OUT = Path(__file__).resolve().parent / "code.html"

# Order matters: the analyser first, then what builds on it.
SECTIONS = [
    ("Configuration and dependencies", [
        ("config.py", "Paths, model names, device selection and batch settings."),
        ("requirements.txt", "Python packages the project needs."),
    ]),
    ("Core modules — src/", [
        ("src/morph.py", "(I) Morphological analyser. Written from scratch; no training data."),
        ("src/models.py", "Lazy, cached loading of every transformer model."),
        ("src/categorize.py", "(D) Topic categorisation into 12 news topics."),
        ("src/summarize.py", "(E) Extractive summarisation: TextRank over a sentence graph, then MMR."),
        ("src/ner.py", "(G) Entity extraction, span repair and alias merging."),
        ("src/sentiment.py", "(F) Sentence, document and per-entity sentiment."),
        ("src/translate.py", "(A) Marathi to English translation with NLLB-200."),
        ("src/retrieve.py", "(C) BM25, dense retrieval and reciprocal rank fusion."),
        ("src/pipeline.py", "Runs every stage over one article."),
        ("src/ingest.py", "Live news ingestion from RSS feeds."),
        ("src/lite.py", "Small in-browser student models used by the demo page."),
    ]),
    ("Scripts — scripts/", [
        ("scripts/download_data.py", "Downloads the XL-Sum and MahaNews datasets."),
        ("scripts/build_corpus.py", "Runs the pipeline over the corpus."),
        ("scripts/build_index.py", "Builds the BM25 and dense search indexes."),
        ("scripts/train_classifier.py", "Fine-tunes MahaBERT as a topic classifier."),
        ("scripts/evaluate.py", "All experiments, plus a Marathi-aware ROUGE implementation."),
        ("scripts/distill_label.py", "Labels the corpus with the full models, to train the students."),
        ("scripts/train_lite.py", "Trains and scores the in-browser student models."),
        ("scripts/export_demo.py", "Exports the demo page's data."),
        ("scripts/build_demo_page.py", "Assembles the single-file demo page."),
        ("scripts/verify_demo.py", "Checks the page's JavaScript against the Python."),
    ]),
    ("Applications — app/", [
        ("app/dashboard.py", "Streamlit dashboard."),
        ("app/server.py", "Local model server the demo page can use."),
        ("app/demo_engine.js", "The demo page's NLP engine: a JavaScript port of the modules above."),
        ("app/demo_template.html", "The demo page's layout and interface code."),
    ]),
    ("Tests — tests/", [
        ("tests/test_pipeline.py", "51 fast tests plus 8 that load the models."),
    ]),
]

THEORY = """
<h2>4. Theory</h2>

<p><b>Natural Language Processing (NLP)</b> is the branch of artificial intelligence that
lets computers work with human language. This project applies seven standard NLP tasks to
Marathi news, a language with far fewer ready-made tools than English.</p>

<h3>4.1 Morphological analysis</h3>
<p>Marathi is an <b>agglutinative</b> language: grammatical information is added by joining
suffixes to a root, so what English writes as several words becomes one word.</p>
<pre class="plain">घर        house                       (root)
घरात      = घर + आत          in the house
घरातून    = घर + आतून        from inside the house
घरांमधून  = घर + ां + मधून   from inside the houses
घराच्या   = घर + आ + च्या     of the house</pre>
<p>A whitespace tokeniser treats these as five unrelated words, which harms every component
that compares words. A <b>morphological analyser</b> reduces each surface form to its root.
The analyser here is <b>rule-based</b>: it strips the suffix chain in four ordered layers
(clitic, postposition, verb ending, case and number) and then repairs the stem. Length
guards count <b>aksharas</b> (written syllables) rather than Unicode code points, because
Devanagari vowel signs are separate code points. In the 3,000-article corpus, 36% of tokens
carry at least one removable suffix, and normalising them reduces the index vocabulary from
186,593 to 84,480 types (54.7% smaller).</p>

<h3>4.2 Information retrieval</h3>
<p><b>BM25</b> is the standard ranking function for keyword search. A document scores highly
when it contains the query terms often (term frequency), when those terms are rare across the
collection (inverse document frequency), and it is penalised for length. <b>Dense retrieval</b>
instead embeds text as vectors with a neural encoder and ranks by cosine similarity, which can
match paraphrases sharing no words. <b>Reciprocal Rank Fusion</b> combines the two ranked lists
by rank position rather than by score, because BM25 scores and cosine similarities are on
different scales. Ranking quality is reported with recall@k and <b>MRR</b> (mean reciprocal
rank).</p>

<h3>4.3 Text summarisation</h3>
<p><b>TextRank</b> is an extractive, graph-based method. Sentences are nodes; edges are weighted
by TF-IDF cosine similarity; <b>PageRank</b> then scores each sentence by how central it is in
that graph. <b>Maximal Marginal Relevance (MMR)</b> selects the final sentences, trading
relevance against redundancy so near-duplicates are not all chosen. Summaries are scored with
<b>ROUGE</b>, the word overlap with human-written summaries.</p>

<h3>4.4 Classification, extraction and sentiment</h3>
<p>These three use <b>BERT</b>-family transformer models pre-trained on Marathi (MahaBERT,
from L3Cube Pune). <b>Text categorisation</b> predicts one of 12 news topics. <b>Named entity
recognition</b> is token classification: each token is labelled as part of a person, location,
organisation, date and so on. <b>Sentiment analysis</b> classifies polarity; because the model
is trained on short text, it is applied per sentence and aggregated into a document tone.
<b>Fine-tuning</b> (training a pre-trained model further on a task's own data) raised topic
accuracy from 0.847 to 0.954 on a 3-class benchmark.</p>

<h3>4.5 Machine translation</h3>
<p>Translation uses <b>NLLB-200</b>, a multilingual sequence-to-sequence transformer, applied
sentence by sentence so long inputs stay within its reliable length.</p>

<h3>4.6 Knowledge distillation</h3>
<p>The browser demo cannot load 700 MB models, so small linear <b>student</b> models were
trained to reproduce the large models' outputs over 12,262 articles. Trained on the teacher's
probability distributions, they agree with it on 83.0% of topics, 75.7% of sentence sentiment
labels and reach F1 0.72 on entities, while fitting in a few megabytes.</p>

<h3>4.7 System design</h3>
<p>The seven tasks form one pipeline rather than seven programs. The analyser runs first
because retrieval, summarisation and entity counting all consume its roots; the enriched
articles are then indexed for search. Every claim is measured against a simple baseline, and
the negative results are reported alongside the positive ones.</p>
"""


def file_block(rel: str, note: str) -> str:
    path = PROJECT / rel
    code = path.read_text(encoding="utf-8")
    try:
        lexer = get_lexer_for_filename(path.name, stripall=False)
    except Exception:                                     # noqa: BLE001
        lexer = get_lexer_for_filename("x.txt")
    body = highlight(code, lexer, HtmlFormatter(nowrap=True, linenos=False))
    numbered = "".join(
        f'<span class="ln">{i}</span>{line}\n'
        for i, line in enumerate(body.rstrip("\n").split("\n"), 1))
    return (f'<div class="file">'
            f'<h3 class="fname">{html.escape(rel)}'
            f'<span class="lines">{code.count(chr(10)) + 1} lines</span></h3>'
            f'<p class="note">{html.escape(note)}</p>'
            f'<pre class="code">{numbered}</pre></div>')


def main():
    contents = []
    for title, files in SECTIONS:
        items = "".join(f"<li><code>{html.escape(f)}</code> — {html.escape(n)}</li>"
                        for f, n in files)
        contents.append(f"<li><b>{html.escape(title)}</b><ul>{items}</ul></li>")

    blocks = []
    for title, files in SECTIONS:
        blocks.append(f'<h2 class="section">{html.escape(title)}</h2>')
        blocks.extend(file_block(f, n) for f, n in files)

    pyg = HtmlFormatter().get_style_defs(".code")
    OUT.write_text(TEMPLATE.replace("__PYGMENTS__", pyg)
                   .replace("__THEORY__", THEORY)
                   .replace("__CONTENTS__", "".join(contents))
                   .replace("__CODE__", "".join(blocks)), encoding="utf-8")
    total = sum((PROJECT / f).read_text(encoding="utf-8").count("\n") + 1
                for _, fs in SECTIONS for f, _ in fs)
    print(f"wrote {OUT}  ({total:,} lines of source)")


TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Source Code — Marathi News Intelligence Pipeline</title>
<link rel="stylesheet" href="doc.css">
<style>
__PYGMENTS__
  h2.section{break-before:page}
  .file{break-before:page}
  /* the first file of a section shares the section's page */
  h2.section + .file{break-before:auto}
  .fname{font-family:var(--mono);font-size:10.5pt;color:var(--accent);background:var(--accent-bg);
    border-radius:5px;padding:5px 9px;margin:0 0 4px;display:flex;justify-content:space-between}
  .fname .lines{color:var(--mute);font-size:8pt;font-weight:400}
  .note{font-size:9.4pt;color:var(--soft);margin:0 0 7px}
  pre.code{font-family:var(--mono);font-size:7.3pt;line-height:1.45;background:#fff;border:1px solid var(--line);
    border-radius:6px;padding:8px 10px 8px 0;margin:0;white-space:pre-wrap;word-break:break-word;
    break-inside:auto;overflow-wrap:anywhere}
  pre.code .ln{display:inline-block;width:34px;padding-right:9px;margin-right:7px;text-align:right;
    color:#b3c4c9;border-right:1px solid var(--line);user-select:none}
  pre.plain{font-family:var(--deva),var(--mono);font-size:9.4pt;line-height:1.7}
  ul.toc2{font-family:var(--sans);font-size:9.3pt;padding-left:18px}
  ul.toc2 ul{padding-left:16px;margin:3px 0 7px}
  ul.toc2 li{margin:2px 0}
  ul.toc2 code{font-size:8.2pt}
</style></head>
<body>

<div class="kicker">Natural Language Processing · Mini project</div>
<h1>Marathi News Intelligence Pipeline</h1>
<p class="sub">Source code listing</p>

<h2>1. Aim</h2>
<p>Mini Project based on Real Life NLP Application.</p>

<h2>2. Objectives</h2>
<ul>
  <li>To understand natural language processing and to learn how to apply basic algorithms in this field.</li>
  <li>To design and implement applications based on natural language processing.</li>
</ul>
<p><b>Outcomes:</b> Be able to apply NLP techniques to design real world NLP applications such as
machine translation, text categorization, text summarization, information extraction, etc.</p>

<h2>3. Hardware / Software Required</h2>
<table>
  <tr><td class="k">Language</td><td>Python 3.10 or newer (the analyser, pipeline, scripts and apps);
      JavaScript for the in-browser demo engine.</td></tr>
  <tr><td class="k">Libraries</td><td>PyTorch, Hugging Face Transformers, NumPy, pandas, scikit-learn,
      Streamlit, BeautifulSoup, pytest.</td></tr>
  <tr><td class="k">Models</td><td>MahaBERT topic / NER / sentiment / sentence-similarity models
      (L3Cube, Pune) and NLLB-200 distilled 600M (Meta AI).</td></tr>
  <tr><td class="k">Datasets</td><td>XL-Sum Marathi (13,627 BBC Marathi articles with summaries),
      MahaNews (11,721 labelled headlines), and live RSS feeds.</td></tr>
  <tr><td class="k">Hardware</td><td>Any laptop. A GPU (NVIDIA CUDA or Apple Silicon) only makes it
      faster; developed on a 4 GB RTX 3050 and an Apple Silicon Mac.</td></tr>
  <tr><td class="k">Other</td><td>A web browser for the single-file demo page; Node.js only for the
      JavaScript/Python equivalence check.</td></tr>
</table>

__THEORY__

<h2>5. Source code</h2>
<p>The complete source of the project, in the order the pipeline uses it. Generated files
(the built demo page, model weights, downloaded data) are not listed.</p>
<ul class="toc2">__CONTENTS__</ul>

__CODE__

<div class="foot">Marathi News Intelligence Pipeline · Source code listing</div>
</body></html>
"""

if __name__ == "__main__":
    main()
