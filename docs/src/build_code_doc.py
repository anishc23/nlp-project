# -*- coding: utf-8 -*-
"""
Build the source-code document (docs/Source-Code.pdf).

    python docs/src/build_code_doc.py     # writes docs/src/code.html
    cd docs/src && node render_code.js    # prints it to ../Source-Code.pdf

A short document: the front matter, the theory, and the *core* code of each
of the seven NLP tasks -- the functions that do the actual work, not every
file in the repository. Long docstrings are collapsed to their first line and
skipped regions are marked, so the listing stays readable at ~20 pages.

Pass --full to list the complete files instead.
"""
from __future__ import annotations

import argparse
import ast
import html
import re
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_for_filename

ROOT = Path(__file__).resolve().parent.parent.parent
PROJECT = ROOT / "Marathi-news-intelligence-pipeline"
OUT = Path(__file__).resolve().parent / "code.html"

# (file, note, [symbols to show]) -- empty list means the whole file.
SECTIONS = [
    ("I — Morphological analyser (written from scratch)", [
        ("src/morph.py",
         "Strips the suffix chain layer by layer, counting aksharas rather than characters, "
         "and returns the root plus the grammatical features removed. Everything else in the "
         "project consumes these roots.",
         ["akshara_len", "_strip_once", "_restore_stem", "analyse", "stem",
          "tokenize", "normalize_tokens"]),
    ]),
    ("C — Information retrieval", [
        ("src/retrieve.py",
         "BM25 implemented directly so the tokenisation path can be switched between raw words "
         "and analyser roots — that switch is the project's central experiment. Dense retrieval "
         "and rank fusion complete the hybrid retriever.",
         ["BM25Index", "reciprocal_rank_fusion", "HybridRetriever"]),
    ]),
    ("E — Text summarisation", [
        ("src/summarize.py",
         "TextRank: a sentence-similarity graph built from morphologically normalised TF-IDF, "
         "ranked by PageRank, then de-duplicated with MMR.",
         ["_tfidf_matrix", "_cosine_graph", "_pagerank", "_mmr", "summarize"]),
    ]),
    ("D — Text categorisation", [
        ("src/categorize.py",
         "The 12-topic MahaBERT classifier. active_model() documents why the better-scoring "
         "fine-tuned head is deliberately not the default.",
         ["active_model", "classify"]),
    ]),
    ("G — Information extraction", [
        ("src/ner.py",
         "MahaNER emits flat labels, so spans are merged here, snapped back to whole words "
         "(wordpiece tokenisers split Devanagari mid-letter), and grouped by de-inflected root.",
         ["Entity", "_snap_to_word", "extract", "summarize_entities"]),
    ]),
    ("F — Sentiment analysis", [
        ("src/sentiment.py",
         "Scored per sentence, then aggregated into a document tone by length-weighted expected "
         "polarity, and again per entity.",
         ["classify_sentences", "analyse_document", "entity_sentiment"]),
    ]),
    ("A — Machine translation", [
        ("src/translate.py",
         "NLLB-200, applied sentence by sentence so long articles stay within the model's "
         "reliable length.",
         ["_target_lang_id", "translate", "translate_document"]),
    ]),
    ("The pipeline and its evaluation", [
        ("src/pipeline.py",
         "Runs every stage over one article and returns a single enriched record.",
         ["process"]),
        ("scripts/evaluate.py",
         "The morphology ablation: the same queries against the same corpus, indexed with and "
         "without the analyser.",
         ["eval_retrieval", "rouge_n"]),
        ("tests/test_pipeline.py",
         "One of the five test classes: the analyser's rules, including the guarantee that every "
         "inflected form of a word reduces to a single root.",
         ["TestMorphology"]),
    ]),
    ("The browser demo", [
        ("src/lite.py",
         "Small linear models trained to imitate the MahaBERT topic, sentiment and NER models, so "
         "the demo page can analyse pasted text with no Python. The features below are ported to "
         "JavaScript in app/demo_engine.js and checked against this Python on 1,000 articles.",
         ["sentiment_features", "ner_features", "analyse"]),
        ("app/server.py",
         "Optional local server: when it runs, the demo page sends pasted text to the full models "
         "instead of the students above.",
         ["analyse", "translate"]),
    ]),
]

FULL_LISTING_NOTE = (
    "This listing shows the core of each task. The complete project — 5,830 lines across "
    "28 files, including the data-preparation scripts, the Streamlit dashboard, the demo "
    "page's JavaScript engine and all 59 tests — is at "
    "github.com/anishc23/nlp-project.")

THEORY = """
<h2>4. Theory</h2>

<p><b>Natural Language Processing (NLP)</b> is the branch of artificial intelligence that lets
computers work with human language. This project applies seven standard NLP tasks to Marathi
news, a language with far fewer ready-made tools than English.</p>

<h3>4.1 Morphological analysis (task I)</h3>
<p>Marathi is <b>agglutinative</b>: grammar is added by joining suffixes to a root, so what
English writes as several words becomes one word.</p>
<pre class="plain">घर        house                       (root)
घरात      = घर + आत          in the house
घरातून    = घर + आतून        from inside the house
घरांमधून  = घर + ां + मधून   from inside the houses
घराच्या   = घर + आ + च्या     of the house</pre>
<p>A whitespace tokeniser treats these as five unrelated words, which harms every component that
compares words. A <b>morphological analyser</b> reduces each form to its root. This one is
<b>rule-based</b>: it strips suffixes in four ordered layers (clitic, postposition, verb ending,
case and number), then repairs the stem. Length guards count <b>aksharas</b> (written syllables),
not Unicode code points, because Devanagari vowel signs are separate code points. In the
3,000-article corpus, 36% of tokens carry a removable suffix, and normalising them shrinks the
index vocabulary from 186,593 to 84,480 types (54.7% smaller).</p>

<h3>4.2 Information retrieval (task C)</h3>
<p><b>BM25</b> ranks a document higher when it contains the query terms often, when those terms
are rare in the collection, and it penalises length. <b>Dense retrieval</b> embeds text as
vectors with a neural encoder and ranks by cosine similarity, matching paraphrases that share no
words. <b>Reciprocal Rank Fusion</b> merges the two ranked lists by rank rather than score,
because the two scores are on different scales. Quality is reported as recall@k and <b>MRR</b>.</p>

<h3>4.3 Text summarisation (task E)</h3>
<p><b>TextRank</b> is extractive and graph-based: sentences are nodes, edges are TF-IDF cosine
similarity, and <b>PageRank</b> scores how central each sentence is. <b>Maximal Marginal
Relevance</b> then picks the final sentences, trading relevance against redundancy. Summaries
are scored with <b>ROUGE</b>, the word overlap with human-written summaries.</p>

<h3>4.4 Categorisation, extraction and sentiment (tasks D, G, F)</h3>
<p>These use <b>BERT</b>-family transformers pre-trained on Marathi (MahaBERT, L3Cube Pune).
Categorisation predicts one of 12 topics; <b>named entity recognition</b> is token
classification (each token labelled as part of a person, location, organisation, date…);
sentiment classifies polarity per sentence, aggregated into a document tone. <b>Fine-tuning</b>
a pre-trained model on the task's own data raised topic accuracy from 0.847 to 0.954.</p>

<h3>4.5 Translation and distillation (task A)</h3>
<p>Translation uses <b>NLLB-200</b>, a multilingual sequence-to-sequence transformer, sentence by
sentence. For the browser demo, <b>knowledge distillation</b> was used: small linear
<b>student</b> models were trained on the large models' own output over 12,262 articles, and
agree with them on 83.0% of topics and 75.7% of sentence sentiment labels (F1 0.72 on entities)
while fitting in a few megabytes.</p>

<h3>4.6 System design</h3>
<p>The seven tasks form one pipeline. The analyser runs first because retrieval, summarisation
and entity counting all consume its roots; the enriched articles are then indexed for search.
Every claim is measured against a simple baseline, and negative results are reported too.</p>
"""

ELLIPSIS = "# ...\n"


def collapse_docstrings(src: str) -> str:
    """Shorten multi-line docstrings to their first line, keeping the code."""
    out, lines, i = [], src.split("\n"), 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(\s*)(?:[rRfFbB]*)("""|\'\'\')(.*)$', line)
        if not m or m.group(2) in m.group(3):          # not a docstring opener
            out.append(line)
            i += 1
            continue
        indent, quote, rest = m.groups()
        body = [rest]
        j = i + 1
        while j < len(lines) and quote not in lines[j]:
            body.append(lines[j])
            j += 1
        first = next((b.strip() for b in body if b.strip()), "")
        out.append(f'{indent}{quote}{first} ...{quote}' if first else f'{indent}{quote}{quote}')
        i = j + 1
    return "\n".join(out)


def select(path: Path, symbols: list[str]) -> str:
    """The named top-level definitions, in file order, with gaps marked."""
    src = path.read_text(encoding="utf-8")
    if not symbols or path.suffix != ".py":
        return src
    tree = ast.parse(src)
    lines = src.split("\n")
    chunks, wanted = [], set(symbols)
    for node in tree.body:
        name = getattr(node, "name", None)
        if name in wanted:
            start = min([node.lineno] + [d.lineno for d in
                                         getattr(node, "decorator_list", [])]) - 1
            chunks.append("\n".join(lines[start:node.end_lineno]))
    return ELLIPSIS + f"\n{ELLIPSIS}".join(chunks) + f"\n{ELLIPSIS}"


def file_block(rel: str, note: str, symbols: list[str], full: bool) -> str:
    path = PROJECT / rel
    code = select(path, [] if full else symbols)
    if not full:
        code = collapse_docstrings(code)
    lexer = get_lexer_for_filename(path.name, stripall=False)
    body = highlight(code, lexer, HtmlFormatter(nowrap=True))
    numbered = "".join(f'<span class="ln"></span>{line}\n'
                       for line in body.rstrip("\n").split("\n"))
    shown = "" if full else f'<span class="lines">{len(symbols)} of its definitions</span>'
    return (f'<div class="file"><h3 class="fname">{html.escape(rel)}{shown}</h3>'
            f'<p class="note">{html.escape(note)}</p>'
            f'<pre class="code">{numbered}</pre></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="list complete files")
    args = ap.parse_args()

    blocks = []
    for title, files in SECTIONS:
        blocks.append(f'<h2 class="section">{html.escape(title)}</h2>')
        blocks.extend(file_block(f, n, s, args.full) for f, n, s in files)

    pyg = HtmlFormatter().get_style_defs(".code")
    OUT.write_text(TEMPLATE.replace("__PYGMENTS__", pyg)
                   .replace("__THEORY__", THEORY)
                   .replace("__NOTE__", html.escape(FULL_LISTING_NOTE))
                   .replace("__CODE__", "".join(blocks)), encoding="utf-8")
    shown = sum(f.count("\n") for f in blocks)
    print(f"wrote {OUT}  (~{shown:,} lines shown)")


TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Source Code — Marathi News Intelligence Pipeline</title>
<link rel="stylesheet" href="doc.css">
<style>
__PYGMENTS__
  h2.section{margin-top:20px;break-after:avoid}
  .file{break-inside:auto}
  /* never leave a file's heading or note stranded at the foot of a page */
  .fname,.note{break-after:avoid}
  .fname{font-family:var(--mono);font-size:10pt;color:var(--accent);background:var(--accent-bg);
    border-radius:5px;padding:4px 9px;margin:8px 0 4px;display:flex;justify-content:space-between}
  .fname .lines{color:var(--mute);font-size:7.6pt;font-weight:400}
  .note{font-size:9.2pt;color:var(--soft);margin:0 0 6px}
  pre.code{font-family:var(--mono);font-size:7.2pt;line-height:1.42;background:#fff;
    border:1px solid var(--line);border-radius:6px;padding:7px 9px;margin:0 0 10px;
    white-space:pre-wrap;overflow-wrap:anywhere}
  pre.code .ln{display:inline-block;width:0}
  pre.plain{font-family:var(--deva),var(--mono);font-size:9.3pt;line-height:1.7}
</style></head>
<body>

<div class="kicker">Natural Language Processing · Mini project</div>
<h1>Marathi News Intelligence Pipeline</h1>
<p class="sub">Source code</p>

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
  <tr><td class="k">Language</td><td>Python 3.10 or newer; JavaScript for the in-browser demo engine.</td></tr>
  <tr><td class="k">Libraries</td><td>PyTorch, Hugging Face Transformers, NumPy, pandas, scikit-learn,
      Streamlit, BeautifulSoup, pytest.</td></tr>
  <tr><td class="k">Models</td><td>MahaBERT topic / NER / sentiment / sentence-similarity models
      (L3Cube, Pune) and NLLB-200 distilled 600M (Meta AI).</td></tr>
  <tr><td class="k">Datasets</td><td>XL-Sum Marathi (13,627 BBC Marathi articles with summaries),
      MahaNews (11,721 labelled headlines), and live RSS feeds.</td></tr>
  <tr><td class="k">Hardware</td><td>Any laptop. A GPU (NVIDIA CUDA or Apple Silicon) only makes it
      faster; developed on a 4 GB RTX 3050 and an Apple Silicon Mac.</td></tr>
</table>

__THEORY__

<h2>5. Source code</h2>
<p>__NOTE__ Long docstrings are shortened to one line, and <code># ...</code> marks code left out
between the parts shown.</p>

__CODE__

<div class="foot">Marathi News Intelligence Pipeline · Source code</div>
</body></html>
"""

if __name__ == "__main__":
    main()
