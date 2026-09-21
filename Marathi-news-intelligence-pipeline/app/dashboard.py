# -*- coding: utf-8 -*-
"""
Marathi News Intelligence -- Streamlit dashboard.

    streamlit run app/dashboard.py

Five tabs, one per part of the pipeline:
  Search      -- morphology-aware hybrid retrieval over the enriched corpus (C)
  Overview    -- topic and sentiment distribution, most-covered entities (D/F/G)
  Entity tone -- how a given person/party is being covered (F over G)
  Morphology  -- the analyser, and a side-by-side plain vs morph search (I)
  Analyse     -- run the full pipeline on pasted text or live RSS (A/D/E/F/G)

Heavy objects are cached: the corpus and indexes load once per session, and
each transformer is held by src.models for the process lifetime.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                   # noqa: E402
from src.morph import analyse, normalize_tokens  # noqa: E402
from src.ner import merge_aliases                # noqa: E402
from src.retrieve import BM25Index, HybridRetriever  # noqa: E402

st.set_page_config(page_title="Marathi News Intelligence", page_icon="📰",
                   layout="wide")

TOPIC_COLORS = {
    "Politics": "#c0392b", "Sports": "#27ae60", "Tech": "#2980b9",
    "Manoranjan": "#8e44ad", "Crime": "#7f8c8d", "Health": "#16a085",
    "International": "#d35400", "Education": "#2c3e50", "Auto": "#34495e",
    "Bhakti": "#b7950b", "Fashion": "#e91e63", "Travel": "#00838f",
    "state": "#c0392b", "sports": "#27ae60", "entertainment": "#8e44ad",
}
SENT_ICON = {"Positive": "🟢", "Neutral": "⚪", "Negative": "🔴"}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_corpus():
    if not config.CORPUS_JSONL.exists():
        return []
    with open(config.CORPUS_JSONL, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


@st.cache_resource(show_spinner=False)
def load_retriever():
    return HybridRetriever.load_default()


@st.cache_resource(show_spinner=False)
def load_plain_index(corpus_len: int):
    """Morphology-free index, built lazily for the side-by-side comparison."""
    corpus = load_corpus()
    texts = [f"{r.get('title','')} {r.get('text','')}" for r in corpus]
    return BM25Index(use_morph=False).build(texts, list(range(len(corpus))))


@st.cache_data(show_spinner=False)
def entity_counts():
    """Corpus-wide entity tallies, with short-form aliases folded in."""
    buckets = defaultdict(Counter)
    for r in load_corpus():
        for label, vals in (r.get("entities") or {}).items():
            for v in vals:
                buckets[label][v] += 1
    return {label: merge_aliases(c) for label, c in buckets.items()}


def topic_badge(topic: str) -> str:
    color = TOPIC_COLORS.get(topic, "#555")
    return (f"<span style='background:{color};color:#fff;padding:2px 9px;"
            f"border-radius:11px;font-size:0.76rem;font-weight:600'>{topic}</span>")


def render_article(rec: dict, score: float = None, key_prefix: str = ""):
    """One result card."""
    sent = rec.get("sentiment", "Neutral")
    with st.container(border=True):
        st.markdown(
            f"{topic_badge(rec.get('topic','?'))} &nbsp; "
            f"<b>{rec.get('title','(untitled)')}</b>",
            unsafe_allow_html=True)

        meta = [f"tone **{sent}** ({rec.get('polarity',0):+.2f})"]
        if score is not None:
            meta.append(f"score `{score:.3f}`")
        if rec.get("tokens"):
            meta.append(f"{rec['tokens']} tokens, "
                        f"{rec.get('inflected_ratio',0):.0%} inflected")
        st.caption(" · ".join(meta))

        if rec.get("summary"):
            st.write(rec["summary"])
        if rec.get("summary_en"):
            st.caption("🌐 " + rec["summary_en"])

        ents = rec.get("entities") or {}
        if ents:
            chips = []
            for label in ("Person", "Organization", "Location", "Date", "Designation"):
                for v in (ents.get(label) or [])[:4]:
                    chips.append(f"`{label[:3]}: {v}`")
            if chips:
                st.markdown(" ".join(chips))
        if rec.get("url"):
            st.caption(f"[source]({rec['url']})")


# --------------------------------------------------------------------------
corpus = load_corpus()

st.title("📰 Marathi News Intelligence")
st.caption("Morphology-aware retrieval, categorisation, summarisation, "
           "entity extraction, sentiment and translation over Marathi news.")

if not corpus:
    st.error("No corpus found. Build one first:\n\n"
             "```\npython scripts/build_corpus.py --limit 3000\n"
             "python scripts/build_index.py --plain-bm25\n```")
    st.stop()

tab_search, tab_overview, tab_entity, tab_morph, tab_live = st.tabs(
    ["🔎 Search", "📊 Overview", "🏷️ Entity tone", "🔤 Morphology", "⚡ Analyse"])


# ------------------------------------------------------------------ search --
with tab_search:
    c1, c2, c3 = st.columns([5, 1.4, 1.4])
    query = c1.text_input("Search the corpus (Marathi)", value="शाळा",
                          key="q", label_visibility="collapsed",
                          placeholder="उदा. शाळा, निवडणूक, क्रिकेट")
    mode = c2.selectbox("Mode", ["hybrid", "bm25", "dense"],
                        label_visibility="collapsed")
    top_k = c3.number_input("Top-k", 1, 50, 10, label_visibility="collapsed")

    topics = sorted({r.get("topic", "?") for r in corpus})
    f1, f2 = st.columns(2)
    want_topics = f1.multiselect("Filter by topic", topics, default=[])
    want_sent = f2.multiselect("Filter by tone",
                               ["Positive", "Neutral", "Negative"], default=[])

    if query:
        retriever = load_retriever()
        if retriever.bm25 is None and retriever.dense is None:
            st.warning("No index found. Run `python scripts/build_index.py`.")
        else:
            use_mode = mode
            if mode == "dense" and retriever.dense is None:
                st.info("No dense index; falling back to BM25.")
                use_mode = "bm25"
            hits = retriever.search(query, top_k=int(top_k) * 3, mode=use_mode)

            shown = 0
            for h in hits:
                rec = corpus[h.doc_id]
                if want_topics and rec.get("topic") not in want_topics:
                    continue
                if want_sent and rec.get("sentiment") not in want_sent:
                    continue
                render_article(rec, score=h.score)
                shown += 1
                if shown >= top_k:
                    break
            if shown == 0:
                st.info("No matching documents.")
            else:
                st.caption(f"{shown} result(s) · mode = {use_mode} · "
                           f"query normalised to `{' '.join(normalize_tokens(query))}`")


# ---------------------------------------------------------------- overview --
with tab_overview:
    st.subheader(f"Corpus overview — {len(corpus):,} articles")

    topic_counts = Counter(r.get("topic", "?") for r in corpus)
    sent_counts = Counter(r.get("sentiment", "?") for r in corpus)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Topic distribution** (module D)")
        st.bar_chart(pd.Series(topic_counts).sort_values(ascending=False))
    with c2:
        st.markdown("**Tone distribution** (module F)")
        st.bar_chart(pd.Series(sent_counts))

    st.markdown("**Tone by topic**")
    rows = defaultdict(lambda: {"Positive": 0, "Neutral": 0, "Negative": 0})
    for r in corpus:
        rows[r.get("topic", "?")][r.get("sentiment", "Neutral")] += 1
    st.dataframe(pd.DataFrame(rows).T.sort_values("Negative", ascending=False),
                 use_container_width=True)

    st.markdown("**Most-covered entities** (module G, de-inflected by module I)")
    buckets = entity_counts()
    cols = st.columns(4)
    for col, label in zip(cols, ["Person", "Organization", "Location", "Date"]):
        with col:
            st.caption(label)
            top = buckets[label].most_common(12)
            if top:
                st.dataframe(pd.DataFrame(top, columns=[label, "n"]),
                             hide_index=True, use_container_width=True)

    infl = [r.get("inflected_ratio", 0) for r in corpus if r.get("tokens")]
    if infl:
        st.metric("Mean inflected-token ratio",
                  f"{sum(infl)/len(infl):.1%}",
                  help="Share of tokens carrying at least one strippable "
                       "suffix — the reason plain keyword search underperforms.")


# ------------------------------------------------------------- entity tone --
with tab_entity:
    st.subheader("How is an entity being covered?")
    st.caption("Mean tone of the articles each entity appears in "
               "(module F aggregated over module G).")

    buckets = entity_counts()

    label = st.selectbox("Entity type",
                         ["Person", "Organization", "Location", "Designation"])
    options = [e for e, n in buckets[label].most_common(150) if n >= 2]
    if not options:
        st.info("Not enough repeated entities of this type in the corpus.")
    else:
        ent = st.selectbox("Entity", options)
        mentions = [r for r in corpus
                    if ent in (r.get("entities") or {}).get(label, [])]
        pols = [r.get("polarity", 0.0) for r in mentions]
        mean_pol = sum(pols) / len(pols) if pols else 0.0

        m1, m2, m3 = st.columns(3)
        m1.metric("Articles", len(mentions))
        m2.metric("Mean tone", f"{mean_pol:+.3f}")
        m3.metric("Dominant topic",
                  Counter(r.get("topic", "?") for r in mentions).most_common(1)[0][0])

        st.bar_chart(pd.Series(Counter(r.get("sentiment", "?") for r in mentions)))
        st.markdown("**Articles mentioning this entity**")
        for r in mentions[:15]:
            render_article(r)


# -------------------------------------------------------------- morphology --
with tab_morph:
    st.subheader("Morphological analyser (module I)")

    word = st.text_input("Analyse a Marathi word", value="घरांमधून")
    if word:
        a = analyse(word.strip())
        c1, c2, c3 = st.columns(3)
        c1.metric("Root", a.root)
        c2.metric("Features", "+".join(a.features) or "—")
        c3.metric("Suffixes stripped", " + ".join(a.stripped) or "—")

    st.divider()
    st.markdown("**Why it matters for search**")
    st.caption("The same query run against a morphology-normalised index and a "
               "plain whitespace index over the identical corpus.")

    q = st.text_input("Query", value="शाळा", key="morph_q")
    if q:
        retriever = load_retriever()
        plain = load_plain_index(len(corpus))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("❌ **Plain tokens**")
            hits = plain.search(q, top_k=5)
            st.caption(f"{len(hits)} hit(s) · indexed terms: "
                       f"`{' '.join(normalize_tokens(q, use_morph=False))}`")
            for h in hits:
                st.write(f"• {corpus[h.doc_id].get('title','')[:90]}")
            if not hits:
                st.info("No documents matched.")
        with c2:
            st.markdown("✅ **Morphology-normalised**")
            hits = retriever.bm25.search(q, top_k=5) if retriever.bm25 else []
            st.caption(f"{len(hits)} hit(s) · indexed terms: "
                       f"`{' '.join(normalize_tokens(q))}`")
            for h in hits:
                st.write(f"• {corpus[h.doc_id].get('title','')[:90]}")
            if not hits:
                st.info("No documents matched.")


# ------------------------------------------------------------------- live --
with tab_live:
    st.subheader("Run the full pipeline on new text")
    src = st.radio("Input", ["Paste text", "Fetch live RSS"], horizontal=True)
    do_translate = st.checkbox("Also translate to English (module A, slow)",
                               value=False)

    if src == "Paste text":
        title = st.text_input("Headline (optional)")
        body = st.text_area("Marathi article text", height=220)
        if st.button("Analyse", type="primary") and body.strip():
            from src.pipeline import process
            with st.spinner("Running pipeline..."):
                doc = process(title, body, translate=do_translate)
            render_article(doc.to_dict())
            with st.expander("Full topic distribution"):
                st.bar_chart(pd.Series(doc.topic_scores).sort_values(ascending=False))
    else:
        n = st.slider("Articles to fetch", 3, 21, 6, step=3)
        if st.button("Fetch & analyse", type="primary"):
            from src.ingest import fetch_all
            from src.pipeline import process
            with st.spinner("Fetching feeds..."):
                rows = fetch_all(limit=n, with_body=True)
            st.success(f"Fetched {len(rows)} articles")
            bar = st.progress(0.0)
            for i, r in enumerate(rows, 1):
                doc = process(r["title"], r["text"], url=r.get("url", ""),
                              translate=do_translate)
                render_article(doc.to_dict())
                bar.progress(i / len(rows))
