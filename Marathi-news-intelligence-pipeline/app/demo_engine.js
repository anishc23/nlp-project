/*
 * Marathi NLP engine for the demo page -- a JavaScript port of the Python
 * reference implementations, run in the browser on whatever text is pasted.
 *
 *   I  analyse / stem         src/morph.py
 *   C  BM25                   src/retrieve.py
 *   E  TextRank + MMR         src/summarize.py
 *   D  topic student          src/lite.py  (imitates marathi-topic-all-doc)
 *   F  sentiment student      src/lite.py  (imitates marathi-sentiment-md)
 *   G  entity tagger student  src/lite.py  (imitates marathi-ner)
 *
 * build_demo_page.py inlines this file into the page, and verify_demo.py
 * loads this same file in Node and checks it against the Python modules, so
 * what is checked is exactly what ships.
 */
function makeEngine(DATA) {
  "use strict";
  const M = DATA.morph;

  /* ============ I: morphological analyser (src/morph.py) ============ */
  const MARKS = new Set(Array.from(M.marks));
  const FINAL = new Set(Array.from(M.finalMatras));
  const STOP = new Set(M.stopwords);
  const byLen = t => t.slice().sort((a, b) => b[0].length - a[0].length);
  const LAYERS = [byLen(M.clitics), byLen(M.postpositions), byLen(M.verb), byLen(M.case)];
  const aksh = s => { let n = 0; for (const c of s) if (!MARKS.has(c)) n++; return n; };

  function restore(s) {
    while (s.endsWith(M.virama)) s = s.slice(0, -1);
    if (s && FINAL.has(s[s.length - 1]) && aksh(s) >= M.minStem) s = s.slice(0, -1);
    return s;
  }
  function strip1(w) {
    for (const table of LAYERS) for (const [suf, feat, repl] of table) {
      if (!w.endsWith(suf)) continue;
      const c = w.slice(0, w.length - suf.length) + repl;
      if (aksh(c) >= M.minStem) return [c, suf, feat];
    }
    return null;
  }
  const cache = new Map();
  function analyse(word) {
    if (cache.has(word)) return cache.get(word);
    let w = word.normalize("NFC"), res;
    if (Object.prototype.hasOwnProperty.call(M.exceptions, word)) {
      res = { root: M.exceptions[word], features: [], stripped: [] };
    } else {
      const features = [], stripped = [];
      for (let i = 0; i < M.maxStrips; i++) {
        const h = strip1(w); if (!h) break;
        w = h[0]; stripped.push(h[1]); features.push(h[2]);
      }
      res = { root: restore(w), features, stripped };
    }
    cache.set(word, res);
    return res;
  }
  const stem = w => analyse(w).root;
  const tokenize = t => (t || "").match(/[\u0900-\u097F]+|[A-Za-z]+|\d+/g) || [];
  const isAscii = t => /^[\x00-\x7F]+$/.test(t);
  function normTokens(text, useMorph = true) {
    const out = [];
    for (const tok of tokenize(text)) {
      if (STOP.has(tok)) continue;
      const t = useMorph ? stem(tok) : tok;
      if (STOP.has(t)) continue;
      if (aksh(t) >= M.minStem || isAscii(t)) out.push(t);
    }
    return out;
  }
  function sentTokenize(text) {
    return (text || "").trim().split(/(?<=[।.!?])\s+/)
      .map(s => s.trim()).filter(Boolean);
  }

  /* ============ C: BM25 (src/retrieve.py) ============ */
  const K1 = 1.5, B = 0.75;
  function buildIndex(texts, useMorph) {
    const postings = new Map(), len = [];
    texts.forEach((txt, i) => {
      const toks = normTokens(txt, useMorph); len.push(toks.length);
      const tf = new Map();
      for (const t of toks) tf.set(t, (tf.get(t) || 0) + 1);
      for (const [t, f] of tf) {
        if (!postings.has(t)) postings.set(t, []);
        postings.get(t).push([i, f]);
      }
    });
    const avg = len.length ? len.reduce((a, b) => a + b, 0) / len.length : 0;
    return { postings, len, avg, n: texts.length, useMorph };
  }
  function search(idx, q, topK) {
    const sc = new Float64Array(idx.n);
    for (const term of normTokens(q, idx.useMorph)) {
      const p = idx.postings.get(term); if (!p) continue;
      const idf = Math.max(Math.log((idx.n - p.length + 0.5) / (p.length + 0.5) + 1), 0);
      for (const [d, f] of p) {
        const dn = f + K1 * (1 - B + B * idx.len[d] / Math.max(idx.avg, 1e-9));
        sc[d] += idf * (f * (K1 + 1)) / Math.max(dn, 1e-9);
      }
    }
    const hits = [];
    for (let i = 0; i < idx.n; i++) if (sc[i] > 0) hits.push([i, sc[i]]);
    hits.sort((a, b) => b[1] - a[1]);
    return topK ? hits.slice(0, topK) : hits;
  }

  /* ============ E: TextRank + MMR (src/summarize.py) ============ */
  const DAMPING = 0.85, MAX_ITER = 100, TOL = 1e-6;
  function summarize(text, k = 3, lambda = 0.7) {
    const sentences = sentTokenize(text)
      .filter(s => s.split(/\s+/).filter(Boolean).length >= 3);
    if (!sentences.length) return [];
    if (sentences.length <= k) return sentences;

    // morphologically normalised TF-IDF
    const docs = sentences.map(s => normTokens(s, true));
    const vocab = new Map();
    for (const d of docs) for (const t of d) if (!vocab.has(t)) vocab.set(t, vocab.size);
    const n = sentences.length, V = Math.max(vocab.size, 1);
    const X = docs.map(() => new Float64Array(V));
    if (vocab.size) {
      docs.forEach((d, i) => { for (const t of d) X[i][vocab.get(t)] += 1; });
      for (const row of X) {
        let s = 0; for (const v of row) s += v;
        s = Math.max(s, 1e-9);
        for (let j = 0; j < V; j++) row[j] /= s;
      }
      for (let j = 0; j < V; j++) {
        let df = 0; for (const row of X) if (row[j] > 0) df++;
        const idf = Math.log((1 + n) / (1 + df)) + 1;
        for (const row of X) row[j] *= idf;
      }
    }
    // cosine graph
    const unit = X.map(row => {
      let s = 0; for (const v of row) s += v * v;
      const nrm = Math.max(Math.sqrt(s), 1e-9);
      return row.map(v => v / nrm);
    });
    const sim = unit.map((a, i) => unit.map((b, j) => {
      if (i === j) return 0;
      let s = 0; for (let t = 0; t < V; t++) s += a[t] * b[t];
      return Math.max(s, 0);
    }));
    // PageRank by power iteration
    const T = sim.map(row => {
      const r = row.reduce((a, b) => a + b, 0);
      return row.map(v => r > 0 ? v / Math.max(r, 1e-9) : 1 / n);
    });
    let scores = new Array(n).fill(1 / n), done = false;
    for (let it = 0; it < MAX_ITER && !done; it++) {
      const nxt = new Array(n);
      for (let j = 0; j < n; j++) {
        let s = 0; for (let i = 0; i < n; i++) s += T[i][j] * scores[i];
        nxt[j] = (1 - DAMPING) / n + DAMPING * s;
      }
      let diff = 0; for (let j = 0; j < n; j++) diff += Math.abs(nxt[j] - scores[j]);
      scores = nxt;
      if (diff < TOL) done = true;
    }
    // rounded, then stable: exact ties keep document order (see summarize.py)
    const r12 = x => Math.round(x * 1e12) / 1e12;
    const order = scores.map((s, i) => [r12(s), i]).sort((a, b) => b[0] - a[0]).map(x => x[1]);
    // MMR
    const rel = new Map(order.map((idx, pos) => [idx, 1 - pos / Math.max(order.length, 1)]));
    const selected = [], cand = order.slice();
    while (cand.length && selected.length < k) {
      let best = null, bestScore = -Infinity;
      for (const idx of cand) {
        let red = 0;
        for (const j of selected) red = Math.max(red, sim[idx][j]);
        const s = r12(lambda * rel.get(idx) - (1 - lambda) * red);
        if (s > bestScore) { best = idx; bestScore = s; }
      }
      selected.push(best);
      cand.splice(cand.indexOf(best), 1);
    }
    return selected.sort((a, b) => a - b).map(i => sentences[i]);
  }

  /* ============ D / F / G: students (src/lite.py) ============ */
  const L = DATA.lite;
  // int8 weights, row-major, base64 (see quantize() in scripts/train_lite.py)
  function unpack(b64, cols) {
    const bin = atob(b64), rows = [];
    for (let i = 0; i < bin.length; i += cols) {
      const row = new Int8Array(cols);
      for (let j = 0; j < cols; j++) row[j] = (bin.charCodeAt(i + j) << 24) >> 24;
      rows.push(row);
    }
    return rows;
  }
  const POLARITY = { Negative: -1, Neutral: 0, Positive: 1 };
  const softmax = z => {
    const m = Math.max(...z), e = z.map(v => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
  };
  const argmax = z => { let b = 0; for (let j = 1; j < z.length; j++) if (z[j] > z[b]) b = j; return b; };

  const T_ = L.topic, topicRow = new Map(), topicIdf = new Map();
  const topicW = unpack(T_.W, T_.labels.length);
  T_.terms.forEach((t, i) => { topicRow.set(t, topicW[i]); topicIdf.set(t, T_.idf[i]); });
  function topic(title, text) {
    const joined = title ? `${title}. ${text}` : text;
    const terms = normTokens(joined, true).slice(0, T_.max_tokens);
    const tf = new Map();
    for (const t of terms) if (topicRow.has(t)) tf.set(t, (tf.get(t) || 0) + 1);
    const vec = new Map();
    for (const [t, c] of tf) vec.set(t, (1 + Math.log(c)) * topicIdf.get(t));
    let ss = 0; for (const v of vec.values()) ss += v * v;
    const norm = Math.sqrt(ss) || 1;
    const k = T_.labels.length, z = new Array(k).fill(0);
    for (const [t, v] of vec) {
      const row = topicRow.get(t);
      for (let j = 0; j < k; j++) z[j] += row[j] * v / norm;
    }
    const p = softmax(z.map((v, j) => v * T_.scale + T_.b[j]));
    const out = {}; T_.labels.forEach((l, j) => { out[l] = p[j]; });
    return out;
  }

  const S_ = L.sentiment, sentRow = new Map();
  const sentW = unpack(S_.W, S_.labels.length);
  S_.feats.forEach((f, i) => sentRow.set(f, sentW[i]));
  function sentimentFeatures(sentence) {
    const toks = tokenize(sentence), feats = [];
    for (const t of toks) {
      feats.push("w:" + t);
      const r = stem(t);
      if (r !== t) feats.push("r:" + r);
    }
    for (let i = 0; i + 1 < toks.length; i++) feats.push("b:" + toks[i] + "_" + toks[i + 1]);
    return feats;
  }
  function sentenceSentiment(sentence) {
    const feats = [...new Set(sentimentFeatures(sentence))].filter(f => sentRow.has(f)).sort();
    const k = S_.labels.length, z = new Array(k).fill(0);
    const inv = feats.length ? 1 / Math.sqrt(feats.length) : 0;
    for (const f of feats) { const row = sentRow.get(f); for (let j = 0; j < k; j++) z[j] += row[j] * inv; }
    const p = softmax(z.map((v, j) => v * S_.scale + S_.b[j]));
    const out = {}; S_.labels.forEach((l, j) => { out[l] = p[j]; });
    return out;
  }
  function documentSentiment(text) {
    const sents = sentTokenize(text);
    const dist = { Negative: 0, Neutral: 0, Positive: 0 };
    if (!sents.length) return { label: "Neutral", polarity: 0, distribution: dist, sentences: [] };
    const scored = sents.map(s => {
      const p = sentenceSentiment(s);
      let label = S_.labels[0];
      for (const l of S_.labels) if (p[l] > p[label]) label = l;
      let pol = 0; for (const l of S_.labels) pol += POLARITY[l] * p[l];
      return { text: s, label, polarity: pol };
    });
    let num = 0, den = 0;
    for (const s of scored) { const w = Math.max(Array.from(s.text).length, 1); num += s.polarity * w; den += w; }
    const pol = num / den;
    for (const s of scored) dist[s.label]++;
    const label = pol > 0.15 ? "Positive" : pol < -0.15 ? "Negative" : "Neutral";
    return { label, polarity: pol, distribution: dist, sentences: scored };
  }

  const N_ = L.ner, nerRow = new Map();
  const nerW = unpack(N_.W, N_.labels.length);
  N_.feats.forEach((f, i) => nerRow.set(f, nerW[i]));
  const BOUNDARY = new Set(Array.from(" \t\n\r " + "।,.;:!?\"'()[]{}<>%/\\|-–—"));
  function words(text) {
    const out = []; let start = null;
    for (let i = 0; i < text.length; i++) {
      if (BOUNDARY.has(text[i])) { if (start !== null) { out.push([start, i]); start = null; } }
      else if (start === null) start = i;
    }
    if (start !== null) out.push([start, text.length]);
    return out;
  }
  const shape = w => /^[0-9०-९]+$/.test(w) ? "digit" : isAscii(w) ? "latin" : "deva";
  function nerFeatures(toks, i, prev, prev2) {
    const w = toks[i], r = stem(w);
    const p = i > 0 ? toks[i - 1] : "<s>";
    const n = i + 1 < toks.length ? toks[i + 1] : "</s>";
    const n2 = i + 2 < toks.length ? toks[i + 2] : "</s>";
    const f = [
      "bias", "w:" + w, "r:" + r, "s1:" + w.slice(-1), "s2:" + w.slice(-2), "s3:" + w.slice(-3),
      "p3:" + w.slice(0, 3), "sh:" + shape(w), "sw:" + (STOP.has(w) ? "True" : "False"),
      "pw:" + p, "pr:" + stem(p), "nw:" + n, "nr:" + stem(n), "nw2:" + n2,
      "pl:" + prev, "pl2:" + prev + "|" + prev2, "plw:" + prev + "|" + r,
      "pwn:" + p + "|" + w, "wn:" + w + "|" + n,
    ];
    if (i === 0) f.push("first");
    return f;
  }
  function entitySpans(text) {
    const offs = words(text), toks = offs.map(([s, e]) => text.slice(s, e));
    const k = N_.labels.length, spans = [];
    let prev = "O", prev2 = "O";
    for (let i = 0; i < toks.length; i++) {
      const z = N_.b.slice();
      for (const f of nerFeatures(toks, i, prev, prev2)) {
        const row = nerRow.get(f);
        if (row) for (let j = 0; j < k; j++) z[j] += row[j] * N_.scale;
      }
      const lab = N_.labels[argmax(z)];
      if (lab !== "O") {
        const [s, e] = offs[i], last = spans[spans.length - 1];
        if (last && last[2] === lab && s - last[1] <= 1) last[1] = e;
        else spans.push([s, e, lab]);
      }
      prev2 = prev; prev = lab;
    }
    return spans;
  }
  function groupEntities(text, spans) {
    const counts = new Map(), surfaces = new Map();
    for (const [s, e, lab] of spans) {
      const surf = text.slice(s, e).trim(); if (!surf) continue;
      const parts = surf.split(/\s+/);
      const canon = [...parts.slice(0, -1), stem(parts[parts.length - 1])].join(" ");
      const key = lab + " " + canon;
      counts.set(key, (counts.get(key) || 0) + 1);
      if (!surfaces.has(key)) surfaces.set(key, new Map());
      const m = surfaces.get(key); m.set(surf, (m.get(surf) || 0) + 1);
    }
    const mostCommon = m => [...m.entries()].sort((a, b) => b[1] - a[1]);  // stable
    const out = {};
    for (const [key] of mostCommon(counts)) {
      const lab = key.split(" ")[0];
      (out[lab] = out[lab] || []).push(mostCommon(surfaces.get(key))[0][0]);
    }
    return out;
  }

  /* ============ the lite pipeline (mirrors src/pipeline.process) ============ */
  function pipeline(title, text) {
    title = (title || "").trim(); text = (text || "").trim();
    const topicScores = topic(title, text);
    const summarySents = summarize(text, 3);
    const summary = summarySents.join(" ");
    const entSource = `${title}. ${summary}`.trim();
    const entities = groupEntities(entSource, entitySpans(entSource));
    const sent = documentSentiment(summary || text);
    const toks = tokenize(text);
    const inflected = toks.filter(t => analyse(t).stripped.length > 0).length;
    return {
      title, topic_scores: topicScores, summary, summary_sentences: summarySents,
      entities, sentiment: sent.label, polarity: sent.polarity,
      sentiment_distribution: sent.distribution, sentence_sentiment: sent.sentences,
      tokens: toks.length, inflected_ratio: toks.length ? inflected / toks.length : 0,
    };
  }

  return { analyse, stem, tokenize, normTokens, sentTokenize, buildIndex, search,
           summarize, topic, sentenceSentiment, documentSentiment, entitySpans,
           groupEntities, pipeline };
}

if (typeof module !== "undefined" && module.exports) module.exports = { makeEngine };
