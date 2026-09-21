# -*- coding: utf-8 -*-
"""
Check that the demo page's JavaScript gives the same answers as the Python.

    python scripts/verify_demo.py            (needs node on PATH)
    python scripts/verify_demo.py --n 1000   # more test articles

The page re-implements the analyser, TextRank and the three in-browser
students (src/lite.py) in JavaScript. That port is only trustworthy if it
agrees with the Python, so this script pulls the engine and the embedded
payload *out of the built page itself* -- i.e. exactly what ships -- runs
them under Node, and compares, stage by stage:

  I  roots of every Devanagari word in the page's search corpus
  E  TextRank summaries of held-out XL-Sum test articles
  D  topic probabilities                    (same label, |diff| < 1e-6)
  F  sentence sentiment probabilities       (same label, |diff| < 1e-6)
  G  entity spans and the grouped entity lists
     and the whole pipeline() output
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config                                 # noqa: E402
from src import lite                          # noqa: E402
from src.morph import sent_tokenize, stem     # noqa: E402
from src.summarize import summarize           # noqa: E402

PAGE = config.ROOT / "Marathi-NLP-Demo.html"
TOL = 1e-6

RUNNER = r"""
const fs = require("fs"), vm = require("vm");
const [enginePath, payloadPath, casesPath] = process.argv.slice(2);
const ctx = { atob: s => Buffer.from(s, "base64").toString("latin1") };
vm.createContext(ctx);
vm.runInContext(fs.readFileSync(enginePath, "utf8") + "\nthis.makeEngine = makeEngine;", ctx);
const E = ctx.makeEngine(JSON.parse(fs.readFileSync(payloadPath, "utf8")));
const C = JSON.parse(fs.readFileSync(casesPath, "utf8"));
const out = {
  stems: C.words.map(w => E.stem(w)),
  summaries: C.docs.map(d => E.summarize(d.text, 3).join(" ")),
  topics: C.docs.map(d => E.topic(d.title, d.text)),
  sentiment: C.sentences.map(s => E.sentenceSentiment(s)),
  spans: C.sentences.map(s => E.entitySpans(s).map(([a, b, l]) => [s.slice(a, b), l])),
  pipeline: C.docs.map(d => {
    const r = E.pipeline(d.title, d.text);
    return { topic_scores: r.topic_scores, summary: r.summary, entities: r.entities,
             sentiment: r.sentiment, polarity: r.polarity,
             sentiment_distribution: r.sentiment_distribution };
  }),
};
process.stdout.write(JSON.stringify(out));
"""


def extract(html: str):
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    if not m:
        sys.exit("no embedded payload found in the page")
    payload = json.loads(m.group(1).replace("<" + chr(92) + "/", "</"))
    engines = [s for s in re.findall(r"<script>(.*?)</script>", html, re.S)
               if "function makeEngine" in s]
    if len(engines) != 1:
        sys.exit("could not find the inlined engine in the page")
    return payload, engines[0]


def close(a: dict, b: dict) -> bool:
    return (max(a, key=a.get) == max(b, key=b.get)
            and all(abs(a[k] - b[k]) < TOL for k in a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="held-out test articles")
    args = ap.parse_args()

    if not PAGE.exists():
        sys.exit(f"{PAGE} not found -- run scripts/build_demo_page.py first")
    payload, engine = extract(PAGE.read_text(encoding="utf-8"))
    model = lite.Lite(payload["lite"])

    words = set()
    for rec in payload["docs"]:
        words.update(re.findall(r"[ऀ-ॿ]+", rec["title"] + " " + rec["summary"]))
    words = sorted(words)

    docs = []
    with open(config.XLSUM_TEST, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if len(r["text"]) >= 200:
                docs.append({"title": r["title"], "text": r["text"]})
            if len(docs) >= args.n:
                break
    sentences = [s for d in docs for s in [d["title"], *sent_tokenize(d["text"])[:6]]]
    print(f"checking {len(words):,} words, {len(docs)} articles, "
          f"{len(sentences):,} sentences")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "engine.js").write_text(engine, encoding="utf-8")
        (td / "payload.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        (td / "cases.json").write_text(json.dumps(
            {"words": words, "docs": docs, "sentences": sentences}, ensure_ascii=False),
            encoding="utf-8")
        (td / "run.js").write_text(RUNNER, encoding="utf-8")
        try:
            r = subprocess.run(["node", str(td / "run.js"), str(td / "engine.js"),
                                str(td / "payload.json"), str(td / "cases.json")],
                               capture_output=True, text=True, encoding="utf-8")
        except FileNotFoundError:
            sys.exit("node not found on PATH -- install Node.js to run this check")
    if r.returncode != 0:
        sys.exit("node failed:\n" + r.stderr)
    js = json.loads(r.stdout)

    checks = {}

    def record(name, bad, total, example=None):
        checks[name] = (bad, total)
        if bad and example:
            print(f"    first mismatch in {name}: {example}")

    bad = [(w, j) for w, j in zip(words, js["stems"]) if stem(w) != j]
    record("I  roots", len(bad), len(words), bad[:1])

    py_sum = [summarize(d["text"], k=3).text for d in docs]
    bad = [i for i, (p, j) in enumerate(zip(py_sum, js["summaries"])) if p != j]
    record("E  summaries", len(bad), len(docs), bad[:1])

    bad = [i for i, (d, j) in enumerate(zip(docs, js["topics"]))
           if not close(model.topic(d["title"], d["text"]), j)]
    record("D  topic", len(bad), len(docs), bad[:1])

    bad = [s for s, j in zip(sentences, js["sentiment"])
           if not close(model.sentence_sentiment(s), j)]
    record("F  sentiment", len(bad), len(sentences), bad[:1])

    bad = []
    for s, j in zip(sentences, js["spans"]):
        py = [[s[a:b], lab] for a, b, lab in model.entities(s)]
        if py != j:
            bad.append((s, py, j))
    record("G  entity spans", len(bad), len(sentences), bad[:1])

    bad = []
    for d, j in zip(docs, js["pipeline"]):
        p = lite.analyse(model, d["title"], d["text"])
        same = (p["summary"] == j["summary"] and p["entities"] == j["entities"]
                and p["sentiment"] == j["sentiment"]
                and p["sentiment_distribution"] == j["sentiment_distribution"]
                and abs(p["polarity"] - j["polarity"]) < TOL
                and close(p["topic_scores"], j["topic_scores"]))
        if not same:
            bad.append(d["title"])
    record("   whole pipeline", len(bad), len(docs), bad[:1])

    failed = False
    for name, (b, n) in checks.items():
        print(f"  {name:18s} {n - b:>6,} / {n:,} identical")
        failed |= b > 0
    if failed:
        sys.exit("PARITY FAILED: the page disagrees with the Python modules")
    print("PARITY OK -- the page computes exactly what the Python modules compute")


if __name__ == "__main__":
    main()
