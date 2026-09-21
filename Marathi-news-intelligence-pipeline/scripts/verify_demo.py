# -*- coding: utf-8 -*-
"""
Check that the demo page's JavaScript analyser agrees with the Python one.

    python scripts/verify_demo.py        (needs node on PATH)

The demo page re-implements src/morph.py in JavaScript so the analyser can run
in the browser. That port is only trustworthy if it produces identical roots,
so this script feeds the same words through both and reports any divergence.
Words are taken from the page's own embedded corpus.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config                 # noqa: E402
from src.morph import stem    # noqa: E402

PAGE = config.ROOT / "Marathi-NLP-Demo.html"

JS = r"""
import fs from "fs";
const [mp, wp, pp] = process.argv.slice(2);
const M = JSON.parse(fs.readFileSync(mp, "utf8"));
const words = JSON.parse(fs.readFileSync(wp, "utf8"));
const py = JSON.parse(fs.readFileSync(pp, "utf8"));
const MARKS = new Set(Array.from(M.marks)), FINAL = new Set(Array.from(M.finalMatras));
const byLen = t => t.slice().sort((a,b) => b[0].length - a[0].length);
const LAYERS = [byLen(M.clitics), byLen(M.postpositions), byLen(M.verb), byLen(M.case)];
const aksh = s => { let n=0; for (const c of s) if (!MARKS.has(c)) n++; return n; };
function restore(s){
  while (s.endsWith(M.virama)) s = s.slice(0,-1);
  if (s && FINAL.has(s[s.length-1]) && aksh(s) >= M.minStem) s = s.slice(0,-1);
  return s;
}
function strip1(w){
  for (const t of LAYERS) for (const [suf,feat,repl] of t){
    if (!w.endsWith(suf)) continue;
    const c = w.slice(0, w.length-suf.length) + repl;
    if (aksh(c) >= M.minStem) return [c,suf,feat];
  }
  return null;
}
function stem(word){
  let w = word.normalize("NFC");
  if (Object.prototype.hasOwnProperty.call(M.exceptions, word)) return M.exceptions[word];
  for (let i=0;i<M.maxStrips;i++){ const h = strip1(w); if (!h) break; w = h[0]; }
  return restore(w);
}
let bad = 0;
for (const w of words){
  if (stem(w) !== py[w]){
    if (bad < 10) console.log(`  MISMATCH ${w}: js=${stem(w)} py=${py[w]}`);
    bad++;
  }
}
console.log(`checked ${words.length} words, ${bad} mismatches`);
process.exit(bad ? 1 : 0);
"""


def main():
    if not PAGE.exists():
        sys.exit(f"{PAGE} not found -- run scripts/build_demo_page.py first")

    html = PAGE.read_text(encoding="utf-8")
    m = re.search(r'<script id="payload" type="application/json">(.*?)</script>',
                  html, re.S)
    if not m:
        sys.exit("no embedded payload found in the page")
    data = json.loads(m.group(1).replace("<" + chr(92) + "/", "</"))

    words = set()
    for rec in data["docs"]:
        words.update(re.findall(r"[\u0900-\u097F]+", rec["title"] + " " + rec["summary"]))
    words = sorted(words)
    print(f"{len(words)} distinct Devanagari types from the page's own corpus")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "morph.json").write_text(json.dumps(data["morph"], ensure_ascii=False),
                                       encoding="utf-8")
        (td / "words.json").write_text(json.dumps(words, ensure_ascii=False),
                                       encoding="utf-8")
        (td / "py.json").write_text(
            json.dumps({w: stem(w) for w in words}, ensure_ascii=False), encoding="utf-8")
        (td / "check.mjs").write_text(JS, encoding="utf-8")

        try:
            r = subprocess.run(
                ["node", str(td / "check.mjs"), str(td / "morph.json"),
                 str(td / "words.json"), str(td / "py.json")],
                capture_output=True, text=True, encoding="utf-8")
        except FileNotFoundError:
            sys.exit("node not found on PATH -- install Node.js to run this check")

    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr.strip())
        sys.exit("PARITY FAILED: the page's analyser disagrees with src/morph.py")
    print("PARITY OK -- the page analyses Marathi exactly as the Python module does")


if __name__ == "__main__":
    main()
