# -*- coding: utf-8 -*-
"""
Assemble the standalone demo page, Marathi-NLP-Demo.html.

    python scripts/export_demo.py        # data + model weights -> outputs/demo_data.json
    python scripts/build_demo_page.py    # -> Marathi-NLP-Demo.html
    python scripts/verify_demo.py        # JavaScript vs Python parity

The page is a single self-contained HTML file: the JSON payload goes into a
<script type="application/json"> tag rather than being fetched, and the
engine (app/demo_engine.js) is inlined, so the page works offline and can be
emailed or opened from a pendrive.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config  # noqa: E402

TEMPLATE = config.ROOT / "app" / "demo_template.html"
ENGINE = config.ROOT / "app" / "demo_engine.js"
DATA = config.OUTPUTS / "demo_data.json"
OUT = config.ROOT / "Marathi-NLP-Demo.html"


def main():
    if not DATA.exists():
        sys.exit("run scripts/export_demo.py first")
    payload = json.loads(DATA.read_text(encoding="utf-8"))

    # A literal </script> inside the JSON would close the tag early;
    # escaping the slash keeps the JSON valid but hides it from the
    # HTML parser.
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) \
        .replace("</", "<" + chr(92) + "/")
    engine = ENGINE.read_text(encoding="utf-8")
    if "</script" in engine:
        sys.exit("demo_engine.js must not contain '</script'")

    html = TEMPLATE.read_text(encoding="utf-8")
    for marker in ("__DATA__", "/*__ENGINE__*/"):
        if html.count(marker) != 1:
            sys.exit(f"template must contain {marker} exactly once")
    html = html.replace("__DATA__", blob).replace("/*__ENGINE__*/", engine)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
