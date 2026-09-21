# -*- coding: utf-8 -*-
"""
Inline the exported demo data into the standalone demo page.

    python scripts/export_demo.py
    python scripts/build_demo_page.py

The page is a single self-contained HTML file: the JSON payload goes into a
<script type="application/json"> tag rather than being fetched, so the page
works offline and can be published as one artifact with no supporting files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import config  # noqa: E402

TEMPLATE = config.ROOT / "app" / "demo_template.html"
DATA = config.OUTPUTS / "demo_data.json"
OUT = config.OUTPUTS / "demo.html"


def main():
    if not DATA.exists():
        sys.exit("run scripts/export_demo.py first")
    payload = json.loads(DATA.read_text(encoding="utf-8"))

    # A literal </script> inside the JSON would close the tag early;
    # escaping the slash keeps the JSON valid but hides it from the
    # HTML parser.
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) \
        .replace("</", "<" + chr(92) + "/")

    html = TEMPLATE.read_text(encoding="utf-8").replace("__DATA__", blob)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
