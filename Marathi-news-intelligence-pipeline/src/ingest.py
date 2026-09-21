# -*- coding: utf-8 -*-
"""
Live Marathi news ingestion over RSS.

Lets the pipeline run on today's news instead of only the static XL-Sum
corpus, which is what makes the dashboard demo feel real.

RSS items carry only a headline and a one-line description, so `fetch_all`
can optionally follow each link and pull the article body out of the page.
That part is best-effort by design: every outlet has a different template, and
a failed body fetch degrades to the RSS description rather than dropping the
article.

Only public RSS endpoints are read, one request at a time with a short delay,
and nothing is sent anywhere.
"""
from __future__ import annotations

import html
import re
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Verified reachable and returning Devanagari RSS.
FEEDS: Dict[str, str] = {
    "BBC Marathi": "https://feeds.bbci.co.uk/marathi/rss.xml",
    "ABP Majha": "https://marathi.abplive.com/home/feed",
    "Lokmat": "https://www.lokmat.com/rss/maharashtra.xml",
}

USER_AGENT = "Mozilla/5.0 (compatible; MarathiNewsPipeline/1.0; academic research)"
TIMEOUT = 20
POLITE_DELAY = 0.5

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass
class Article:
    id: str
    title: str
    text: str
    url: str
    source: str
    published: str = ""


def _clean(raw: str) -> str:
    """Strip tags and entities out of an RSS description or HTML fragment."""
    if not raw:
        return ""
    txt = _TAG_RE.sub(" ", raw)
    txt = html.unescape(txt)
    return _WS_RE.sub(" ", txt).strip()


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read()


def _text_of(item, *names) -> str:
    """First non-empty child among `names`, namespace-insensitive."""
    for name in names:
        for child in item:
            tag = child.tag.split("}")[-1].lower()
            if tag == name.lower() and (child.text or "").strip():
                return child.text.strip()
    return ""


def fetch_feed(name: str, url: str, limit: int = 25) -> List[Article]:
    """Parse one RSS feed into Articles (headline + description only)."""
    try:
        root = ET.fromstring(_get(url))
    except Exception as exc:                       # noqa: BLE001
        print(f"  ! {name}: {exc}")
        return []

    out: List[Article] = []
    # Handles both RSS <item> and Atom <entry>.
    items = root.iter()
    for node in items:
        tag = node.tag.split("}")[-1].lower()
        if tag not in ("item", "entry"):
            continue
        title = _clean(_text_of(node, "title"))
        desc = _clean(_text_of(node, "description", "summary", "content"))
        link = _text_of(node, "link", "guid")
        pub = _text_of(node, "pubDate", "published", "updated")
        if not title:
            continue
        out.append(Article(id=f"{name}:{len(out)}", title=title, text=desc,
                           url=link, source=name, published=pub))
        if len(out) >= limit:
            break
    return out


def fetch_body(url: str) -> Optional[str]:
    """Best-effort article body extraction. Returns None if nothing usable."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(_get(url), "html.parser")
    except Exception:                              # noqa: BLE001
        return None

    for bad in soup(["script", "style", "nav", "header", "footer", "aside"]):
        bad.decompose()

    # Longest run of <p> text on the page is almost always the article.
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paras = [p for p in paras if len(p) > 40]
    body = " ".join(paras)
    body = _WS_RE.sub(" ", body).strip()
    return body if len(body) > 200 else None


def fetch_all(limit: int = 60, with_body: bool = True,
              feeds: Dict[str, str] = None) -> List[Dict[str, str]]:
    """
    Pull headlines from every feed, optionally enriching with article bodies.

    Returns plain dicts shaped like the XL-Sum rows so build_corpus.py can
    treat both sources identically.
    """
    feeds = feeds or FEEDS
    per_feed = max(1, limit // max(len(feeds), 1))
    articles: List[Article] = []

    for name, url in feeds.items():
        got = fetch_feed(name, url, limit=per_feed)
        print(f"  {name:14s} {len(got):3d} items")
        articles.extend(got)
        time.sleep(POLITE_DELAY)

    if with_body:
        for a in articles:
            if not a.url:
                continue
            body = fetch_body(a.url)
            if body:
                a.text = body
            time.sleep(POLITE_DELAY)

    return [{"id": a.id, "title": a.title, "text": a.text or a.title,
             "url": a.url, "source": a.source, "published": a.published}
            for a in articles]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    rows = fetch_all(limit=9, with_body=True)
    print(f"\nfetched {len(rows)} articles")
    for r in rows[:5]:
        print(f"\n  [{r['source']}] {r['title'][:70]}")
        print(f"     {len(r['text'])} chars | {r['text'][:110]}")
