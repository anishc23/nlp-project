# -*- coding: utf-8 -*-
"""
Rule-based morphological analyser for Marathi (module I of the pipeline).

Marathi is agglutinative and heavily suffixing: a single nominal root takes a
number suffix, an oblique-stem change, and then one or more postpositions that
are written joined to the stem:

    घर  (house)
    घरात      = घर + आत          locative      "in the house"
    घरातून    = घर + आतून        ablative      "from inside the house"
    घरांमधून  = घर + ां + मधून   plural + abl. "from inside the houses"
    घराच्या   = घर + आ(obl) + च्या  genitive

A whitespace tokeniser therefore treats all four as unrelated types, which is
why plain keyword search performs poorly on Marathi. This module strips the
suffix chain, undoes the oblique stem change, and returns a normalised root
plus the morphological features it removed.

The analyser is deliberately rule-based rather than statistical: it needs no
training data, runs at roughly 100k tokens/sec, and its output is inspectable,
which is what makes the retrieval ablation in scripts/evaluate.py meaningful.

Design note -- why aksharas, not characters
-------------------------------------------
Devanagari vowel signs (matras) are separate Unicode code points, so len() is
not a measure of how much word is left. "घरा" is three code points but only
two aksharas (घ + र). Every length guard below therefore counts aksharas,
otherwise the minimum-stem check fires inconsistently and related forms such
as शाळा / शाळेत end up with different roots -- which would defeat the whole
purpose of the analyser.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------
# Character classes
# --------------------------------------------------------------------------
DEVANAGARI = r"ऀ-ॿ"
_TOKEN_RE = re.compile(rf"[{DEVANAGARI}]+|[A-Za-z]+|\d+")
_SENT_RE = re.compile(r"(?<=[।\.\!\?])\s+")

VIRAMA = "्"

# Combining marks: anusvara/visarga block, nukta, all vowel signs, virama,
# accents and vocalic marks. These attach to a base akshara.
_MARKS = frozenset(
    [chr(c) for c in range(0x0900, 0x0904)]        # candrabindu, anusvara, visarga
    + [chr(0x093A), chr(0x093B), chr(0x093C)]      # vowel signs oe/ooe, nukta
    + [chr(c) for c in range(0x093E, 0x0950)]      # matras + virama
    + [chr(c) for c in range(0x0951, 0x0958)]      # accents
    + [chr(0x0962), chr(0x0963)]                   # vocalic l matras
)

# Vowel signs that may be shed when normalising a stem's final syllable.
_FINAL_MATRAS = frozenset("ािीुूृेैोौ")


def akshara_len(s: str) -> int:
    """Number of base characters, ignoring combining vowel signs and virama."""
    return sum(1 for ch in s if ch not in _MARKS)


# --------------------------------------------------------------------------
# Suffix inventory.
#
# Entries are (surface_suffix, feature_tag) or (suffix, feature, replacement)
# where `replacement` is re-attached to the stem after stripping -- used where
# removing the suffix also removes part of the root, e.g. नद्या = नदी + PL.
#
# Layers are stripped from the outside in:
#     clitic -> postposition -> verb -> case/number
# Within a layer the table is applied longest-match-first.
# --------------------------------------------------------------------------
CLITICS = [
    ("सुद्धा", "EMPH"), ("देखील", "EMPH"),
    ("ही", "EMPH"), ("च", "EMPH"),
]

POSTPOSITIONS = [
    # ablative / source
    ("पासूनच", "ABL"), ("पासून", "ABL"), ("कडून", "ABL"), ("हून", "ABL"),
    ("ामधून", "ABL"), ("मधून", "ABL"), ("ातून", "ABL"), ("तून", "ABL"),
    ("वरून", "ABL"), ("खालून", "ABL"),
    # locative (incl. the adjectival -il / -la forms)
    ("ामध्ये", "LOC"), ("मध्ये", "LOC"), ("मधे", "LOC"), ("मधील", "LOC"),
    ("मधला", "LOC"), ("मधली", "LOC"), ("मधले", "LOC"),
    ("ातील", "LOC"), ("तील", "LOC"), ("ातला", "LOC"), ("ातली", "LOC"),
    ("ातले", "LOC"),
    ("वरील", "LOC"), ("वरचा", "LOC"), ("वरची", "LOC"), ("वरचे", "LOC"),
    ("वर", "LOC"), ("जवळ", "LOC"), ("खाली", "LOC"), ("समोर", "LOC"),
    ("मागे", "LOC"), ("पुढे", "LOC"), ("शेजारी", "LOC"),
    # goal / benefactive / comitative / other relational
    ("पर्यंत", "TERM"), ("साठी", "BEN"), ("करिता", "BEN"), ("मुळे", "CAUS"),
    ("कडील", "GEN"), ("कडे", "DAT"), ("विषयी", "ABOUT"), ("बद्दल", "ABOUT"),
    ("प्रमाणे", "SIM"), ("सारखा", "SIM"), ("सारखी", "SIM"), ("सारखे", "SIM"),
    ("नुसार", "SIM"), ("शिवाय", "PRIV"), ("विना", "PRIV"), ("सकट", "COM"),
]

# Verbal inflection (tense / aspect / agreement / non-finite).
VERB_SUFFIXES = [
    ("ण्यासाठी", "INF.BEN"), ("ण्यात", "INF.LOC"), ("ण्याचा", "INF.GEN"),
    ("ण्याची", "INF.GEN"), ("ण्याचे", "INF.GEN"),
    ("णारा", "AGT"), ("णारी", "AGT"), ("णारे", "AGT"), ("णार", "FUT"),
    ("णे", "INF"),
    ("ायला", "INF.DAT"), ("ायचा", "PROSP"), ("ायची", "PROSP"),
    ("ायचे", "PROSP"),
    ("तात", "PRS.3P"), ("तोय", "PRS.PROG"), ("ताय", "PRS.PROG"),
    ("तोस", "PRS.2S"), ("तो", "PRS.3SM"), ("ते", "PRS.3SN"), ("ती", "PRS.3SF"),
    ("लेला", "PTCP"), ("लेली", "PTCP"), ("लेले", "PTCP"),
    ("ल्या", "PST.PL"), ("ला", "PST.3SM"), ("ले", "PST.3PM"),
    ("ली", "PST.3SF"), ("लो", "PST.1S"),
    ("ईल", "FUT.3S"),
    ("ून", "CONJ"), ("ऊन", "CONJ"),
]

# Case + number endings written directly on the (oblique) stem.
CASE_NUMBER = [
    # plural oblique + case, longest first
    ("ांनी", "PL.INS"), ("ांना", "PL.DAT"), ("ांचा", "PL.GEN"),
    ("ांची", "PL.GEN"), ("ांचे", "PL.GEN"), ("ांच्या", "PL.GEN"),
    ("ांतून", "PL.ABL"), ("ांकडे", "PL.DAT"), ("ांत", "PL.LOC"),
    ("ां", "PL.OBL"),
    # singular genitive
    ("च्या", "GEN"), ("चा", "GEN"), ("ची", "GEN"), ("चे", "GEN"),
    # core cases
    ("ला", "DAT"), ("ना", "DAT"), ("ने", "INS"), ("नी", "INS"),
    ("शी", "COM"), ("त", "LOC"), ("स", "DAT"),
    # ya-plural of i-final nouns: नद्या -> नदी
    ("्या", "PL", "ी"),
]

# Derivational suffixes are NOT part of the default chain: stripping कार off
# सरकार ("government") yields सर, which is a different word. They are kept
# here for the morphology demo but only applied when explicitly requested.
DERIVATIONAL = [
    ("पणा", "NMLZ"), ("त्व", "NMLZ"), ("कार", "AGT"),
    ("वाला", "AGT"), ("गिरी", "NMLZ"), ("दार", "AGT"),
]


def _norm_table(table) -> List[Tuple[str, str, str]]:
    """Pad 2-tuples with an empty replacement and sort longest-match-first."""
    out = [(t[0], t[1], t[2] if len(t) > 2 else "") for t in table]
    return sorted(out, key=lambda kv: -len(kv[0]))


_LAYERS = [
    ("clitic", _norm_table(CLITICS)),
    ("postp", _norm_table(POSTPOSITIONS)),
    ("verb", _norm_table(VERB_SUFFIXES)),
    ("case", _norm_table(CASE_NUMBER)),
]
_DERIV_LAYER = ("deriv", _norm_table(DERIVATIONAL))

MIN_STEM = 2          # minimum aksharas that must survive a strip
MIN_DERIV_STEM = 3    # derivation needs a longer residue to be plausible
MAX_STRIPS = 4        # guard against runaway peeling

# --------------------------------------------------------------------------
# Marathi stopwords (function words that carry no topical signal)
# --------------------------------------------------------------------------
STOPWORDS = set("""
आणि किंवा पण तर म्हणून की हे हा ही ते तो ती या यांचा यांची यांचे त्या त्याचा
त्याची त्याचे आहे आहेत होते होता होती असे असा अशी नाही नसून एक दोन काही सर्व
खूप अधिक फक्त सुद्धा देखील तसेच मात्र परंतु कारण जर तरी आता मग इथे तिथे कुठे
कसे काय कोण कधी अशा त्यांनी त्यांना त्यांचा आपण आम्ही तुम्ही मी माझा तुमचा
वर खाली मध्ये पासून पर्यंत साठी बद्दल नंतर आधी दरम्यान विरुद्ध सोबत
करण्यात आला आले आली गेला गेले गेली दिला दिले दिली केली केले केला
यात यांच्या त्यांच्या अजून पुन्हा होणार असून येथे
""".split())

# Frequent forms whose rule-based analysis is wrong or misleading. Kept small
# and explicit -- this is the escape hatch a rule system needs, not a lexicon.
EXCEPTIONS = {
    "सरकार": "सरकार", "सरकारने": "सरकार", "सरकारच्या": "सरकार",
    "सरकारला": "सरकार", "सरकारी": "सरकार",
    "पुणे": "पुण", "पुण्यात": "पुण", "पुण्यातील": "पुण", "पुण्याच्या": "पुण",
    "मुंबई": "मुंबई", "मुंबईत": "मुंबई", "मुंबईतील": "मुंबई",
    "भारत": "भारत", "भारताने": "भारत", "भारतीय": "भारत", "भारतात": "भारत",
}


@dataclass
class Analysis:
    """Result of analysing a single surface token."""
    surface: str
    root: str
    features: List[str] = field(default_factory=list)
    stripped: List[str] = field(default_factory=list)

    @property
    def is_inflected(self) -> bool:
        return bool(self.stripped)

    def __str__(self) -> str:
        feats = "+".join(self.features) if self.features else "-"
        return f"{self.surface} -> {self.root} [{feats}]"


# --------------------------------------------------------------------------
# Stem restoration
# --------------------------------------------------------------------------
def _restore_stem(stem_form: str) -> str:
    """
    Normalise the stem left behind after suffix removal.

    The oblique stem of a consonant-final noun ends in -ा (घर -> घरा-), and
    plural/oblique forms surface with -े or -ी. Shedding one final vowel sign
    -- as long as two aksharas remain -- collapses घर / घरा / घरे, and
    शाळा / शाळे, onto a single root.
    """
    if not stem_form:
        return stem_form

    # A dangling virama is never a valid word end (पुण्य- from पुण्यात).
    while stem_form.endswith(VIRAMA):
        stem_form = stem_form[:-1]

    if stem_form and stem_form[-1] in _FINAL_MATRAS and akshara_len(stem_form) >= MIN_STEM:
        stem_form = stem_form[:-1]

    return stem_form


def _strip_once(word: str, layers) -> Optional[Tuple[str, str, str]]:
    """Remove at most one suffix. Returns (stem, suffix, feature) or None."""
    for layer_name, table in layers:
        floor = MIN_DERIV_STEM if layer_name == "deriv" else MIN_STEM
        for suf, feat, repl in table:
            if not word.endswith(suf):
                continue
            candidate = word[: -len(suf)] + repl
            if akshara_len(candidate) >= floor:
                return candidate, suf, feat
    return None


@lru_cache(maxsize=200_000)
def analyse(word: str, derivational: bool = False) -> Analysis:
    """Analyse one token into root + morphological features."""
    surface = word
    word = unicodedata.normalize("NFC", word)

    if surface in EXCEPTIONS:
        return Analysis(surface=surface, root=EXCEPTIONS[surface])

    layers = _LAYERS + [_DERIV_LAYER] if derivational else _LAYERS
    feats: List[str] = []
    strips: List[str] = []

    for _ in range(MAX_STRIPS):
        hit = _strip_once(word, layers)
        if hit is None:
            break
        new_stem, suf, feat = hit
        word = new_stem
        feats.append(feat)
        strips.append(suf)

    root = _restore_stem(word)
    return Analysis(surface=surface, root=root, features=feats, stripped=strips)


def stem(word: str) -> str:
    """Convenience wrapper: surface form -> normalised root."""
    return analyse(word).root


# --------------------------------------------------------------------------
# Tokenisation helpers used by the rest of the pipeline
# --------------------------------------------------------------------------
def tokenize(text: str) -> List[str]:
    """Script-aware word tokeniser (Devanagari, Latin, digits)."""
    return _TOKEN_RE.findall(text or "")


def sent_tokenize(text: str) -> List[str]:
    """Split on danda and western sentence punctuation."""
    parts = _SENT_RE.split((text or "").strip())
    return [p.strip() for p in parts if p and p.strip()]


def normalize_tokens(text: str, *, use_morph: bool = True,
                     drop_stopwords: bool = True) -> List[str]:
    """
    Full normalisation path used for indexing and similarity.

    `use_morph=False` reproduces the plain-tokenisation baseline, which is what
    the retrieval ablation compares against.
    """
    out = []
    for tok in tokenize(text):
        if drop_stopwords and tok in STOPWORDS:
            continue
        t = stem(tok) if use_morph else tok
        if drop_stopwords and t in STOPWORDS:
            continue
        if akshara_len(t) >= MIN_STEM or t.isascii():
            out.append(t)
    return out


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    print("-- analysis --")
    demo = [
        "घर", "घरात", "घरातून", "घरांमधून", "घराच्या", "घरासाठी",
        "मुलांना", "नद्या", "नदी", "शाळा", "शाळेत", "शाळेतून",
        "पुण्यातील", "सरकारने", "खेळणारा", "जिंकले", "बोलून", "मुंबईपासून",
    ]
    for w in demo:
        print("  ", analyse(w))

    print("\n-- conflation check (each group must share one root) --")
    groups = [
        ["घर", "घरात", "घरातून", "घरांमधून", "घराच्या", "घरासाठी", "घरे"],
        ["शाळा", "शाळेत", "शाळेतून", "शाळांना"],
        ["नदी", "नद्या", "नदीत"],
        ["सरकार", "सरकारने", "सरकारच्या", "सरकारला"],
        ["मुंबई", "मुंबईत", "मुंबईतील", "मुंबईपासून"],
    ]
    ok = True
    for g in groups:
        roots = {stem(w) for w in g}
        good = len(roots) == 1
        ok &= good
        print(f"  [{'OK ' if good else 'BAD'}] {g[0]:10s} -> {sorted(roots)}")
    print("\nall groups conflated:", ok)
