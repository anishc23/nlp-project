# -*- coding: utf-8 -*-
"""
Unit tests.

    python -m pytest tests/ -v
    python tests/test_pipeline.py        # runs without pytest too

The model-free components (morphology, BM25, TextRank, ROUGE) are tested
directly. Tests that need a transformer are marked `slow` and skipped unless
RUN_SLOW_TESTS=1, so the suite stays runnable on a machine with no GPU and no
model cache.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.morph import (Analysis, akshara_len, analyse, normalize_tokens,  # noqa: E402
                       sent_tokenize, stem, tokenize)
from src.retrieve import BM25Index, Hit, reciprocal_rank_fusion  # noqa: E402
from src.summarize import summarize  # noqa: E402

SLOW = pytest.mark.skipif(os.environ.get("RUN_SLOW_TESTS") != "1",
                          reason="set RUN_SLOW_TESTS=1 to run model-backed tests")


# ---------------------------------------------------------------- morphology
class TestMorphology:
    def test_akshara_length_ignores_matras(self):
        # घरा is 3 code points but 2 aksharas (घ + र with a vowel sign).
        assert len("घरा") == 3
        assert akshara_len("घरा") == 2
        assert akshara_len("घर") == 2

    @pytest.mark.parametrize("surface,expected", [
        ("घरात", "घर"), ("घरातून", "घर"), ("घरांमधून", "घर"),
        ("घराच्या", "घर"), ("घरासाठी", "घर"),
        ("मुंबईपासून", "मुंबई"), ("बोलून", "बोल"),
    ])
    def test_known_analyses(self, surface, expected):
        assert stem(surface) == expected

    @pytest.mark.parametrize("group", [
        ["घर", "घरात", "घरातून", "घरांमधून", "घराच्या", "घरासाठी", "घरे"],
        ["शाळा", "शाळेत", "शाळेतून", "शाळांना"],
        ["नदी", "नद्या", "नदीत"],
        ["सरकार", "सरकारने", "सरकारच्या", "सरकारला"],
        ["मुंबई", "मुंबईत", "मुंबईतील", "मुंबईपासून"],
    ])
    def test_inflectional_family_shares_one_root(self, group):
        """The whole point of the analyser: one lemma, one index term."""
        assert len({stem(w) for w in group}) == 1

    def test_no_overstripping_of_lexical_suffixes(self):
        # कार is a real derivational suffix, but सरकार is not सर + कार.
        assert stem("सरकारने") == "सरकार"

    def test_stem_never_empties_a_word(self):
        for w in ["ला", "ने", "त", "च", "आहे", "घर", "अ"]:
            assert stem(w), f"{w} stemmed to empty string"

    def test_features_are_reported(self):
        a = analyse("घरांमधून")
        assert isinstance(a, Analysis)
        assert a.is_inflected
        assert "ABL" in a.features

    def test_uninflected_word_is_unchanged(self):
        a = analyse("घर")
        assert a.root == "घर" and not a.is_inflected

    def test_tokenizer_handles_mixed_scripts(self):
        toks = tokenize("भारत India 2024 मध्ये")
        assert toks == ["भारत", "India", "2024", "मध्ये"]

    def test_sentence_split_on_danda(self):
        assert len(sent_tokenize("हे पहिले वाक्य आहे। हे दुसरे वाक्य आहे।")) == 2

    def test_stopwords_removed(self):
        assert "आणि" not in normalize_tokens("शाळा आणि महाविद्यालय")

    def test_morph_can_be_disabled(self):
        """The ablation switch must actually change the output."""
        assert normalize_tokens("घरात", use_morph=False) == ["घरात"]
        assert normalize_tokens("घरात", use_morph=True) == ["घर"]

    def test_empty_and_none_input(self):
        assert tokenize("") == [] and tokenize(None) == []
        assert normalize_tokens("") == []


# ----------------------------------------------------------------- retrieval
DOCS = [
    "पुण्यातील शाळांमध्ये नवीन शैक्षणिक धोरण लागू करण्यात आले आहे.",
    "भारतीय क्रिकेट संघाने कसोटी सामन्यात विजय मिळवला.",
    "मुंबईत मेट्रो प्रकल्पाचे काम वेगाने सुरू आहे.",
    "शाळेत विद्यार्थ्यांसाठी नवीन अभ्यासक्रम सुरू झाला.",
]


class TestRetrieval:
    def test_morphology_finds_what_plain_tokens_miss(self):
        """
        The corpus says शाळांमध्ये and शाळेत; the query says शाळा. A plain
        index cannot match either, a normalised one matches both.
        """
        plain = BM25Index(use_morph=False).build(DOCS)
        morph = BM25Index(use_morph=True).build(DOCS)
        assert len(plain.search("शाळा", top_k=5)) == 0
        assert len(morph.search("शाळा", top_k=5)) == 2

    def test_morphology_shrinks_the_vocabulary(self):
        plain = BM25Index(use_morph=False).build(DOCS)
        morph = BM25Index(use_morph=True).build(DOCS)
        assert len(morph.postings) < len(plain.postings)

    def test_hits_are_rank_ordered(self):
        idx = BM25Index().build(DOCS)
        hits = idx.search("नवीन", top_k=5)
        assert [h.rank for h in hits] == list(range(len(hits)))
        assert all(a.score >= b.score for a, b in zip(hits, hits[1:]))

    def test_empty_query_and_empty_index(self):
        assert BM25Index().build(DOCS).search("", top_k=5) == []
        assert BM25Index().build([]).search("शाळा", top_k=5) == []

    def test_unknown_term_returns_nothing(self):
        assert BM25Index().build(DOCS).search("झेब्रा", top_k=5) == []

    def test_rrf_prefers_documents_both_runs_agree_on(self):
        run_a = [Hit(doc_id=1, score=9.0, rank=0), Hit(doc_id=2, score=8.0, rank=1)]
        run_b = [Hit(doc_id=2, score=0.9, rank=0), Hit(doc_id=3, score=0.8, rank=1)]
        fused = reciprocal_rank_fusion([run_a, run_b], top_k=3)
        assert fused[0].doc_id == 2          # only doc ranked highly by both

    def test_bm25_index_round_trips(self, tmp_path):
        idx = BM25Index().build(DOCS)
        p = tmp_path / "idx.pkl"
        idx.save(p)
        assert BM25Index.load(p).search("शाळा", 5) == idx.search("शाळा", 5)


# -------------------------------------------------------------- summarization
class TestSummarization:
    TEXT = ("पुण्यात आज शिक्षण परिषद पार पडली। "
            "परिषदेत नवीन शैक्षणिक धोरणावर चर्चा झाली। "
            "शिक्षणमंत्र्यांनी विद्यार्थ्यांसाठी नवीन योजना जाहीर केली। "
            "अनेक शाळांचे मुख्याध्यापक उपस्थित होते। "
            "पुढील वर्षापासून धोरण लागू होणार आहे।")

    def test_returns_requested_number_of_sentences(self):
        assert len(summarize(self.TEXT, k=2).sentences) == 2

    def test_summary_sentences_come_from_the_source(self):
        s = summarize(self.TEXT, k=2)
        for sent in s.sentences:
            assert sent in self.TEXT

    def test_original_order_is_preserved(self):
        assert summarize(self.TEXT, k=3).indices == sorted(summarize(self.TEXT, k=3).indices)

    def test_lead_baseline_takes_the_opening(self):
        assert summarize(self.TEXT, k=1, method="lead").indices == [0]

    def test_short_input_is_returned_whole(self):
        one = "फक्त एकच वाक्य आहे।"
        assert summarize(one, k=3).sentences == [one]

    def test_empty_input(self):
        assert summarize("", k=3).sentences == []

    @SLOW
    def test_embedding_method_runs(self):
        assert len(summarize(self.TEXT, k=2, method="embedding").sentences) == 2


# ---------------------------------------------------------------------- rouge
class TestRouge:
    def setup_method(self):
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
        import evaluate
        self.ev = evaluate

    def test_identical_text_scores_one(self):
        r = self.ev.rouge_all("शाळेत नवीन अभ्यासक्रम", "शाळेत नवीन अभ्यासक्रम", False)
        assert r["rouge1"] == pytest.approx(1.0)
        assert r["rougeL"] == pytest.approx(1.0)

    def test_disjoint_text_scores_zero(self):
        assert self.ev.rouge_all("क्रिकेट सामना", "शैक्षणिक धोरण", False)["rouge1"] == 0.0

    def test_morphology_credits_inflectional_variants(self):
        """
        Same content, different case endings: raw ROUGE-1 misses it, the
        normalised variant catches it. This is why the evaluation reports both.
        """
        pred, gold = "शाळेत विद्यार्थी", "शाळांमध्ये विद्यार्थ्यांना"
        assert self.ev.rouge_all(pred, gold, False)["rouge1"] == 0.0
        assert self.ev.rouge_all(pred, gold, True)["rouge1"] > 0.0

    def test_empty_inputs_do_not_crash(self):
        assert self.ev.rouge_all("", "काहीतरी", False)["rouge1"] == 0.0


# --------------------------------------------------------------- aggregation
class TestAliasMerging:
    def setup_method(self):
        from src.ner import merge_aliases
        self.merge = merge_aliases

    def test_short_name_folds_into_full_name(self):
        from collections import Counter
        out = self.merge(Counter({"नरेंद्र मोदी": 70, "मोदी": 55}))
        assert out["नरेंद्र मोदी"] == 125
        assert "मोदी" not in out

    def test_merges_when_forms_are_equally_common(self):
        from collections import Counter
        out = self.merge(Counter({"डोनाल्ड ट्रंप": 39, "ट्रंप": 40}))
        assert out["डोनाल्ड ट्रंप"] == 79

    def test_common_surname_not_absorbed_by_rare_full_name(self):
        from collections import Counter
        out = self.merge(Counter({"पवार": 100, "अजित पवार": 2}))
        assert out["पवार"] == 100 and out["अजित पवार"] == 2

    def test_two_full_names_are_never_merged(self):
        from collections import Counter
        counts = Counter({"अजित पवार": 30, "शरद पवार": 62})
        assert self.merge(counts) == counts

    def test_unrelated_names_untouched(self):
        from collections import Counter
        counts = Counter({"मुंबई": 10, "पुणे": 8})
        assert self.merge(counts) == counts


# ------------------------------------------------------ in-browser students
LITE = Path(__file__).resolve().parent.parent / "app" / "lite_models.json"
NEEDS_LITE = pytest.mark.skipif(not LITE.exists(),
                                reason="run scripts/train_lite.py to build app/lite_models.json")


class TestLite:
    def test_words_split_on_boundaries_and_keep_offsets(self):
        from src.lite import words
        text = "इंडो-तिबेटन पोलीस, मुंबईत."
        toks = [text[s:e] for s, e in words(text)]
        assert toks == ["इंडो", "तिबेटन", "पोलीस", "मुंबईत"]

    def test_digit_shape_is_explicit(self):
        from src.lite import _shape
        assert _shape("2024") == "digit"
        assert _shape("२०२४") == "digit"
        assert _shape("BBC") == "latin"
        assert _shape("मुंबई") == "deva"

    @NEEDS_LITE
    def test_topic_student_recognises_sport(self):
        from src.lite import Lite
        m = Lite.load()
        p = m.topic("भारताचा विजय", "भारतीय क्रिकेट संघाने कसोटी सामन्यात इंग्लंडचा "
                    "नऊ गडी राखून पराभव केला. कर्णधाराने संघाचे कौतुक केले.")
        assert max(p, key=p.get) == "Sports"
        assert abs(sum(p.values()) - 1.0) < 1e-9

    @NEEDS_LITE
    def test_entity_spans_are_whole_words(self):
        from src.lite import Lite, words
        m = Lite.load()
        text = "मुख्यमंत्री एकनाथ शिंदे यांनी सोमवारी मुंबईत घोषणा केली."
        starts = {s for s, _ in words(text)}
        ends = {e for _, e in words(text)}
        for s, e, _ in m.entities(text):
            assert s in starts and e in ends

    @NEEDS_LITE
    def test_pipeline_output_shape(self):
        from src.lite import Lite, analyse
        out = analyse(Lite.load(), "शीर्षक", "पहिले वाक्य इथे आहे. दुसरे वाक्य इथे आहे. "
                      "तिसरे वाक्य इथे आहे. चौथे वाक्य इथे आहे.")
        assert set(out) >= {"topic_scores", "summary", "entities", "sentiment",
                            "polarity", "sentiment_distribution"}
        assert out["sentiment"] in ("Positive", "Neutral", "Negative")


class TestSummaryTies:
    def test_exact_ties_keep_document_order(self):
        """Repeated sentences tie exactly; the earlier one must win, so the
        page's JavaScript port (different float rounding) agrees."""
        from src.summarize import summarize
        s = "हा पहिला वेगळा मुद्दा आहे."
        dup = "सरकारने नवीन योजना जाहीर केली आहे."
        text = " ".join([dup, s, "आणखी एक वेगळे वाक्य आहे.", dup, "शेवटचे वेगळे वाक्य आहे."])
        # The two copies (indices 0 and 3) have identical PageRank scores.
        for k in (1, 2, 3):
            idx = summarize(text, k=k).indices
            assert not (3 in idx and 0 not in idx)


# ------------------------------------------------------------- model server
class TestServer:
    def test_health_and_cors_preflight(self):
        import json as _json
        import threading
        import urllib.request
        from http.server import ThreadingHTTPServer
        from app.server import Handler

        srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/api/health") as r:
                assert _json.loads(r.read())["ok"] is True
                assert r.headers["Access-Control-Allow-Origin"] == "*"
            req = urllib.request.Request(base + "/api/analyse", method="OPTIONS")
            with urllib.request.urlopen(req) as r:
                assert r.status == 204
                assert r.headers["Access-Control-Allow-Private-Network"] == "true"
            req = urllib.request.Request(base + "/api/analyse", data=b'{"text": "  "}',
                                         method="POST",
                                         headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req)
                raise AssertionError("empty text should be rejected")
            except urllib.error.HTTPError as err:
                assert err.code == 400
        finally:
            srv.shutdown()


# ------------------------------------------------------------- model-backed
class TestModels:
    @SLOW
    def test_topic_classifier(self):
        from src import categorize
        p = categorize.classify(["भारतीय क्रिकेट संघाने सामना जिंकला."])[0]
        assert p.label == "Sports" and p.score > 0.5

    @SLOW
    def test_ner_finds_person_and_location(self):
        from src import ner
        ents = ner.extract("मुख्यमंत्री एकनाथ शिंदे यांनी मुंबईत घोषणा केली.")
        labels = {e.label for e in ents}
        assert "Person" in labels and "Location" in labels

    @SLOW
    def test_ner_spans_are_whole_words(self):
        """Regression: wordpiece splits used to yield ंतप्रधान for पंतप्रधान."""
        from src import ner
        text = "पंतप्रधान नरेंद्र मोदी यांनी भाषण केले."
        for e in ner.extract(text):
            assert not e.text[0] in "ािीुूेैोौं्", f"span starts mid-akshara: {e.text}"

    @SLOW
    def test_entity_canonical_form_deinflects(self):
        from src.ner import Entity
        e = Entity(text="मुंबईत", label="Location", start=0, end=6, score=1.0)
        assert e.canonical == "मुंबई"

    @SLOW
    def test_sentiment_polarity_direction(self):
        from src import sentiment
        pos = sentiment.analyse_document("भारताने शानदार विजय मिळवला.")
        neg = sentiment.analyse_document("भीषण अपघातात अनेक लोक जखमी झाले.")
        assert pos.polarity > neg.polarity

    @SLOW
    def test_embeddings_are_normalised(self):
        import numpy as np
        from src import models
        v = models.embed(["शाळेत नवीन अभ्यासक्रम", "क्रिकेट सामना"])
        assert v.shape[0] == 2
        assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3)

    @SLOW
    def test_translation_produces_english(self):
        from src import translate
        out = translate.translate("भारतीय क्रिकेट संघाने सामना जिंकला.")
        assert out and out.isascii()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
