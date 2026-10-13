"""FinBERT of PLAN_V2 section 8.7, its scorer finbert_v1 and scripts/export_finbert.py, offline.

A fake session and a tiny word level tokenizer built in the test check windowing, weighting by token count, label
order, strength and every hash and manifest mismatch, which raise FinbertError and stop the news process. The scorer
runs on a temporary news.db with a fake model and a fake clock. The export script is checked without running the
export, so nothing is downloaded and torch is never imported. Two needs_model tests run on the exported model and are
skipped while it is absent, one checking bitwise identical scores on two runs."""

import ast
import asyncio
import importlib.util
import json
import logging
import sqlite3
import struct
import sys
import tomllib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from btc_survivor.clock import MS_PER_HOUR, MS_PER_SECOND, LiveClock, iso, parse_iso
from btc_survivor.v2.config import load_news_config
from btc_survivor.v2.news import collector, db, finbert, finbert_scorer, health
from btc_survivor.v2.news.finbert import LABELS, Finbert, FinbertError, Probs
from btc_survivor.v2.news.finbert_scorer import SCORER, FinbertScorer
from btc_survivor.v2.news.prefilter import Prefilter
from btc_survivor.v2.news.schedule import Schedule
from helpers import ROOT

CONFIG = load_news_config(ROOT / "configs" / "v2" / "news.toml")
NOW = parse_iso("2026-10-14T15:00:00.000Z")


def _load_script():
    spec = importlib.util.spec_from_file_location("export_finbert", ROOT / "scripts" / "export_finbert.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


export_finbert = _load_script()

# A tiny word level tokenizer in the shape the runtime reads, with the words the fake session counts.
WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", ".", "good", "bad", "flat", "news", "today"]
CLS_ID, SEP_ID, DOT, GOOD, BAD, FLAT = 2, 3, 4, 5, 6, 7
LABEL_WORD = {"positive": GOOD, "negative": BAD, "neutral": FLAT}


def tiny_tokenizer(truncate: int | None = None) -> str:
    tok = Tokenizer(models.WordLevel({word: i for i, word in enumerate(WORDS)}, unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    if truncate:
        tok.enable_truncation(truncate)
    return tok.to_str()


class FakeSession:
    """Logits of 2 per occurrence of each label's word, in the label order given, recording every feed."""

    def __init__(self, labels=LABELS, inputs=finbert.INPUTS):
        self.labels = tuple(labels)
        self.inputs = tuple(inputs)
        self.feeds = []

    def get_inputs(self):
        return [SimpleNamespace(name=name) for name in self.inputs]

    def run(self, output_names, feed):
        assert output_names is None
        assert set(feed) == set(self.inputs)
        for array in feed.values():
            assert array.dtype == np.int64 and array.shape[0] == 1 and array.shape == feed["input_ids"].shape
        self.feeds.append({name: array.copy() for name, array in feed.items()})
        ids = feed["input_ids"]
        return [np.array([[2.0 * int(np.sum(ids == LABEL_WORD[label])) for label in self.labels]],
                         dtype=np.float32)]


def make(window_tokens=4, max_windows=3, labels=LABELS, session=None, truncate=None):
    session = session or FakeSession(labels)
    return Finbert(session, Tokenizer.from_str(tiny_tokenizer(truncate)), labels, window_tokens, max_windows), session


def ids_of(feed):
    return feed["input_ids"][0].tolist()


# Windowing


def test_windows_cut_at_the_size_keep_at_most_the_limit_and_never_return_none():
    assert finbert.windows(list(range(10)), 4, 16) == [(0, 1, 2, 3), (4, 5, 6, 7), (8, 9)]
    assert finbert.windows(list(range(10)), 4, 2) == [(0, 1, 2, 3), (4, 5, 6, 7)]
    assert finbert.windows(list(range(4)), 4, 16) == [(0, 1, 2, 3)]
    assert finbert.windows([], 4, 16) == [()]
    size, limit = CONFIG.finbert.window_tokens, CONFIG.finbert.max_windows
    assert (size, limit) == (510, 16)
    assert [len(w) for w in finbert.windows(range(510), size, limit)] == [510]
    assert [len(w) for w in finbert.windows(range(511), size, limit)] == [510, 1]
    long = finbert.windows(range(size * limit + 700), size, limit)
    assert len(long) == 16 and all(len(w) == 510 for w in long) and long[-1][-1] == size * limit - 1


def test_the_headline_and_text_are_joined_by_a_period():
    assert finbert.join_text("Good news", "Bad today") == "Good news. Bad today"
    assert finbert.join_text(" Good news ", None) == "Good news"
    assert finbert.join_text("Good news", "  ") == "Good news"
    assert finbert.join_text("", "") == ""


def test_each_window_is_scored_alone_between_cls_and_sep_without_padding():
    model, session = make(window_tokens=4)
    probs = model.score("good good good good", "bad bad flat")   # 8 tokens with the joining period
    assert [ids_of(feed) for feed in session.feeds] == [
        [CLS_ID, GOOD, GOOD, GOOD, GOOD, SEP_ID], [CLS_ID, DOT, BAD, BAD, FLAT, SEP_ID]]
    for feed in session.feeds:
        assert feed["attention_mask"].tolist() == [[1] * 6]
        assert feed["token_type_ids"].tolist() == [[0] * 6]
    assert (probs.windows, probs.tokens) == (2, 8)


def test_windows_past_the_limit_are_dropped():
    model, session = make(window_tokens=4, max_windows=1)
    probs = model.score("good good good good", "bad bad bad bad bad")
    alone, _ = make(window_tokens=4, max_windows=1)
    assert len(session.feeds) == 1 and (probs.windows, probs.tokens) == (1, 4)
    assert probs == alone.score("good good good good")


def test_a_truncation_saved_in_the_tokenizer_never_cuts_a_text():
    model, session = make(window_tokens=4, truncate=3)
    assert model.score("good good good good bad bad").tokens == 6
    assert [len(ids_of(feed)) for feed in session.feeds] == [6, 4]


def test_an_empty_text_scores_one_window_of_special_tokens():
    model, session = make()
    probs = model.score("", None)
    assert [ids_of(feed) for feed in session.feeds] == [[CLS_ID, SEP_ID]]
    assert (probs.windows, probs.tokens) == (1, 0)
    assert probs.pos == probs.neg == probs.neu == pytest.approx(1 / 3)


def test_the_model_must_take_input_ids_and_only_the_three_bert_inputs():
    tokenizer = Tokenizer.from_str(tiny_tokenizer())
    Finbert(FakeSession(inputs=("input_ids", "attention_mask")), tokenizer, LABELS, 4, 3)
    for inputs in (("attention_mask", "token_type_ids"), (*finbert.INPUTS, "position_ids")):
        with pytest.raises(FinbertError, match="inputs"):
            Finbert(FakeSession(inputs=inputs), tokenizer, LABELS, 4, 3)
    model, _ = make(session=FakeSession(inputs=("input_ids", "attention_mask")))
    assert model.score("good").pos > model.score("good").neg


def test_a_tokenizer_without_cls_and_sep_is_refused():
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "good": 1}, unk_token="[UNK]"))
    with pytest.raises(FinbertError, match="CLS"):
        Finbert(FakeSession(), tok, LABELS, 4, 3)


# Weighting


def test_probabilities_are_averaged_with_each_window_weighted_by_its_token_count():
    model, _ = make(window_tokens=4)
    probs = model.score("good good good good bad")   # windows of 4 tokens and of 1 token
    first, second = finbert.softmax(np.array([8.0, 0.0, 0.0])), finbert.softmax(np.array([0.0, 2.0, 0.0]))
    expected = (4 * first + 1 * second) / 5
    np.testing.assert_allclose([probs.pos, probs.neg, probs.neu], expected, rtol=1e-12, atol=0)
    unweighted = (first + second) / 2
    assert abs(probs.pos - unweighted[0]) > 0.1
    assert probs.pos + probs.neg + probs.neu == pytest.approx(1.0, abs=1e-12)


def test_softmax_and_the_weighted_mean_by_hand():
    np.testing.assert_allclose(finbert.softmax(np.array([0.0, np.log(3.0), 0.0], dtype=np.float32)),
                               [0.2, 0.6, 0.2], rtol=1e-6)
    assert finbert.softmax(np.array([1000.0, 0.0, 0.0])).tolist() == [1.0, 0.0, 0.0]
    rows = np.array([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1]])
    np.testing.assert_allclose(finbert.weighted_mean(rows, [3, 1]), [0.4, 0.3875, 0.2125], rtol=1e-12)
    np.testing.assert_allclose(finbert.weighted_mean(rows[:1], [0]), rows[0], rtol=0)


# Label order


def test_probabilities_follow_the_label_order_of_the_model_outputs():
    order = ("neutral", "positive", "negative")
    model, _ = make(labels=order)
    probs = model.score("good good bad")
    expected = finbert.softmax(np.array([4.0, 2.0, 0.0]))     # positive, negative, neutral
    np.testing.assert_allclose([probs.pos, probs.neg, probs.neu], expected, rtol=1e-12)
    assert probs.pos > probs.neg > probs.neu
    default, _ = make()
    assert default.score("good good bad") == pytest.approx(probs)


def test_a_wrong_label_order_would_swap_the_probabilities():
    session = FakeSession(("negative", "positive", "neutral"))
    model, _ = make(labels=LABELS, session=session)
    probs = model.score("good good good")
    assert probs.neg > probs.pos     # what a manifest that misstates the order would do, so load reads it


# Strength


def test_strength_is_p_pos_minus_p_neg():
    assert Probs(0.7, 0.2, 0.1, 1, 3).strength == 0.7 - 0.2
    assert Probs(0.1, 0.6, 0.3, 1, 3).strength == 0.1 - 0.6
    model, _ = make()
    probs = model.score("bad bad good")
    assert probs.strength == probs.pos - probs.neg < 0


# Loading, pins and hash mismatches on a fake session


MODEL_BYTES = b"a fake onnx model, never run by onnxruntime"


def write_export(folder, model=MODEL_BYTES, labels=LABELS, positions=512):
    """The three files in the shape scripts/export_finbert.py writes them, its own manifest builder included."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = {role: folder / name for role, name in (("model", "model.onnx"), ("tokenizer", "tokenizer.json"),
                                                    ("manifest", "manifest.json"))}
    paths["model"].write_bytes(model)
    paths["tokenizer"].write_text(tiny_tokenizer(), encoding="utf-8")
    manifest = export_finbert.build_manifest(paths["model"], paths["tokenizer"], labels, positions, {}, {}, [], {})
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    return paths, manifest


def pinned_config(paths, manifest, **changes):
    """The committed news config with [finbert] pointed at the files and pinned from the export script's lines."""
    pins = tomllib.loads("\n".join(export_finbert.pins_text(manifest)))
    settings = replace(CONFIG.finbert, model_file=str(paths["model"]), tokenizer_file=str(paths["tokenizer"]),
                       manifest_file=str(paths["manifest"]), **{**pins, **changes})
    return replace(CONFIG, finbert=settings)


class Factory:
    """A session factory that records what it was given and returns a fake session."""

    def __init__(self, labels=LABELS):
        self.calls = []
        self.labels = labels

    def __call__(self, model, threads):
        self.calls.append((model, threads))
        return FakeSession(self.labels)


def test_a_pinned_export_loads_from_the_bytes_it_checked(tmp_path):
    paths, manifest = write_export(tmp_path)
    factory = Factory()
    model = Finbert.load(pinned_config(paths, manifest), session_factory=factory)
    assert factory.calls == [(MODEL_BYTES, 1)]
    assert (model.window_tokens, model.max_windows) == (510, 16)
    assert model.score("good news today").pos > model.score("good news today").neu


def test_the_label_order_comes_from_the_manifest(tmp_path):
    order = ("neutral", "negative", "positive")
    paths, manifest = write_export(tmp_path, labels=order)
    model = Finbert.load(pinned_config(paths, manifest), session_factory=Factory(order))
    probs = model.score("good good bad")
    assert probs.pos > probs.neg > probs.neu
    assert manifest["labels"] == list(order)


@pytest.mark.parametrize("role", ["model", "tokenizer"])
def test_a_file_changed_after_pinning_is_a_hash_mismatch(tmp_path, role):
    paths, manifest = write_export(tmp_path)
    config = pinned_config(paths, manifest)
    paths[role].write_bytes(paths[role].read_bytes() + b" ")
    factory = Factory()
    with pytest.raises(FinbertError, match=f"the {role} file .* has sha256 .* but \\[finbert\\] pins") as caught:
        Finbert.load(config, session_factory=factory)
    assert "the news process stops" in str(caught.value) and "the manifest says" in str(caught.value)
    assert factory.calls == []


@pytest.mark.parametrize("role", ["model", "tokenizer"])
def test_a_config_pin_that_differs_from_the_file_is_a_hash_mismatch(tmp_path, role):
    paths, manifest = write_export(tmp_path)
    config = pinned_config(paths, manifest, **{f"{role}_sha256": "0" * 64})
    with pytest.raises(FinbertError, match=f"the {role} file .* pins {'0' * 64}"):
        Finbert.load(config, session_factory=Factory())


def test_a_manifest_hash_that_differs_from_the_file_is_a_mismatch(tmp_path):
    paths, manifest = write_export(tmp_path)
    config = pinned_config(paths, manifest)
    manifest["model"]["sha256"] = "f" * 64
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FinbertError, match="the manifest says f{64}"):
        finbert.verify(config)


def test_unpinned_hashes_and_revision_are_refused(tmp_path):
    paths, manifest = write_export(tmp_path)
    config = pinned_config(paths, manifest, model_sha256="", tokenizer_sha256="", source_revision="")
    with pytest.raises(FinbertError) as caught:
        finbert.verify(config)
    for key in ("model_sha256", "tokenizer_sha256", "source_revision"):
        assert f"[finbert] {key} is not pinned" in str(caught.value)
    assert not CONFIG.finbert.model_sha256 or CONFIG.finbert.source_revision == export_finbert.REVISION


def test_a_manifest_of_another_revision_is_refused(tmp_path):
    paths, manifest = write_export(tmp_path)
    config = pinned_config(paths, manifest, source_revision="0123456789abcdef0123456789abcdef01234567")
    with pytest.raises(FinbertError, match="manifest revision .* is not the pinned source_revision"):
        finbert.verify(config)


@pytest.mark.parametrize("labels", [["positive", "negative"], ["positive", "negative", "bullish"],
                                    ["positive", "positive", "neutral"], "positive"])
def test_a_manifest_label_order_that_is_not_the_three_labels_is_refused(tmp_path, labels):
    paths, manifest = write_export(tmp_path)
    config = pinned_config(paths, manifest)
    manifest["labels"] = labels
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FinbertError, match="manifest labels"):
        finbert.verify(config)


def test_windows_that_do_not_fit_the_position_limit_are_refused(tmp_path):
    paths, manifest = write_export(tmp_path, positions=511)
    with pytest.raises(FinbertError, match="window_tokens 510 plus 2 special tokens"):
        finbert.verify(pinned_config(paths, manifest))
    assert finbert.verify(pinned_config(*write_export(tmp_path / "ok", positions=512))).max_positions == 512


@pytest.mark.parametrize("role", ["model", "tokenizer", "manifest"])
def test_a_missing_file_names_the_export_script(tmp_path, role):
    paths, manifest = write_export(tmp_path)
    config = pinned_config(paths, manifest)
    paths[role].unlink()
    with pytest.raises(FinbertError, match="scripts/export_finbert.py"):
        Finbert.load(config, session_factory=Factory())


def test_a_manifest_that_is_not_a_json_object_is_refused(tmp_path):
    paths, manifest = write_export(tmp_path)
    config = pinned_config(paths, manifest)
    for text in ("not json", "[1, 2]"):
        paths["manifest"].write_text(text, encoding="utf-8")
        with pytest.raises(FinbertError, match="manifest"):
            finbert.verify(config)


def test_a_tokenizer_file_that_cannot_be_read_is_refused(tmp_path):
    paths, manifest = write_export(tmp_path)
    paths["tokenizer"].write_text("{}", encoding="utf-8")
    _, manifest = write_export(tmp_path / "other")
    manifest["tokenizer"] = export_finbert.file_entry(paths["tokenizer"])
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FinbertError, match="tokenizer cannot be read"):
        Finbert.load(pinned_config(paths, manifest), session_factory=Factory())


def test_finbert_error_is_a_value_error_so_the_news_runner_exits_with_code_2():
    assert issubclass(FinbertError, ValueError)


def test_the_session_runs_on_the_cpu_with_one_thread_and_sequential_execution(monkeypatch):
    options = finbert.session_options(CONFIG.finbert.threads)
    assert (options.intra_op_num_threads, options.inter_op_num_threads) == (1, 1)
    assert options.execution_mode == finbert.ort.ExecutionMode.ORT_SEQUENTIAL
    seen = {}

    def fake_session(model, sess_options, providers):
        seen.update(model=model, options=sess_options, providers=providers)
        return FakeSession()

    monkeypatch.setattr(finbert.ort, "InferenceSession", fake_session)
    finbert.open_session(MODEL_BYTES, 1)
    assert seen["model"] == MODEL_BYTES and seen["providers"] == ["CPUExecutionProvider"]
    assert seen["options"].intra_op_num_threads == 1 and seen["options"].inter_op_num_threads == 1


# The real onnxruntime on a tiny model encoded by hand, since the onnx package stays out of the project environment.
# It takes the three BERT inputs and returns the mean over the window of a fixed row per token, 2 for the label word.


def _varint(n):
    out = bytearray()
    while True:
        byte, n = n & 0x7F, n >> 7
        out.append(byte | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _field(number, value):
    """One protobuf field, an integer as a varint and text or bytes length delimited."""
    if isinstance(value, int):
        return _varint(number << 3) + _varint(value)
    value = value.encode() if isinstance(value, str) else value
    return _varint(number << 3 | 2) + _varint(len(value)) + value


def _value_info(name, elem_type, dims):
    shape = b"".join(_field(1, _field(1, d) if isinstance(d, int) else _field(2, d)) for d in dims)
    return _field(1, name) + _field(2, _field(1, _field(1, elem_type) + _field(2, shape)))


def tiny_onnx_model():
    """ModelProto bytes at IR 8 and opset 17, Gather of input_ids from a table then ReduceMean over the window."""
    table = np.zeros((len(WORDS), 3), dtype=np.float32)
    for column, label in enumerate(LABELS):
        table[LABEL_WORD[label], column] = 2.0
    tensor = b"".join(_field(1, d) for d in table.shape) + _field(2, 1) + _field(8, "E") + _field(9, table.tobytes())
    gather = _field(1, "E") + _field(1, "input_ids") + _field(2, "emb") + _field(4, "Gather")
    reduce = (_field(1, "emb") + _field(2, "logits") + _field(4, "ReduceMean")
              + _field(5, _field(1, "axes") + _field(8, 1) + _field(20, 7))
              + _field(5, _field(1, "keepdims") + _field(3, 0) + _field(20, 2)))
    inputs = b"".join(_field(11, _value_info(name, 7, [1, "sequence"])) for name in finbert.INPUTS)
    graph = (_field(1, gather) + _field(1, reduce) + _field(2, "tiny") + _field(5, tensor) + inputs
             + _field(12, _value_info("logits", 1, [1, 3])))
    return _field(1, 8) + _field(7, graph) + _field(8, _field(1, "") + _field(2, 17))


def test_the_real_onnxruntime_scores_a_tiny_model_with_windows_and_weights(tmp_path):
    paths, manifest = write_export(tmp_path, model=tiny_onnx_model())
    config = pinned_config(paths, manifest, window_tokens=4)
    model = Finbert.load(config)
    probs = model.score("good good good good bad")   # windows of 4 and 1 tokens, each between [CLS] and [SEP]
    first = finbert.softmax(np.array([8.0, 0.0, 0.0], dtype=np.float32) / 6)
    second = finbert.softmax(np.array([0.0, 2.0, 0.0], dtype=np.float32) / 3)
    np.testing.assert_allclose([probs.pos, probs.neg, probs.neu], (4 * first + second) / 5, rtol=1e-6)
    assert (probs.windows, probs.tokens) == (2, 5)


def test_two_runs_on_the_real_onnxruntime_give_bitwise_identical_scores(tmp_path):
    paths, manifest = write_export(tmp_path, model=tiny_onnx_model())
    config = pinned_config(paths, manifest, window_tokens=3)
    texts = [("good news today", "bad bad flat"), ("flat", None), ("good " * 40, "bad " * 7), ("", None)]
    runs = [b"".join(struct.pack("<3d", p.pos, p.neg, p.neu) for p in (Finbert.load(config).score(*t)
                                                                       for t in texts)) for _ in range(2)]
    assert runs[0] == runs[1]


# The scorer on a temporary news.db


class FakeClock(LiveClock):
    def __init__(self, ms):
        super().__init__()
        self.ms = ms

    def now_ms(self):
        return self.ms


class FakeModel:
    """Probabilities from the words of a text, recording each call, and an error for any text holding boom."""

    def __init__(self):
        self.calls = []

    def score(self, headline, text=None):
        self.calls.append((headline, text))
        joined = finbert.join_text(headline, text).lower()
        if "boom" in joined:
            raise RuntimeError("a fake model error about " + headline)
        good, bad = joined.count("good"), joined.count("bad")
        pos, neg = (1 + good) / (3 + good + bad), (1 + bad) / (3 + good + bad)
        return Probs(pos, neg, 1 - pos - neg, 1, len(joined.split()))


@pytest.fixture
def conn(tmp_path):
    connection = db.open_writer(tmp_path / "news.db")
    yield connection
    connection.close()


_vendor_ids = iter(range(1, 10**6))


def add_item(conn, title, route="llm", tickers="AAPL", summary=None, body=None, vendor_ts=None, received=NOW,
             kind="media", source="finnhub", status="new"):
    row = {"source": source, "channel": "finnhub_company_AAPL", "vendor_id": str(next(_vendor_ids)), "kind": kind,
           "received_at": iso(received), "detected_at": None, "vendor_ts": vendor_ts, "title": title,
           "summary": summary, "url": None, "publisher": None, "tickers": tickers, "relevance": "title",
           "status": status, "filter_reason": None, "title_hash": "0" * 16, "items_8k": None, "route": route}
    with db.transaction(conn):
        seq = db.insert(conn, "news_items", row)
        if body is not None:
            db.insert(conn, "news_bodies", {"item_seq": seq, "text": body, "chars_total": len(body),
                                            "truncated": 0, "doc_urls": "[]"})
    return seq


def scores(conn):
    return [dict(row) for row in conn.execute("SELECT * FROM item_scores ORDER BY seq")]


def impacts(conn, item_seq):
    return [dict(row) for row in conn.execute("SELECT * FROM score_impacts WHERE item_seq = ? ORDER BY ticker",
                                              (item_seq,))]


def drain(scorer):
    """Score everything pending, as one pass of the run loop does."""
    async def go():
        for item in scorer.pending():
            await scorer.score(item)
    asyncio.run(go())


def test_every_item_routed_to_llm_or_finbert_gets_one_scored_row_with_an_impact_per_ticker(conn):
    model, clock = FakeModel(), FakeClock(NOW + 5 * MS_PER_SECOND)
    llm = add_item(conn, "Good quarter for Apple", tickers="AAPL", summary="good good")
    fin = add_item(conn, "Bad day for chips", route="finbert", tickers="NVDA,QQQ")
    add_item(conn, "Vendor tagged only", route="none", tickers="MSFT")
    add_item(conn, "A seed item", route="none", status="seed")
    scorer = FinbertScorer(CONFIG, conn, clock, model)
    drain(scorer)
    rows = scores(conn)
    assert [(r["item_seq"], r["scorer"], r["route"], r["status"], r["attempts"], r["call_id"]) for r in rows] == [
        (llm, SCORER, "llm", "scored", 1, None), (fin, SCORER, "finbert", "scored", 1, None)]
    assert SCORER == "finbert_v1" == health.FINBERT_SCORER
    first = model.score("Good quarter for Apple", "good good")
    assert (rows[0]["p_pos"], rows[0]["p_neg"], rows[0]["p_neu"]) == (first.pos, first.neg, first.neu)
    assert [(i["ticker"], i["strength"]) for i in impacts(conn, llm)] == [("AAPL", first.strength)]
    second = model.score("Bad day for chips")
    assert [(i["ticker"], i["strength"]) for i in impacts(conn, fin)] == [("NVDA", second.strength),
                                                                          ("QQQ", second.strength)]
    for row in impacts(conn, fin):
        assert row["strength"] == rows[1]["p_pos"] - rows[1]["p_neg"] < 0
        assert all(row[key] is None for key in ("direction", "surprise", "event_type", "confidence", "horizon",
                                                "is_new"))
    assert scorer.counts == {"scored": 2}


def test_filings_and_statements_use_their_body_and_other_items_their_summary(conn):
    model = FakeModel()
    add_item(conn, "8-K filed", kind="8k", source="edgar", summary="atom summary", body="main document text")
    add_item(conn, "Media item", summary="the summary")
    add_item(conn, "Bare headline")
    drain(FinbertScorer(CONFIG, conn, FakeClock(NOW), model))
    assert model.calls == [("8-K filed", "main document text"), ("Media item", "the summary"),
                           ("Bare headline", "")]


def test_every_scorer_guard_gives_stale_vendor_without_running_the_model(conn):
    model = FakeModel()
    hours = CONFIG.news.max_vendor_age_hours
    day_old = iso(NOW - int(hours * MS_PER_HOUR))
    older = iso(NOW - int(hours * MS_PER_HOUR) - 1)
    edge = add_item(conn, "Exactly a day old", vendor_ts=day_old)
    stale = add_item(conn, "Older than a day", vendor_ts=older)
    untagged_and_stale = add_item(conn, "Older and untagged", vendor_ts=older, tickers="")
    drain(FinbertScorer(CONFIG, conn, FakeClock(NOW), model))
    status = {r["item_seq"]: (r["status"], r["attempts"], r["p_pos"]) for r in scores(conn)}
    assert status[stale] == ("stale_vendor", 0, None) and status[untagged_and_stale] == ("stale_vendor", 0, None)
    assert status[edge][0] == "scored"
    assert model.calls == [("Exactly a day old", "")]
    assert impacts(conn, stale) == [] and impacts(conn, untagged_and_stale) == []
    # The one guard of the Phase 14 collector, reused rather than copied.
    assert finbert_scorer.vendor_status is collector.vendor_status
    for item in (edge, stale):
        row = conn.execute("SELECT received_at, vendor_ts FROM news_items WHERE seq = ?", (item,)).fetchone()
        expected = collector.vendor_status(row[0], row[1], hours) or "scored"
        assert status[item][0] == expected


def test_an_item_without_tickers_ends_untagged_without_running_the_model(conn):
    model = FakeModel()
    seq = add_item(conn, "Nothing tagged", tickers="")
    drain(FinbertScorer(CONFIG, conn, FakeClock(NOW), model))
    assert [(r["item_seq"], r["status"], r["attempts"]) for r in scores(conn)] == [(seq, "untagged", 0)]
    assert model.calls == [] and impacts(conn, seq) == []


def test_a_text_the_model_cannot_score_ends_failed_and_the_next_item_is_scored(conn, caplog):
    model = FakeModel()
    bad = add_item(conn, "Secret headline boom")
    good = add_item(conn, "Fine headline")
    with caplog.at_level(logging.ERROR, logger=finbert_scorer.__name__):
        drain(FinbertScorer(CONFIG, conn, FakeClock(NOW), model))
    assert [(r["item_seq"], r["status"], r["attempts"]) for r in scores(conn)] == [(bad, "failed", 1),
                                                                                   (good, "scored", 1)]
    assert "RuntimeError" in caplog.text and "Secret headline" not in caplog.text


def test_scored_at_comes_from_the_one_clock_so_seq_order_is_scored_at_order(conn):
    clock = FakeClock(NOW)
    for i in range(4):
        add_item(conn, f"Item {i}", received=NOW - 1000)
    scorer = FinbertScorer(CONFIG, conn, clock, FakeModel())

    async def go():
        for i, item in enumerate(scorer.pending()):
            clock.ms = NOW + (2 - i) * 1000      # the wall clock steps back, the stamps never do
            await scorer.score(item)

    asyncio.run(go())
    stamped = [parse_iso(r["scored_at"]) for r in scores(conn)]
    assert stamped == sorted(stamped) == [NOW + 2000] * 4
    received = [parse_iso(r[0]) for r in conn.execute("SELECT received_at FROM news_items")]
    assert min(stamped) >= max(received)


def test_a_second_final_row_for_one_item_is_refused(conn):
    seq = add_item(conn, "Once only")
    drain(FinbertScorer(CONFIG, conn, FakeClock(NOW), FakeModel()))
    item = finbert_scorer.Item(seq, "llm", iso(NOW), None, "Once only", "", ("AAPL",))
    with pytest.raises(sqlite3.IntegrityError):
        finbert_scorer.write(conn, item, "scored", Probs(0.5, 0.25, 0.25, 1, 2), 1, NOW)
    assert len(scores(conn)) == 1 and len(impacts(conn, seq)) == 1


def test_a_restart_scores_only_the_routed_items_still_without_a_row(conn):
    first = add_item(conn, "Before the restart")
    drain(FinbertScorer(CONFIG, conn, FakeClock(NOW), FakeModel()))
    second = add_item(conn, "During the outage")
    model = FakeModel()
    drain(FinbertScorer(CONFIG, conn, FakeClock(NOW + 60_000), model))
    assert [r["item_seq"] for r in scores(conn)] == [first, second]
    assert model.calls == [("During the outage", "")]


def test_items_inserted_while_idle_are_still_found(conn):
    scorer = FinbertScorer(CONFIG, conn, FakeClock(NOW), FakeModel())
    add_item(conn, "Not routed", route="none")
    assert scorer.pending() == []
    later = add_item(conn, "Arrives later")
    assert [item.seq for item in scorer.pending()] == [later]


def test_the_run_loop_scores_new_arrivals_and_stops_when_asked(conn):
    add_item(conn, "First")
    arrivals = ["Second", "Third"]
    scorer = FinbertScorer(CONFIG, conn, FakeClock(NOW), FakeModel())

    async def go():
        stop = asyncio.Event()
        waits = []

        async def wait(event, seconds):
            waits.append(seconds)
            if arrivals:
                add_item(conn, arrivals.pop(0))
            else:
                event.set()
            return event.is_set()

        await scorer.run(stop, wait)
        return waits

    waits = asyncio.run(go())
    assert [r["status"] for r in scores(conn)] == ["scored"] * 3
    assert waits == [finbert_scorer.POLL_SECONDS] * 3 and scorer.state == health.ON


def test_scored_rows_let_news_health_stop_waiting_for_a_primary_score(conn):
    heart = health.Heartbeat(CONFIG, conn, Schedule(CONFIG))
    fin = add_item(conn, "FinBERT alone", route="finbert", received=NOW - 1000)
    llm = add_item(conn, "Haiku and FinBERT", received=NOW - 1000)
    assert heart.waiting(NOW).count == 2
    drain(FinbertScorer(CONFIG, conn, FakeClock(NOW), FakeModel()))
    assert heart.waiting(NOW).count == 1      # the llm item still needs its final Haiku row
    db.insert(conn, "item_scores", {"item_seq": llm, "scorer": CONFIG.llm.scorer_name, "route": "llm",
                                    "status": "expired", "attempts": 0, "scored_at": iso(NOW)})
    assert heart.waiting(NOW).count == 0
    assert {r["item_seq"] for r in scores(conn) if r["scorer"] == SCORER} == {fin, llm}


# Imports


def _imports(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_only_finbert_imports_onnxruntime_and_torch_lives_only_in_the_export_script():
    package = sorted((ROOT / "src").rglob("*.py"))
    assert [p.relative_to(ROOT).as_posix() for p in package if "onnxruntime" in _imports(p)] == [
        "src/btc_survivor/v2/news/finbert.py"]
    heavy = {"torch", "transformers", "huggingface_hub", "onnx", "optimum"}
    assert not [p for p in package if _imports(p) & heavy]
    scripts = sorted((ROOT / "scripts").rglob("*.py"))
    assert [p.name for p in scripts if "torch" in _imports(p)] == ["export_finbert.py"]
    # the one exception to section 3's rule, the export's parity check in the throwaway environment
    assert [p.name for p in scripts if "onnxruntime" in _imports(p)] == ["export_finbert.py"]
    tree = ast.parse((ROOT / "scripts" / "export_finbert.py").read_text(encoding="utf-8"))
    inside = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
              and any(isinstance(sub, (ast.Import, ast.ImportFrom)) and "onnxruntime" in
                      {alias.name.split(".")[0] for alias in sub.names} | {(getattr(sub, "module", "") or "")
                                                                              .split(".")[0]}
                      for sub in ast.walk(node))]
    assert inside == ["export"]


def test_the_export_script_imports_its_heavy_packages_only_inside_the_export():
    tree = ast.parse((ROOT / "scripts" / "export_finbert.py").read_text(encoding="utf-8"))
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top.add((node.module or "").split(".")[0])
    assert not top & {"torch", "transformers", "huggingface_hub", "onnx", "onnxruntime", "numpy", "tokenizers",
                      "btc_survivor"}


# The export script, checked without exporting anything


HEAVY = ("torch", "transformers", "huggingface_hub", "onnx")


def test_the_statement_names_the_source_revision_files_licence_packages_and_destinations():
    text = "\n".join(export_finbert.statement(export_finbert.destinations()))
    assert export_finbert.SOURCE_URL == "https://huggingface.co/ProsusAI/finbert"
    assert len(export_finbert.REVISION) == 40 and export_finbert.REVISION in text
    for name in ("config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer_config.json",
                 "vocab.txt", *(f"{n}=={v}" for n, v in export_finbert.PINS.items())):
        assert name in text
    assert "CC BY NC SA 3.0" in text and "without a token" in text
    for name in ("model.onnx", "tokenizer.json", "manifest.json"):
        assert f"data/models/finbert/{name}" in text


def test_the_destinations_are_the_paths_the_config_names():
    paths = export_finbert.destinations()
    settings = CONFIG.finbert
    assert paths == {"model": CONFIG.path(settings.model_file).resolve(),
                     "tokenizer": CONFIG.path(settings.tokenizer_file).resolve(),
                     "manifest": CONFIG.path(settings.manifest_file).resolve()}
    assert paths["model"] == ROOT / "data" / "models" / "finbert" / "model.onnx"


def test_the_pins_match_the_project_where_both_hold_a_package():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    pinned = dict(spec.split("==") for spec in project["optional-dependencies"]["v2news"])
    for name in ("onnxruntime", "tokenizers"):
        assert export_finbert.PINS[name] == pinned[name]
    assert export_finbert.OPSET == 17 and export_finbert.TOLERANCE == 1e-4
    assert export_finbert.LABELS == LABELS and export_finbert.INPUTS == finbert.INPUTS


def test_the_environment_check_refuses_the_project_environment_and_other_versions(tmp_path):
    pins = export_finbert.PINS
    assert export_finbert.environment_problems(tmp_path, False, pins.get) == []
    inside = export_finbert.environment_problems(ROOT / ".venv-v2", True, pins.get)
    assert len(inside) == 2 and "inside the repository" in inside[0] and "project environment" in inside[1]
    cpu = export_finbert.environment_problems(tmp_path, False, {**pins, "torch": pins["torch"] + "+cpu"}.get)
    assert cpu == []
    other = export_finbert.environment_problems(tmp_path, False, {**pins, "torch": "2.13.0", "onnx": None}.get)
    assert other == ["torch is 2.13.0, the export needs torch==2.14.0", "onnx is not installed, the export needs "
                                                                        "onnx==1.23.0"]


def tmp_news_config(tmp_path):
    path = tmp_path / "configs" / "news.toml"
    path.parent.mkdir(parents=True)
    path.write_text('[finbert]\nmodel_file = "../models/model.onnx"\ntokenizer_file = "../models/tokenizer.json"\n'
                    'manifest_file = "../models/manifest.json"\n', encoding="utf-8")
    return path


def test_preflight_refuses_existing_files_and_a_work_folder_inside_the_repository_or_not_empty(tmp_path):
    paths = export_finbert.destinations(tmp_news_config(tmp_path))
    assert paths["model"] == (tmp_path / "models" / "model.onnx").resolve()
    paths["tokenizer"].parent.mkdir()
    paths["tokenizer"].write_text("{}", encoding="utf-8")
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "file").write_text("x", encoding="utf-8")
    problems = " ".join(export_finbert.preflight(paths, ROOT / "build" / "finbert_work"))
    assert "tokenizer.json exists" in problems and "outside the repository" in problems
    assert "must be empty" in " ".join(export_finbert.preflight(paths, busy))


def test_without_confirm_the_script_only_states_and_touches_nothing(tmp_path, capsys):
    before = set(sys.modules)
    config = tmp_news_config(tmp_path)
    assert export_finbert.main(["--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert export_finbert.REVISION in out and "nothing done" in out
    assert not (tmp_path / "models").exists()
    assert not {name for name in set(sys.modules) - before if name.split(".")[0] in HEAVY}


def test_confirm_in_the_project_environment_is_refused_before_anything_is_imported(tmp_path, capsys):
    before = set(sys.modules)
    config = tmp_news_config(tmp_path)
    work = tmp_path / "work"
    assert export_finbert.main(["--confirm", "--config", str(config), "--work", str(work)]) == 2
    err = capsys.readouterr().err
    assert "refused" in err and "inside the repository" in err and "project environment" in err
    assert not (tmp_path / "models").exists() and not work.exists()
    assert not {name for name in set(sys.modules) - before if name.split(".")[0] in HEAVY}


def test_the_check_sentences_are_fifty_distinct_fictional_sentences():
    sentences = export_finbert.CHECK_SENTENCES
    assert len(sentences) == 50 == len(set(sentences))
    prefilter = Prefilter(CONFIG)
    assert [s for s in sentences if prefilter.symbols(s)] == []
    assert all(s.strip() == s and s.endswith(".") for s in sentences)


def test_the_manifest_the_script_writes_is_the_one_the_runtime_accepts(tmp_path):
    paths, manifest = write_export(tmp_path)
    assert manifest["revision"] == export_finbert.REVISION and manifest["labels"] == list(LABELS)
    assert manifest["model"] == {"file": "model.onnx", "sha256": finbert.sha256(MODEL_BYTES),
                                 "bytes": len(MODEL_BYTES)}
    assert manifest["check_sentences"] == list(export_finbert.CHECK_SENTENCES)
    verified = finbert.verify(pinned_config(paths, manifest))
    assert verified.model == MODEL_BYTES and verified.labels == LABELS


def test_place_renames_only_after_every_copy_is_checked_in_a_folder_for_the_owner_alone(tmp_path):
    sources = {}
    for role in export_finbert.ROLES:
        sources[role] = tmp_path / f"{role}.src"
        sources[role].write_bytes(f"checked {role} bytes".encode())
    paths = {role: tmp_path / "out" / "deep" / f"{role}.bin" for role in export_finbert.ROLES}
    export_finbert.place(sources, paths)
    for role, target in paths.items():
        assert target.read_bytes() == f"checked {role} bytes".encode()
        assert target.stat().st_mode & 0o777 == 0o600
    assert target.parent.stat().st_mode & 0o777 == 0o700
    assert sorted(p.name for p in target.parent.iterdir()) == ["manifest.bin", "model.bin", "tokenizer.bin"]
    missing = {**sources, "manifest": tmp_path / "absent.src"}
    fresh = {role: tmp_path / "fresh" / f"{role}.bin" for role in export_finbert.ROLES}
    with pytest.raises(FileNotFoundError):
        export_finbert.place(missing, fresh)
    assert not any(target.exists() for target in fresh.values())


# On the exported model, skipped while it is absent


EXPORTED = [CONFIG.path(getattr(CONFIG.finbert, key)) for key in ("model_file", "tokenizer_file", "manifest_file")]
needs_model = pytest.mark.skipif(not all(path.is_file() for path in EXPORTED),
                                 reason="the exported FinBERT model is absent, scripts/export_finbert.py writes it")


def exported_config():
    """The committed config, pinned from the manifest while its pins are still empty."""
    manifest = json.loads(EXPORTED[2].read_text(encoding="utf-8"))
    if CONFIG.finbert.model_sha256:
        return CONFIG, manifest
    pins = tomllib.loads("\n".join(export_finbert.pins_text(manifest)))
    return replace(CONFIG, finbert=replace(CONFIG.finbert, **pins)), manifest


def check_texts(manifest):
    sentences = manifest["check_sentences"]
    return [(s, None) for s in sentences] + [(sentences[0], " ".join(sentences) * 3)]


@needs_model
def test_two_runs_on_the_exported_model_give_bitwise_identical_scores():
    config, manifest = exported_config()
    runs = []
    for _ in range(2):
        model = Finbert.load(config)
        runs.append(b"".join(struct.pack("<3d", p.pos, p.neg, p.neu)
                             for p in (model.score(headline, text) for headline, text in check_texts(manifest))))
        del model
    assert runs[0] == runs[1]


@needs_model
def test_the_exported_model_matches_the_pytorch_reference_of_the_export():
    config, manifest = exported_config()
    model = Finbert.load(config)
    labels = manifest["labels"]
    for sentence, logits in zip(manifest["check_sentences"], manifest["reference_logits"]):
        reference = finbert.softmax(np.array(logits))
        probs = model.score(sentence)
        expected = [reference[labels.index(label)] for label in LABELS]
        np.testing.assert_allclose([probs.pos, probs.neg, probs.neu], expected, atol=1e-4, rtol=0)
        assert probs.windows == 1
    assert model.score(*check_texts(manifest)[-1]).windows > 1
