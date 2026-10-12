"""Default command output holds only counts, verdicts, hashes, timings and state words. Canary prices, volumes
and fills seeded in a database appear only with --detail."""

import argparse
import hashlib
import io
import sqlite3

import numpy as np
import pytest

from btc_survivor.v2 import output, redact
from btc_survivor.v2.output import Count, Hash, Report, Seconds, Word, render

CANARY_PRICE = 187.4321
CANARY_VOLUME = 6543217
CANARY_FILL = 187.4399
CANARIES = (str(CANARY_PRICE), str(CANARY_VOLUME), str(CANARY_FILL))


def seeded_db(folder):
    db = sqlite3.connect(folder / "run.db")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE fills (ts TEXT, symbol TEXT, side TEXT, price REAL, volume INTEGER, fill REAL)")
    db.executemany("INSERT INTO fills VALUES (?, ?, ?, ?, ?, ?)", [
        ("2026-11-02T14:31:00Z", "AAPL", "buy", CANARY_PRICE, CANARY_VOLUME, CANARY_FILL),
        ("2026-11-02T14:32:00Z", "MSFT", "sell", 1.0, 2, 3.0),
    ])
    return db


def fills_report(db):
    """A command shaped report over the seeded rows, the way a version 2 command builds one."""
    rows = tuple(dict(row) for row in db.execute("SELECT * FROM fills ORDER BY ts"))
    return Report("fills", summary=(
        ("fill rows", Count(len(rows))),
        ("symbols", Count(np.int64(len({row["symbol"] for row in rows})))),
        ("schema check", True),
        ("order check", False),
        ("config hash", Hash(hashlib.sha256(b"config").hexdigest())),
        ("query time", Seconds(0.0123)),
        ("feed state", Word("healthy")),
    ), rows=rows)


def found(text):
    return [canary for canary in CANARIES if canary in text]


def test_default_output_holds_no_canary_from_a_seeded_database(tmp_path):
    text = render(fills_report(seeded_db(tmp_path)))
    assert found(text) == []
    assert text.splitlines() == [
        "fills",
        "fill rows: 2",
        "symbols: 2",
        "schema check: pass",
        "order check: fail",
        "config hash: " + hashlib.sha256(b"config").hexdigest(),
        "query time: 0.012 s",
        "feed state: healthy",
        "2 rows hidden, pass --detail in your own terminal to see them",
    ]


def test_detail_adds_the_rows(tmp_path):
    text = render(fills_report(seeded_db(tmp_path)), detail=True)
    assert found(text) == list(CANARIES)
    lines = text.splitlines()
    assert lines[8].split() == ["ts", "symbol", "side", "price", "volume", "fill"]
    assert lines[9].split()[1:] == ["AAPL", "buy", *CANARIES]
    assert lines[8].index("symbol") == lines[9].index("AAPL")
    assert "hidden" not in text


def test_emit_writes_the_rendered_report(tmp_path):
    report = fills_report(seeded_db(tmp_path))
    stream = io.StringIO()
    output.emit(report, stream=stream)
    assert stream.getvalue() == render(report) + "\n"
    assert found(stream.getvalue()) == []


@pytest.mark.parametrize("value", [CANARY_PRICE, np.float64(CANARY_FILL), str(CANARY_VOLUME), CANARY_VOLUME,
                                   np.int64(CANARY_VOLUME), 18743210, None, [1]])
def test_the_summary_refuses_anything_but_aggregates(value):
    with pytest.raises(TypeError, match="detail rows"):
        render(Report("t", summary=(("figure", value),)))


def test_a_volume_read_from_the_seeded_database_is_refused_in_the_summary(tmp_path):
    volume = seeded_db(tmp_path).execute("SELECT volume FROM fills WHERE symbol = 'AAPL'").fetchone()[0]
    with pytest.raises(TypeError, match="detail rows"):
        render(Report("fills", summary=(("volume", volume),)))


@pytest.mark.parametrize("value", [CANARY_PRICE, str(CANARY_VOLUME), True, None])
def test_a_count_holds_only_an_integer(value):
    with pytest.raises(TypeError):
        Count(value)


@pytest.mark.parametrize("make, text", [(Word, "price 187"), (Word, "Healthy"), (Word, "a.b"),
                                        (Hash, "187.4321"), (Hash, "abc"), (Hash, "ABCDEF0123"), (Hash, "18743210"),
                                        (Hash, hashlib.sha256(b"x").hexdigest()[:63])])
def test_words_and_hashes_cannot_carry_digits_or_decimals(make, text):
    with pytest.raises(ValueError):
        make(text)


@pytest.mark.parametrize("title, label", [(f"fills at {CANARY_PRICE}", "rows"), ("fills", f"volume {CANARY_VOLUME}"),
                                          ("fills", "price 187"), ("Fills", "rows"), ("fills", "")])
def test_titles_and_labels_cannot_carry_a_figure(title, label):
    with pytest.raises(ValueError):
        render(Report(title, summary=((label, Count(1)),)))


def test_rows_with_missing_keys_and_no_rows():
    assert output.table(()) == ["no rows"]
    assert output.table(({"a": 1}, {"b": "xy"})) == ["a  b", "1", "   xy"]
    assert render(Report("empty"), detail=True) == "empty\nno rows"
    assert render(Report("empty")) == "empty"


def test_rendered_output_masks_registered_secrets(monkeypatch):
    monkeypatch.setattr(redact, "_values", set())
    redact.register("FAKEtok3n0003")
    text = render(Report("title", rows=({"note": "FAKEtok3n0003"}, {"note": "url?token=FAKEtok3n0003"})), detail=True)
    assert "FAKEtok3n0003" not in text and text.count(redact.MASK) == 2


def test_detail_flag_is_off_by_default():
    parser = argparse.ArgumentParser()
    output.add_detail_flag(parser)
    assert parser.parse_args([]).detail is False
    assert parser.parse_args(["--detail"]).detail is True
    text = " ".join(parser.format_help().split())
    assert "only in your own terminal and never in an assistant session" in text
