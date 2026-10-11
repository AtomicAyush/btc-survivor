import json

from btc_survivor.synthetic import Fault, SyntheticFeed

START = 1_791_000_000_000


def sequences(lines):
    """Sequence numbers per connection, in order."""
    seqs = {}
    for kind, raw, _, conn in lines:
        if kind == "message":
            seqs.setdefault(conn, []).append(json.loads(raw)["sequence_num"])
    return seqs


def test_same_seed_gives_identical_lines_and_different_seed_does_not():
    a = list(SyntheticFeed(9, START).lines(120))
    b = list(SyntheticFeed(9, START).lines(120))
    c = list(SyntheticFeed(10, START).lines(120))
    assert a == b and a != c


def test_receive_times_never_go_backwards():
    lines = list(SyntheticFeed(1, START).lines(300, (Fault(100, "crossed"), Fault(200, "disconnect", 5))))
    times = [t for _, _, t, _ in lines]
    assert times == sorted(times)


def test_clean_connection_has_contiguous_sequence_numbers():
    seqs = sequences(SyntheticFeed(1, START).lines(120))
    assert list(seqs) == [1]
    assert seqs[1] == list(range(len(seqs[1])))


def test_gap_skips_a_number_and_then_reconnects():
    lines = list(SyntheticFeed(1, START).lines(120, (Fault(60, "gap"),)))
    seqs = sequences(lines)
    assert len(seqs) == 2
    first = seqs[1]
    assert any(b - a == 2 for a, b in zip(first, first[1:]))
    assert [k for k, _, _, _ in lines].count("close") == 1


def test_long_silence_reconnects_and_short_silence_does_not():
    short = list(SyntheticFeed(1, START).lines(60, (Fault(20, "silence", 3),)))
    long = list(SyntheticFeed(1, START).lines(60, (Fault(20, "silence", 30),)))
    assert [k for k, *_ in short].count("open") == 1
    assert [k for k, *_ in long].count("open") == 2


def test_skew_offsets_envelope_timestamps():
    from btc_survivor.clock import parse_iso

    lines = list(SyntheticFeed(1, START).lines(40, (Fault(20, "skew", 10, 5_000),)))
    offsets = [t - parse_iso(json.loads(raw)["timestamp"]) for k, raw, t, _ in lines if k == "message"]
    assert min(offsets) < -4_000 and max(offsets) > 0


def test_silence_just_over_the_stale_limit_reconnects_so_replay_recovers():
    from test_market import drive

    for duration in (4.9, 5.0):
        lines = list(SyntheticFeed(1, START).lines(400, (Fault(150, "silence", duration),)))
        assert [k for k, *_ in lines].count("open") == 2, duration
        _, minutes, _ = drive(lines)
        assert minutes[-1][1], duration
