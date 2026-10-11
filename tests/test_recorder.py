import gzip

from btc_survivor.clock import MS_PER_DAY, iso
from btc_survivor.recorder import Recorder, read_recording, recording_files

DAY_END = 1_791_072_000_000  # a UTC midnight


def test_round_trip_keeps_order_and_fields(tmp_path):
    rec = Recorder(tmp_path)
    rows = [("open", "", DAY_END - 3000, 1), ("message", '{"a": "b\\"c"}', DAY_END - 2000, 1),
            ("close", "server closed", DAY_END - 1000, 1)]
    for row in rows:
        rec.write(*row)
    rec.close()
    assert list(read_recording([tmp_path])) == rows


def test_files_rotate_at_utc_midnight_and_read_back_in_order(tmp_path):
    rec = Recorder(tmp_path)
    rec.write("open", "", DAY_END - 1, 1)
    rec.write("message", "x", DAY_END, 1)
    rec.write("message", "y", DAY_END + MS_PER_DAY, 1)
    rec.close()
    files = recording_files([tmp_path])
    assert DAY_END % MS_PER_DAY == 0
    assert [f.name[:10] for f in files] == [iso(t)[:10] for t in (DAY_END - 1, DAY_END, DAY_END + MS_PER_DAY)]
    assert len({f.name[:10] for f in files}) == 3
    assert [r[1] for r in read_recording([tmp_path])] == ["", "x", "y"]


def test_truncated_tail_ends_the_file_cleanly(tmp_path):
    rec = Recorder(tmp_path)
    for i in range(2000):
        rec.write("message", f'{{"n": {i}, "pad": "{"z" * 50}"}}', DAY_END + i, 1)
    rec.close()
    path = recording_files([tmp_path])[0]
    data = path.read_bytes()
    path.write_bytes(data[: len(data) * 2 // 3])
    rows = list(read_recording([path]))
    assert 0 < len(rows) < 2000
    assert [int(r[1].split(":")[1].split(",")[0]) for r in rows] == list(range(len(rows)))


def test_partial_last_line_is_skipped(tmp_path):
    path = tmp_path / "2026-10-03_000000.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write('{"t": 1, "k": "message", "c": 1, "d": "ok"}\n{"t": 2, "k": "mess')
    assert list(read_recording([path])) == [("message", "ok", 1, 1)]


def test_each_recorder_start_writes_a_new_file(tmp_path):
    for second in (0, 5):
        rec = Recorder(tmp_path)
        rec.write("open", "", DAY_END + second * 1000, 1)
        rec.close()
    assert len(recording_files([tmp_path])) == 2


def test_rows_are_flushed_while_the_recorder_is_still_open(tmp_path):
    rec = Recorder(tmp_path)
    for i in range(30):
        rec.write("message", f"row {i}", DAY_END + i * 1_000, 1)  # 29 s of rows, past the 10 s flush interval
    rows = list(read_recording([tmp_path]))  # read without closing, as after a crash
    assert len(rows) >= 20
    rec.close()
