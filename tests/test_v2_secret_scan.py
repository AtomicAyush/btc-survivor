"""The pre push secret scan, on a throwaway repository and a secrets file holding fake values only."""

import gzip
import subprocess
import sys
import zlib

import pytest

from btc_survivor.v2 import redact
from btc_survivor.v2.local_secrets import ANTHROPIC, CONTACT, FINNHUB
from helpers import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from v2_secret_scan import file_holds, main  # noqa: E402

FAKE = {FINNHUB: "fakefinnhub0001token", ANTHROPIC: "fake-anthropic-0002-value", CONTACT: "fake.owner@example.invalid"}
GIT = ["git", "-c", "user.name=Fake Owner", "-c", "user.email=owner@example.invalid", "-c", "commit.gpgsign=false"]


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(redact, "_values", set())


def git(repo, *args):
    subprocess.run([*GIT, *args], cwd=repo, check=True, capture_output=True)


def secrets_file(folder):
    path = folder / "secrets.env"
    path.write_text("".join(f"{name}={value}\n" for name, value in FAKE.items()), encoding="utf-8")
    path.chmod(0o600)
    return path


def clean_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / ".gitignore").write_text("data/\n", encoding="utf-8")
    (repo / "readme.txt").write_text("nothing secret @ here\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "first")
    return repo


def run(repo, secrets, capsys, *extra):
    code = main(["--root", str(repo), "--secrets-file", str(secrets), *extra])
    out = capsys.readouterr()
    assert not [name for name, value in FAKE.items() if value.lower() in (out.out + out.err).lower()]
    return code, out.out.splitlines()


def test_a_clean_repository_passes_with_counts_only(tmp_path, capsys):
    repo = clean_repo(tmp_path)
    code, lines = run(repo, secrets_file(tmp_path), capsys)
    assert code == 0
    assert lines == ["secret scan", "values: 3", "working tree files: 2", "index entries: 2", "data/v2 files: 0",
                     "log files: 0", "objects to push: 4, of which commits: 1", "hits: 0", "verdict: pass"]


def test_every_place_a_value_can_hide_is_found(tmp_path, capsys):
    repo = clean_repo(tmp_path)
    (repo / "old.txt").write_text(f"key {FAKE[FINNHUB]}\n", encoding="utf-8")
    git(repo, "add", "old.txt")
    git(repo, "commit", "-q", "-m", "add old")
    git(repo, "rm", "-q", "old.txt")
    git(repo, "commit", "-q", "-m", f"remove old, write to {FAKE[CONTACT].upper()}")
    (repo / "staged.txt").write_text(f"key {FAKE[ANTHROPIC]}\n", encoding="utf-8")
    git(repo, "add", "staged.txt")
    (repo / "staged.txt").write_text("cleaned in the working tree only\n", encoding="utf-8")
    (repo / "new.txt").write_text(f"untracked {FAKE[CONTACT]}\n", encoding="utf-8")
    (repo / "data" / "v2").mkdir(parents=True)
    with gzip.open(repo / "data" / "v2" / "rec.jsonl.gz", "wb") as fh:
        fh.write(f'{{"close": "token {FAKE[FINNHUB]}"}}\n'.encode())
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "news.log").write_bytes(b"x" * ((1 << 24) - 5) + FAKE[ANTHROPIC].encode())
    code, lines = run(repo, secrets_file(tmp_path), capsys, "--logs", str(logs))
    assert code == 1 and lines[-1] == "verdict: fail"
    start = next(i for i, line in enumerate(lines) if line.startswith("hits: "))
    hits = lines[start + 1:-1]
    assert lines[start] == f"hits: {len(hits)}" and len(hits) == 6
    assert "file new.txt" in hits and "index staged.txt" in hits and "blob" in " ".join(hits)
    assert any(line.startswith("blob ") and line.endswith(" old.txt") for line in hits)
    assert sum(line.startswith("commit ") for line in hits) == 1
    assert any(line.startswith("data/v2 ") and line.endswith("rec.jsonl.gz") for line in hits)
    assert any(line.startswith("log ") and line.endswith("news.log") for line in hits)


def test_only_the_push_range_is_scanned(tmp_path, capsys):
    repo = clean_repo(tmp_path)
    (repo / "old.txt").write_text(f"key {FAKE[FINNHUB]}\n", encoding="utf-8")
    git(repo, "add", "old.txt")
    git(repo, "commit", "-q", "-m", "add old")
    git(repo, "rm", "-q", "old.txt")
    git(repo, "commit", "-q", "-m", "remove old")
    git(repo, "tag", "pushed")
    (repo / "later.txt").write_text("later\n", encoding="utf-8")
    git(repo, "add", "later.txt")
    git(repo, "commit", "-q", "-m", "later")
    code, lines = run(repo, secrets_file(tmp_path), capsys, "--base", "pushed")
    assert code == 0 and "objects to push: 3, of which commits: 1" in lines


def test_an_unsafe_secrets_file_stops_the_scan_without_a_value(tmp_path, capsys):
    repo = clean_repo(tmp_path)
    secrets = secrets_file(tmp_path)
    secrets.chmod(0o644)
    code, lines = run(repo, secrets, capsys)
    assert code == 2 and lines == []


def test_a_recording_still_being_written_is_read_up_to_its_last_flush(tmp_path):
    path = tmp_path / "live.jsonl.gz"
    with path.open("wb") as raw:
        fh = gzip.GzipFile(fileobj=raw, mode="wb")
        fh.write(b"x" * 300_000 + f"token {FAKE[FINNHUB]}\n".encode())
        fh.flush(zlib.Z_SYNC_FLUSH)
        raw.flush()
        assert file_holds([FAKE[FINNHUB].encode()], path)
        assert not file_holds([FAKE[ANTHROPIC].encode()], path)


def test_every_member_of_a_gzip_file_is_read(tmp_path):
    path = tmp_path / "joined.jsonl.gz"
    path.write_bytes(gzip.compress(b"first part\n") + gzip.compress(f"second {FAKE[CONTACT]}\n".encode()))
    assert file_holds([FAKE[CONTACT].encode()], path)
