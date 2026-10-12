"""The contact address must appear in no file the repository tracks or would add, and in no version the index holds.
The scan knows only the address's SHA 256, its length and where its @ sits, so the address never enters an assert,
a failure message or the output.

The real scan runs in a child process that prints only file names, because pytest shows the arguments of the frames
of a failing test, and a frame of the secrets reader or of the scan would otherwise hold the file text or the address.
Seeded dashboard JSON joins the scan when the private dashboard is built.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from helpers import ROOT

TESTS = Path(__file__).resolve().parent


def fingerprint(address):
    """The SHA 256 of the lower case address, its length in bytes and the offset of its @."""
    data = address.lower().encode("utf-8")
    return hashlib.sha256(data).hexdigest(), len(data), data.index(b"@")


def holds(digest, length, at, data):
    """Whether the window around some @ in data hashes to digest, compared in lower case."""
    data = data.lower()
    position = data.find(b"@")
    while position != -1:
        start = position - at
        if start >= 0 and hashlib.sha256(data[start:start + length]).hexdigest() == digest:
            return True
        position = data.find(b"@", position + 1)
    return False


def files_holding(digest, length, at, paths, root):
    """Names of files that hold the address."""
    return [str(path.relative_to(root)) for path in paths if path.is_file() and holds(digest, length, at,
                                                                                       path.read_bytes())]


def repository_files(root=ROOT):
    """Tracked files plus untracked ones that are not ignored, so a file about to be committed is scanned too."""
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root,
                         capture_output=True, check=True).stdout
    return [root / name for name in out.decode("utf-8").split("\0") if name]


def index_blobs(root=ROOT):
    """Each file's name and contents as the index holds them, which may differ from the working tree."""
    out = subprocess.run(["git", "ls-files", "-s", "-z"], cwd=root, capture_output=True, check=True).stdout
    entries = [entry.split(b"\t", 1) for entry in out.split(b"\0") if entry]
    if not entries:
        return []
    shas = b"".join(meta.split()[1] + b"\n" for meta, _ in entries)
    batch = subprocess.run(["git", "cat-file", "--batch"], cwd=root, input=shas, capture_output=True,
                           check=True).stdout
    blobs, offset = [], 0
    for _, name in entries:
        end = batch.index(b"\n", offset)
        header = batch[offset:end].split()
        if header[-1] == b"missing":
            offset = end + 1
            continue
        size = int(header[2])
        blobs.append((name.decode("utf-8"), batch[end + 1:end + 1 + size]))
        offset = end + 1 + size + 1
    return blobs


def index_holding(digest, length, at, root=ROOT):
    """Names of staged files whose staged version holds the address."""
    return [name for name, data in index_blobs(root) if holds(digest, length, at, data)]


def test_the_scan_finds_a_planted_fake_address(tmp_path):
    fake = "planted.person@example.invalid"
    (tmp_path / "upper.txt").write_text(f"write to {fake.upper()} today", encoding="utf-8")
    (tmp_path / "inside.bin").write_bytes(b"\x00\xffx" + fake.encode() + b"y\x00@")
    (tmp_path / "near.txt").write_text("planted.person@example.invalid.other is close, @ alone too", encoding="utf-8")
    (tmp_path / "other.txt").write_text("@ first, then someone.else@example.invalid", encoding="utf-8")
    paths = sorted(tmp_path.iterdir()) + [tmp_path / "absent.txt"]
    assert files_holding(*fingerprint(fake), paths, tmp_path) == ["inside.bin", "near.txt", "upper.txt"]


def test_the_scan_finds_a_fake_address_left_only_in_the_index(tmp_path):
    fake = "planted.person@example.invalid"
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "staged.txt").write_text(f"contact {fake}\n", encoding="utf-8")
    (tmp_path / "clean.txt").write_text("nothing here @ all\n", encoding="utf-8")
    subprocess.run(["git", "add", "staged.txt", "clean.txt"], cwd=tmp_path, check=True)
    (tmp_path / "staged.txt").write_text("contact removed from the working tree only\n", encoding="utf-8")
    found = fingerprint(fake)
    assert files_holding(*found, repository_files(tmp_path), tmp_path) == []
    assert index_holding(*found, tmp_path) == ["staged.txt"]


def test_the_scan_covers_this_repository():
    names = {str(path.relative_to(ROOT)) for path in repository_files()}
    assert {"PLAN_V2.md", "pyproject.toml", "tests/test_v2_contact_absent.py"} <= names
    assert not any(name.startswith(("data/", ".venv")) for name in names)
    assert "PLAN_V2.md" in {name for name, _ in index_blobs()}


# Prints one line per hit and then done, or skip and a reason, or error and an exception type, and never a value.
REAL_SCAN = """
import sys
sys.path.insert(0, sys.argv[1])
try:
    from btc_survivor.v2.local_secrets import CONTACT, DEFAULT_PATH, load_secrets
    import test_v2_contact_absent as scan
    if not DEFAULT_PATH.expanduser().exists():
        print("skip no secrets file on this machine")
        sys.exit(0)
    secrets = load_secrets(DEFAULT_PATH)
    if CONTACT not in secrets.present:
        print("skip " + CONTACT + " is not in the secrets file")
        sys.exit(0)
    found = scan.fingerprint(secrets.get(CONTACT))
    del secrets
    for name in scan.files_holding(*found, scan.repository_files(), scan.ROOT):
        print("hit " + name)
    for name in scan.index_holding(*found):
        print("hit index:" + name)
    print("done")
except Exception as exc:
    print("error " + type(exc).__name__)
"""


def test_the_real_contact_address_is_in_no_repository_file():
    result = subprocess.run([sys.executable, "-c", REAL_SCAN, str(TESTS)], capture_output=True, text=True,
                            timeout=300, cwd=ROOT)
    lines = result.stdout.splitlines()
    if lines and lines[0].startswith("skip "):
        pytest.skip(lines[0].removeprefix("skip "))
    errors = [line for line in lines if line.startswith("error ")]
    assert not errors and lines[-1:] == ["done"], "the scan could not finish"
    hits = [line.removeprefix("hit ") for line in lines if line.startswith("hit ")]
    assert hits == [], "files holding the contact address"
