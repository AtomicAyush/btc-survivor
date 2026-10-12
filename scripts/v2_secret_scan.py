"""The secret scan of PLAN_V2 section 14.4, run before every push.

It takes every value the secrets reader holds, without naming a key, and looks for each one, ignoring case, in every
file the repository tracks or would add, every staged version in the index, every file under data/v2, the log folders
given, data/logs/v2 with the launchd logs by default, and every object from the remote branch to HEAD, which covers
each blob, tree, commit message and author and committer field. Gzip files are read decompressed as a stream, so a
recording still being written is read up to its last flush and a file of several gzip members is read whole. It prints
only counts and paths, never a value, and exits 1 when anything holds a value.

    python scripts/v2_secret_scan.py [--logs FOLDER ...] [--base REF]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zlib
from collections.abc import Iterable, Iterator
from pathlib import Path

from btc_survivor.v2.local_secrets import DEFAULT_PATH, SecretsError, load_secrets

ROOT = Path(__file__).resolve().parent.parent
REMOTE_BRANCH = "origin/feat/v2-stocks"
FORK_BRANCHES = ("origin/feat/initial-bot", "feat/initial-bot")
CHUNK_BYTES = 1 << 24
GZIP_READ_BYTES = 1 << 20


def git(root: Path, *args: str, data: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], cwd=root, input=data, capture_output=True, check=True).stdout


def secret_values(path: Path) -> list[bytes]:
    """Every value in the secrets file in lower case, taken by the names the reader reports present."""
    secrets = load_secrets(path)
    return [secrets.get(name).lower().encode("utf-8") for name in secrets.present]


def holds(values: list[bytes], data: bytes) -> bool:
    data = data.lower()
    return any(value in data for value in values)


def chunks(path: Path) -> Iterator[bytes]:
    """The file's contents in pieces, inflating gzip member by member and stopping quietly where a file still being
    written ends, since the gzip module refuses a file without its end marker."""
    with path.open("rb") as fh:
        if path.suffix != ".gz":
            while chunk := fh.read(CHUNK_BYTES):
                yield chunk
            return
        inflate = zlib.decompressobj(zlib.MAX_WBITS | 16)
        while raw := fh.read(GZIP_READ_BYTES):
            while raw:
                yield inflate.decompress(raw)
                raw = inflate.unused_data if inflate.eof else b""
                if inflate.eof:
                    inflate = zlib.decompressobj(zlib.MAX_WBITS | 16)
        yield inflate.flush()


def file_holds(values: list[bytes], path: Path) -> bool:
    """Read a file in chunks that overlap by one value length, so a value split across two chunks is still found."""
    overlap = max(map(len, values)) - 1
    tail = b""
    for chunk in chunks(path):
        data = (tail + chunk).lower()
        if holds(values, data):
            return True
        tail = data[max(0, len(data) - overlap):]
    return False


def files_under(folders: Iterable[Path]) -> list[Path]:
    return sorted(p for folder in folders if folder.is_dir() for p in folder.rglob("*") if p.is_file())


def objects(root: Path, ids: list[bytes]) -> list[tuple[bytes, bytes, bytes]]:
    """Each object's id, type and raw contents from one git cat-file batch, skipping ids git does not have."""
    if not ids:
        return []
    out = git(root, "cat-file", "--batch", data=b"".join(i + b"\n" for i in ids))
    found, offset = [], 0
    while offset < len(out):
        end = out.index(b"\n", offset)
        header = out[offset:end].split()
        if header[-1] == b"missing":
            offset = end + 1
            continue
        size = int(header[2])
        found.append((header[0], header[1], out[end + 1:end + 1 + size]))
        offset = end + 1 + size + 1
    return found


def push_base(root: Path) -> str | None:
    """The remote branch, else the fork commit with feat/initial-bot, else None to scan all of HEAD's history."""
    for ref in (REMOTE_BRANCH, *FORK_BRANCHES):
        done = subprocess.run(["git", "merge-base", "HEAD", ref], cwd=root, capture_output=True, text=True)
        if done.returncode == 0:
            return ref if ref == REMOTE_BRANCH else done.stdout.strip()
    return None


def scan(root: Path, values: list[bytes], logs: list[Path], base: str | None) -> tuple[list[str], list[str]]:
    """The count lines and the hit lines, each hit named by where it was found and its path or object id."""
    counts, hits = [], []
    names = [n for n in git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split(b"\0") if n]
    tree = [root / n.decode("utf-8") for n in names]
    hits += [f"file {p.relative_to(root)}" for p in tree if p.is_file() and file_holds(values, p)]
    counts.append(f"working tree files: {len(tree)}")

    staged: dict[bytes, list[str]] = {}
    for entry in git(root, "ls-files", "-s", "-z").split(b"\0"):
        if entry:
            meta, name = entry.split(b"\t", 1)
            staged.setdefault(meta.split()[1], []).append(name.decode("utf-8"))
    bad = [oid for oid, _, data in objects(root, list(staged)) if holds(values, data)]
    hits += [f"index {name}" for oid in bad for name in staged[oid]]
    counts.append(f"index entries: {sum(map(len, staged.values()))}")

    for label, folders in (("data/v2", [root / "data" / "v2"]), ("log", logs)):
        found = files_under(folders)
        hits += [f"{label} {p}" for p in found if file_holds(values, p)]
        counts.append(f"{label} files: {len(found)}")

    span = f"{base}..HEAD" if base else "HEAD"
    listed = [line.split(b" ", 1) for line in git(root, "rev-list", "--objects", span).splitlines() if line]
    where = {parts[0]: parts[1].decode("utf-8") if len(parts) > 1 else "" for parts in listed}
    pushed = objects(root, list(where))
    hits += [f"{kind.decode()} {oid.decode()} {where[oid]}".rstrip() for oid, kind, data in pushed
             if holds(values, data)]
    commits = sum(kind == b"commit" for _, kind, _ in pushed)
    counts.append(f"objects to push: {len(pushed)}, of which commits: {commits}")
    return counts, hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Look for every secret value in the repository, data/v2, the logs "
                                                 "and the push range, printing only counts and paths.")
    parser.add_argument("--secrets-file", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--root", type=Path, default=ROOT, help="the repository to scan")
    parser.add_argument("--logs", type=Path, nargs="*", help="log folders, by default data/logs/v2 of the repository, "
                                                             "which holds the launchd logs")
    parser.add_argument("--base", help="scan objects from this ref to HEAD, by default the remote branch")
    args = parser.parse_args(argv)
    try:
        values = secret_values(args.secrets_file)
        if not values:
            print("error: the secrets file holds no value to look for", file=sys.stderr)
            return 2
        logs = args.logs if args.logs is not None else [args.root / "data" / "logs" / "v2"]
        counts, hits = scan(args.root, values, logs, args.base or push_base(args.root))
    except (SecretsError, OSError, subprocess.CalledProcessError) as exc:
        detail = str(exc) if isinstance(exc, SecretsError) else type(exc).__name__
        print(f"error: the scan could not run, {detail}", file=sys.stderr)
        return 2
    print("\n".join(["secret scan", f"values: {len(values)}", *counts, f"hits: {len(hits)}", *hits,
                     f"verdict: {'fail' if hits else 'pass'}"]))
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
