"""Version 1 must not move. Against the fork commit, the merge base with feat/initial-bot, nothing may change
or appear under src/btc_survivor outside v2 or in the version 1 run configs."""

import subprocess

from helpers import ROOT

V1_PATHS = ["src/btc_survivor", ":(exclude)src/btc_survivor/v2", "configs/run_*.toml"]


def _git(*args: str) -> list[str]:
    """Run git in the repository and return its output lines."""
    done = subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True, text=True)
    return done.stdout.splitlines()


def _fork_commit() -> str:
    """The merge base of HEAD with feat/initial-bot, from the local branch or else its remote copy."""
    for ref in ("feat/initial-bot", "origin/feat/initial-bot"):
        done = subprocess.run(["git", "-C", str(ROOT), "merge-base", "HEAD", ref], capture_output=True, text=True)
        if done.returncode == 0:
            return done.stdout.strip()
    raise AssertionError("no merge base with feat/initial-bot, so the fork commit is unknown")


def test_version_1_source_and_run_configs_match_the_fork_commit():
    changed = _git("diff", "--name-only", _fork_commit(), "--", *V1_PATHS)
    added = _git("ls-files", "--others", "--exclude-standard", "--", *V1_PATHS)
    assert not changed, "version 1 files changed since the fork commit\n" + "\n".join(changed)
    assert not added, "new files under version 1 paths\n" + "\n".join(added)


def test_the_paths_cover_version_1_and_leave_out_version_2():
    listed = set(_git("ls-files", "--cached", "--others", "--exclude-standard", "--", *V1_PATHS))
    assert {"src/btc_survivor/engine.py", "src/btc_survivor/dashboard.html", "src/btc_survivor/feed/coinbase.py",
            "configs/run_a.toml", "configs/run_b.toml", "configs/run_c.toml", "configs/run_d.toml"} <= listed
    assert not [p for p in listed if p.startswith("src/btc_survivor/v2/")]
