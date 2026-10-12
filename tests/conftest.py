import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session")
def selecting_run(tmp_path_factory):
    """Three replayed hours where a round trip costs 8 percent of start equity, so learners die, are replaced, and
    some parents have a skill correlation when they reproduce. Shared by the read only analysis tests."""
    from btc_survivor.replay import replay_lines
    from enginekit import selecting_config, synthetic

    folder = tmp_path_factory.mktemp("selecting")
    replay_lines(selecting_config(folder / "unused"), synthetic(5, 3 * 3600), folder / "run_s")
    return selecting_config(folder / "run_s")
