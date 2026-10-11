"""The three run configs differ from the example only where PLAN section 21 says they do."""

import dataclasses

import pytest

from btc_survivor.config import load_config
from helpers import EXAMPLE_CONFIG, ROOT

EXAMPLE = load_config(EXAMPLE_CONFIG)


def differences(cfg):
    out = {}
    for section in ("run", "feed", "time", "features", "model", "trading", "costs", "genes", "population",
                    "controls", "persistence"):
        a, b = dataclasses.asdict(getattr(EXAMPLE, section)), dataclasses.asdict(getattr(cfg, section))
        out.update({f"{section}.{k}": b[k] for k in a if a[k] != b[k]})
    return out


@pytest.mark.parametrize("name, changes", [
    ("run_a", {}),
    ("run_b", {"costs.fee_profile": "low_fee_tier"}),
    ("run_c", {"trading.start_equity": 500.0}),
])
def test_each_run_changes_only_its_own_settings(name, changes):
    cfg = load_config(ROOT / "configs" / f"{name}.toml")
    assert differences(cfg) == {"run.data_dir": f"../data/{name}", **changes}
    assert cfg.data_dir.resolve() == (ROOT / "data" / name).resolve()
