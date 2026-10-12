"""Version 1 cannot reach version 2: no import of btc_survivor.v2, no shared config, and no model runtime in trading code."""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from btc_survivor.config import ConfigError as V1ConfigError
from btc_survivor.config import load_config as load_v1_config
from btc_survivor.v2.config import ConfigError, load_news_config, load_stock_config
from helpers import ROOT

SRC = ROOT / "src"
PACKAGE = SRC / "btc_survivor"
V2 = PACKAGE / "v2"
V1_FILES = sorted(p for p in PACKAGE.rglob("*.py") if V2 not in p.parents)
TRADING_FILES = sorted(p for p in PACKAGE.rglob("*.py") if V2 / "news" not in p.parents)
V2_CONFIGS = sorted((ROOT / "configs" / "v2").glob("*.toml"))
V1_RUNS = ["run_a", "run_b", "run_c", "run_d"]
MODEL_RUNTIMES = {"anthropic", "onnxruntime", "torch"}


def module_name(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def imported(path: Path, src: Path = SRC) -> set[str]:
    """Every module a file imports, with relative imports resolved and import_module calls on a constant included."""
    package = list(path.relative_to(src).with_suffix("").parts[:-1])
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[:len(package) - node.level + 1] if node.level else []
            module = ".".join(base + ([node.module] if node.module else []))
            names.add(module)
            names.update(f"{module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
                and isinstance(node.args[0].value, str):
            func = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if func in ("import_module", "__import__", "find_spec"):
                names.add(node.args[0].value)
    return names


def is_v2(name: str) -> bool:
    return name == "btc_survivor.v2" or name.startswith("btc_survivor.v2.")


def test_the_scan_sees_the_version_1_tree():
    names = {module_name(p) for p in V1_FILES}
    assert {"btc_survivor", "btc_survivor.config", "btc_survivor.runner", "btc_survivor.feed.base",
            "btc_survivor.cli.stats"} <= names
    assert not any(is_v2(n) for n in names)


def test_import_resolution_catches_every_form(tmp_path):
    fake = tmp_path / "btc_survivor" / "cli" / "sample.py"
    fake.parent.mkdir(parents=True)
    samples = {
        "import btc_survivor.v2.config\n": "btc_survivor.v2.config",
        "from btc_survivor.v2 import config\n": "btc_survivor.v2",
        "from .. import v2\n": "btc_survivor.v2",
        "from ..v2.stocks import engine\n": "btc_survivor.v2.stocks",
        "importlib.import_module('btc_survivor.v2.news')\n": "btc_survivor.v2.news",
        "import anthropic\n": "anthropic",
        "from onnxruntime import InferenceSession\n": "onnxruntime",
    }
    for source, expected in samples.items():
        fake.write_text(source, encoding="utf-8")
        assert expected in imported(fake, tmp_path), source


@pytest.mark.parametrize("path", V1_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_no_version_1_module_imports_version_2(path):
    names = imported(path)
    assert not [n for n in names if is_v2(n)]
    assert "btc_survivor.v2" not in path.read_text(encoding="utf-8")


def test_importing_all_of_version_1_loads_no_version_2_module_or_model_runtime():
    modules = [module_name(p) for p in V1_FILES]
    code = ("import importlib, sys\n"
            f"for name in {modules!r}:\n"
            "    importlib.import_module(name)\n"
            "bad = sorted(n for n in sys.modules if n == 'btc_survivor.v2' or n.startswith('btc_survivor.v2.')\n"
            f"             or n.split('.')[0] in {sorted(MODEL_RUNTIMES)!r})\n"
            "print(','.join(bad))\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == ""


def test_version_2_configs_exist():
    assert {"s1_prices.toml", "s2_news.toml", "news.toml"} <= {p.name for p in V2_CONFIGS}


@pytest.mark.parametrize("path", V2_CONFIGS, ids=lambda p: p.name)
def test_version_1_loader_rejects_every_version_2_config(path):
    with pytest.raises(V1ConfigError, match="unknown section"):
        load_v1_config(path)


@pytest.mark.parametrize("name", V1_RUNS)
def test_version_1_run_configs_still_load(name):
    cfg = load_v1_config(ROOT / "configs" / f"{name}.toml")
    assert cfg.data_dir.resolve() == (ROOT / "data" / name).resolve()
    assert cfg.feed.exchange == "coinbase"


@pytest.mark.parametrize("name", V1_RUNS)
def test_version_2_loaders_reject_every_version_1_config(name):
    path = ROOT / "configs" / f"{name}.toml"
    with pytest.raises(ConfigError, match='kind = "stocks"'):
        load_stock_config(path)
    with pytest.raises(ConfigError, match='kind = "news"'):
        load_news_config(path)


@pytest.mark.parametrize("path", TRADING_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_no_trading_module_imports_a_model_runtime(path):
    names = imported(path)
    assert not {n.split(".")[0] for n in names} & MODEL_RUNTIMES


def test_torch_is_imported_nowhere_in_the_package():
    hits = [str(p.relative_to(SRC)) for p in sorted(PACKAGE.rglob("*.py"))
            if "torch" in {n.split(".")[0] for n in imported(p)}]
    assert not hits
