"""Export ProsusAI/finbert to ONNX for the FinBERT scorer of PLAN_V2 section 8.7, once and with the owner's approval.

It runs only in a throwaway virtual environment outside this repository holding exactly the packages in PINS, because
torch never enters the project environment, and it refuses to run anywhere else. Without --confirm it only states the
model source, revision, files, licence, packages and destinations and touches nothing.

With --confirm it downloads the five files in SOURCE_FILES of ProsusAI/finbert at the pinned commit REVISION, about
440 MB, sending no token, into a work folder outside the repository that is deleted afterwards. It exports ONNX at
opset 17 with dynamic batch and sequence axes as one file, saves the fast tokenizer as tokenizer.json, checks that the
saved tokenizer gives the same ids as transformers, that ONNX and PyTorch logits agree within 0.0001 on 50 fictional
sentences and one full 512 token window, and that two ONNX runs are bitwise identical. It then writes model.onnx,
tokenizer.json and manifest.json, with the revision, label order, sizes and SHA 256 hashes, to the paths [finbert]
names in configs/v2/news.toml, refusing to replace any that exist, and prints the three pins to copy into that
config. It prints only verdicts, hashes, sizes and timings. Exit code 0 means done, 2 a refusal or a failed check,
and 1 a crash.

    python3.11 -m venv /tmp/finbert_export
    /tmp/finbert_export/bin/pip install torch==2.14.0 transformers==5.17.0 tokenizers==0.23.2 onnx==1.23.0 \\
        onnxruntime==1.30.0 numpy==2.4.6
    /tmp/finbert_export/bin/python -I scripts/export_finbert.py              # the statement alone
    /tmp/finbert_export/bin/python -I scripts/export_finbert.py --confirm    # the export
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import sys
import tempfile
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "v2" / "news.toml"

SOURCE = "ProsusAI/finbert"
SOURCE_URL = "https://huggingface.co/ProsusAI/finbert"
REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"     # the newest commit of the model repository, read 12 Oct 2026
SOURCE_FILES = ("config.json", "pytorch_model.bin", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt")
LICENCE = "code Apache 2.0, weights without a stated licence, training data CC BY NC SA 3.0, for personal non " \
          "commercial use"
# Each a release at least two weeks old on 12 Oct 2026, with tokenizers and onnxruntime equal to the project's pins.
PINS = {"torch": "2.14.0", "transformers": "5.17.0", "tokenizers": "0.23.2", "onnx": "1.23.0",
        "onnxruntime": "1.30.0", "numpy": "2.4.6"}
RECORDED = (*PINS, "huggingface_hub", "safetensors")
OPSET = 17
INPUTS = ("input_ids", "attention_mask", "token_type_ids")
OUTPUT = "logits"
LABELS = ("positive", "negative", "neutral")
TOLERANCE = 1e-4
FULL_WINDOW = 510                                          # tokens between [CLS] and [SEP] in the longest window
MANIFEST_FORMAT = 1
ROLES = ("model", "tokenizer", "manifest")
CONFIG_KEYS = {"model": "model_file", "tokenizer": "tokenizer_file", "manifest": "manifest_file"}

# Fictional sentences about invented companies and places, for the parity check here and the repeatability check of
# the first live session. None names a real company, product or person.
CHECK_SENTENCES = (
    "Northwind Systems raised its full year revenue outlook after record quarterly orders.",
    "Bluefield Motors recalled forty thousand vehicles over a faulty brake sensor.",
    "Corvane Labs said its quarterly results were in line with its earlier guidance.",
    "Shares of Tallis Freight fell after the company missed profit estimates by a wide margin.",
    "Marrow Bank will hold its annual shareholder meeting on a Thursday in the spring.",
    "Quillon Energy reported a surprise loss as costs at its new refinery doubled.",
    "Ostrava Chips won a multiyear supply contract from a large cloud provider.",
    "The board of Fennick Retail approved a quarterly dividend unchanged from last year.",
    "Delmont Pharma said regulators rejected its application for a new heart drug.",
    "Harrowgate Software beat expectations and lifted guidance for the coming year.",
    "Pinecrest Foods closed two plants and cut eight hundred jobs amid weak demand.",
    "Ardent Mobile named a new chief financial officer effective next month.",
    "Kestrel Aerospace delivered more jets than analysts expected in the third quarter.",
    "Lumen Grove Media warned that advertising revenue would decline sharply next quarter.",
    "Vexley Insurance completed the previously announced sale of its travel unit.",
    "Gantry Robotics doubled its order backlog and announced a large share buyback.",
    "Sorrel Telecom lost subscribers for the fourth straight quarter.",
    "The central bank of Eldoria held its policy rate steady, as markets had expected.",
    "Inflation in Eldoria rose faster than forecast, raising fears of further rate hikes.",
    "Eldoria added far more jobs than economists predicted, and wage growth stayed moderate.",
    "Unemployment in Eldoria climbed to its highest level in three years.",
    "Brightwater Utilities filed its annual report with the regulator on schedule.",
    "Cobalt Ridge Mining suspended output at its largest mine after a flood.",
    "Talbot Devices unveiled a thinner tablet that will ship in eight colours.",
    "Analysts at a fictional brokerage upgraded Meridian Cloud to buy from hold.",
    "Analysts cut their price target on Ashgrove Apparel after a weak holiday season.",
    "Redfern Logistics agreed to buy a smaller rival for two billion in cash.",
    "A court ordered Halcyon Networks to pay a large fine for breaching privacy rules.",
    "Wexford Semiconductor said gross margin expanded for the sixth quarter in a row.",
    "Orchard Health said its chief executive will retire at the end of the year.",
    "Tamsin Airlines cancelled hundreds of flights after a computer outage.",
    "Granite Peak Bank raised its capital ratio well above the regulatory minimum.",
    "Silverline Games delayed its flagship title by a year, disappointing fans.",
    "Juniper Rail opened a new maintenance depot near the eastern harbour.",
    "Crescent Chemicals posted flat sales as lower prices offset higher volumes.",
    "Nimbus Storage shares jumped after it reported surging demand for its servers.",
    "Elmstead Motors halted production at one plant because of a parts shortage.",
    "Rowan Biotech said a late stage trial of its cancer drug met its main goal.",
    "Harbor Point Realty reported occupancy unchanged at ninety two percent.",
    "Calder Payments said a data breach exposed records of millions of customers.",
    "Osprey Wireless expects its new network to be complete by the middle of next year.",
    "Thornbury Steel swung to a profit on strong demand from builders.",
    "Ivy Lane Cosmetics withdrew its forecast, citing uncertain consumer spending.",
    "The statistics office of Eldoria will publish producer prices on Tuesday morning.",
    "Kingsmere Software settled a patent dispute without admitting wrongdoing.",
    "Ferro Valley Energy raised its dividend by twenty percent after a record year.",
    "Lockhart Chips lost a major customer, and its shares tumbled in early trading.",
    "Mapleton Grocers opened twelve new stores during the quarter.",
    "Starling Media said streaming subscribers grew faster than in any prior quarter.",
    "Westbrook Capital said it had no comment on reports of a possible merger.",
)


class ExportError(RuntimeError):
    """A refusal or a failed check, which exits with code 2 and writes nothing to the destinations."""


def destinations(config: Path = CONFIG) -> dict[str, Path]:
    """The model, tokenizer and manifest paths that [finbert] names, with ~ expanded and relative paths resolved
    from the config file's folder, as the project's config loader does."""
    with config.open("rb") as fh:
        finbert = tomllib.load(fh)["finbert"]
    out = {}
    for role, key in CONFIG_KEYS.items():
        path = Path(finbert[key]).expanduser()
        out[role] = (path if path.is_absolute() else config.resolve().parent / path).resolve()
    return out


def shown(path: Path) -> str:
    """A path relative to the repository when it lies inside it."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def statement(paths: Mapping[str, Path]) -> list[str]:
    """What the export downloads, from where, under which licence, with which packages, and where it writes."""
    return [
        "finbert export of PLAN_V2 section 8.7",
        f"source: {SOURCE_URL}",
        f"revision: {REVISION}",
        f"files: {', '.join(SOURCE_FILES)}, about 440 MB, downloaded without a token",
        f"licence: {LICENCE}",
        f"packages: {' '.join(f'{name}=={version}' for name, version in PINS.items())}",
        *(f"writes {role}: {shown(paths[role])}" for role in ROLES),
    ]


def installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def environment_problems(prefix: Path, project_importable: bool,
                         version: Callable[[str], str | None] = installed_version) -> list[str]:
    """Why this Python may not run the export, none when it is a throwaway environment outside the repository with
    exactly the pinned packages."""
    problems = []
    if Path(prefix).resolve().is_relative_to(ROOT):
        problems.append(f"this Python lives inside the repository at {shown(Path(prefix).resolve())}, and torch "
                        f"never enters the project environment, so use a throwaway environment outside it")
    if project_importable:
        problems.append("the btc_survivor package is importable here, so this is a project environment")
    for name, pinned in PINS.items():
        have = version(name)
        if (have or "").split("+")[0] != pinned:  # a local build tag such as torch's +cpu is the same release
            problems.append(f"{name} is {have or 'not installed'}, the export needs {name}=={pinned}")
    return problems


def file_entry(path: Path) -> dict[str, Any]:
    """The name, SHA 256 and size of one written file."""
    data = path.read_bytes()
    return {"file": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def build_manifest(model: Path, tokenizer: Path, labels: Sequence[str], max_positions: int,
                   source_files: Mapping[str, Mapping[str, Any]], parity: Mapping[str, Any],
                   reference_logits: Sequence[Sequence[float]], versions: Mapping[str, str | None]) -> dict[str, Any]:
    """The manifest the project's FinBERT runtime checks, with the revision, label order, sizes and hashes, and the
    record of how the files were made."""
    return {
        "format": MANIFEST_FORMAT,
        "source": SOURCE,
        "source_url": SOURCE_URL,
        "revision": REVISION,
        "licence": LICENCE,
        "labels": list(labels),
        "max_position_embeddings": max_positions,
        "opset": OPSET,
        "inputs": list(INPUTS),
        "output": OUTPUT,
        "model": file_entry(model),
        "tokenizer": file_entry(tokenizer),
        "source_files": dict(source_files),
        "parity": dict(parity),
        "check_sentences": list(CHECK_SENTENCES),
        "reference_logits": [list(map(float, row)) for row in reference_logits],
        "versions": dict(versions),
        "exported_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def pins_text(manifest: Mapping[str, Any]) -> list[str]:
    """The three [finbert] lines to copy into configs/v2/news.toml."""
    return [f'model_sha256 = "{manifest["model"]["sha256"]}"',
            f'tokenizer_sha256 = "{manifest["tokenizer"]["sha256"]}"',
            f'source_revision = "{manifest["revision"]}"']


def stage(source: Path, target: Path) -> Path:
    """Copy a checked file to a partial file beside its target, in a folder for the owner alone, confirm the copy's
    hash and return the partial file."""
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    shutil.copyfile(source, partial)
    os.chmod(partial, 0o600)
    if file_entry(partial)["sha256"] != file_entry(source)["sha256"]:
        partial.unlink()
        raise ExportError(f"the copy of {source.name} does not match the checked file")
    return partial


def place(sources: Mapping[str, Path], paths: Mapping[str, Path]) -> None:
    """Stage every file first and rename them only once all three copies are checked, the manifest last."""
    staged = [(stage(sources[role], paths[role]), paths[role]) for role in ROLES]
    for partial, target in staged:
        os.replace(partial, target)


def export(paths: Mapping[str, Path], work: Path) -> dict[str, Any]:
    """Download, export, check and write the files, returning the manifest. The heavy packages are imported here
    only, after the environment check."""
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from huggingface_hub import snapshot_download
    from tokenizers import Tokenizer
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    timings = {}
    started = time.monotonic()
    source = Path(snapshot_download(SOURCE, revision=REVISION, allow_patterns=list(SOURCE_FILES),
                                    local_dir=work / "source", cache_dir=work / "cache", token=False))
    missing = [name for name in SOURCE_FILES if not (source / name).is_file()]
    if missing:
        raise ExportError(f"the download lacks {', '.join(missing)}")
    source_files = {name: {k: v for k, v in file_entry(source / name).items() if k != "file"} for name in SOURCE_FILES}
    timings["download"] = time.monotonic() - started

    started = time.monotonic()
    reference_tokenizer = AutoTokenizer.from_pretrained(str(source), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(source), attn_implementation="eager").eval()
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    if sorted(labels) != sorted(LABELS):
        raise ExportError(f"the model's labels {labels} are not an order of {', '.join(LABELS)}")
    max_positions = int(model.config.max_position_embeddings)
    out = work / "out"
    out.mkdir()
    tokenizer_path, model_path = out / "tokenizer.json", out / "model.onnx"
    reference_tokenizer.backend_tokenizer.save(str(tokenizer_path))
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    cls_id, sep_id = tokenizer.token_to_id("[CLS]"), tokenizer.token_to_id("[SEP]")

    def window(ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row = np.array([[cls_id, *ids, sep_id]], dtype=np.int64)
        return row, np.ones_like(row), np.zeros_like(row)

    def content(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False).ids

    for text in CHECK_SENTENCES:
        if window(content(text))[0][0].tolist() != reference_tokenizer(text)["input_ids"]:
            raise ExportError("the saved tokenizer.json gives other ids than the transformers tokenizer")

    class Logits(torch.nn.Module):
        """The classifier with its inputs in a fixed order and its logits as the only output."""

        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask,
                              token_type_ids=token_type_ids).logits

    wrapped = Logits(model).eval()
    axes = {**{name: {0: "batch", 1: "sequence"} for name in INPUTS}, OUTPUT: {0: "batch"}}
    with torch.no_grad():
        torch.onnx.export(wrapped, tuple(torch.from_numpy(a) for a in window(content(CHECK_SENTENCES[0]))),
                          str(model_path), input_names=list(INPUTS), output_names=[OUTPUT], dynamic_axes=axes,
                          opset_version=OPSET, do_constant_folding=True, dynamo=False, external_data=False)
    extra = sorted(p.name for p in out.iterdir() if p not in (model_path, tokenizer_path))
    if extra:
        raise ExportError(f"the export wrote more than one model file, {', '.join(extra)}")
    onnx.checker.check_model(str(model_path))
    opsets = {entry.domain: entry.version for entry in onnx.load(str(model_path)).opset_import}
    if opsets.get("", opsets.get("ai.onnx")) != OPSET:
        raise ExportError(f"the model uses opsets {opsets}, expected {OPSET}")
    timings["export"] = time.monotonic() - started

    started = time.monotonic()
    texts = [*CHECK_SENTENCES, None]
    full = [token for text in CHECK_SENTENCES for token in content(text)] * 4
    if len(full) < FULL_WINDOW:
        raise ExportError("the check sentences are too short to fill a full window")
    windows = [window(content(text)) if text is not None else window(full[:FULL_WINDOW]) for text in texts]

    def onnx_logits() -> list[np.ndarray]:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        return [session.run(None, dict(zip(INPUTS, arrays)))[0][0] for arrays in windows]

    with torch.no_grad():
        reference = [wrapped(*(torch.from_numpy(a) for a in arrays)).numpy()[0] for arrays in windows]
    first, second = onnx_logits(), onnx_logits()
    worst = max(float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) for a, b in zip(reference, first))
    repeatable = all(a.tobytes() == b.tobytes() for a, b in zip(first, second))
    if worst > TOLERANCE or not repeatable:
        raise ExportError(f"the parity check failed, largest logit difference {worst:.3g} against {TOLERANCE}, "
                          f"bitwise repeatable {repeatable}")
    timings["check"] = time.monotonic() - started

    parity = {"windows": len(windows), "max_abs_diff": worst, "tolerance": TOLERANCE, "bitwise_repeatable": repeatable,
              "full_window_tokens": FULL_WINDOW + 2}
    versions = {"python": platform.python_version(), "platform": platform.platform(),
                **{name: installed_version(name) for name in RECORDED}}
    staged = work / "manifest.json"
    manifest = build_manifest(model_path, tokenizer_path, labels, max_positions, source_files, parity,
                              [row.tolist() for row in reference[:len(CHECK_SENTENCES)]], versions)
    staged.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    place({"model": model_path, "tokenizer": tokenizer_path, "manifest": staged}, paths)
    manifest["timings"] = timings
    return manifest


def preflight(paths: Mapping[str, Path], work: Path | None) -> list[str]:
    """Every reason not to start, checked before anything is created or downloaded, so a refusal leaves no trace."""
    problems = environment_problems(Path(sys.prefix), importlib.util.find_spec("btc_survivor") is not None)
    problems += [f"{shown(paths[role])} exists, and the export never replaces a file" for role in ROLES
                 if paths[role].exists()]
    if work is not None:
        if work.resolve().is_relative_to(ROOT):
            problems.append("the work folder must lie outside the repository, which never holds the download")
        elif work.exists() and any(work.iterdir()):
            problems.append("the work folder must be empty")
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export ProsusAI/finbert to ONNX once, in a throwaway environment "
                                                 "outside the repository. Without --confirm it only states what it "
                                                 "would do.")
    parser.add_argument("--confirm", action="store_true", help="download, export, check and write the files")
    parser.add_argument("--config", type=Path, default=CONFIG, help="the news config whose [finbert] paths receive "
                                                                    "the files")
    parser.add_argument("--work", type=Path, help="an empty work folder outside the repository, by default a new "
                                                  "temporary folder, deleted afterwards")
    return parser


def main(argv: list[str] | None = None) -> int:
    """State the export, and with --confirm run it, returning the exit code."""
    args = build_parser().parse_args(argv)
    paths = destinations(args.config)
    print("\n".join(statement(paths)))
    if not args.confirm:
        print("nothing done, pass --confirm in the throwaway environment to export")
        return 0
    try:
        problems = preflight(paths, args.work)
        if problems:
            raise ExportError("refused\n  " + "\n  ".join(problems))
        work = args.work.resolve() if args.work else Path(tempfile.mkdtemp(prefix="finbert_export_"))
        work.mkdir(parents=True, exist_ok=True)
        try:
            manifest = export(paths, work)
        finally:
            shutil.rmtree(work, ignore_errors=True)
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("verdict: pass")
    for role in ("model", "tokenizer"):
        print(f"{role}: {manifest[role]['bytes']} bytes, sha256 {manifest[role]['sha256']}")
    print(f"parity: largest logit difference {manifest['parity']['max_abs_diff']:.3g} over "
          f"{manifest['parity']['windows']} windows, bitwise repeatable")
    print("timings: " + ", ".join(f"{name} {seconds:.1f} s" for name, seconds in manifest["timings"].items()))
    print("copy these pins into [finbert] of configs/v2/news.toml")
    print("\n".join(pins_text(manifest)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
