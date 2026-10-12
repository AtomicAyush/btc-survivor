"""FinBERT on onnxruntime, the local scorer of PLAN_V2 section 8.7 and the only module that imports onnxruntime.

The model, its fast tokenizer and the manifest come from scripts/export_finbert.py. Before anything runs, the files
are read once and their SHA 256 hashes checked against the pins in [finbert] and against the manifest, and the
session and tokenizer are then built from those same bytes, so what was checked is what runs. A missing file, an
unpinned hash, a mismatch or a manifest that disagrees with the config raises FinbertError, which stops the news
process. The session runs on the CPU with one intra op thread, one inter op thread and sequential execution, so
scores are bitwise repeatable on one machine.

A text is the headline and the item text joined by a period. Its tokens are cut into windows of at most window_tokens,
each scored alone between [CLS] and [SEP] with no padding, the first max_windows kept and the rest dropped, and the
softmax probabilities of the windows are averaged with each window weighted by its token count.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from btc_survivor.v2.config import NewsConfig

LABELS = ("positive", "negative", "neutral")   # the order Probs keeps, whatever order the model's outputs use
INPUTS = ("input_ids", "attention_mask", "token_type_ids")
CLS, SEP = "[CLS]", "[SEP]"
SPECIAL_TOKENS = 2                               # [CLS] and [SEP] around every window
PROVIDERS = ["CPUExecutionProvider"]
EXPORT_SCRIPT = "scripts/export_finbert.py"


class FinbertError(ValueError):
    """Raised when the model files are missing, unpinned or do not match their hashes or manifest, which stops the
    news process."""


class Session(Protocol):
    """What the scorer needs from an onnxruntime InferenceSession, so tests can pass a fake one."""

    def get_inputs(self) -> Sequence[Any]: ...

    def run(self, output_names: Sequence[str] | None, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]: ...


@dataclass(frozen=True)
class Probs:
    """One text's averaged probabilities, with the windows scored and the tokens inside them."""

    pos: float
    neg: float
    neu: float
    windows: int
    tokens: int

    @property
    def strength(self) -> float:
        """p_pos minus p_neg, the strength every impact of a FinBERT score carries."""
        return self.pos - self.neg


@dataclass(frozen=True)
class Verified:
    """The model and tokenizer bytes whose hashes matched, with the label order and position limit the manifest
    states."""

    model: bytes
    tokenizer: bytes
    labels: tuple[str, ...]
    max_positions: int


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def join_text(headline: str, text: str | None) -> str:
    """The headline and the text joined by a period, or the headline alone when there is no text."""
    return ". ".join(part for part in ((headline or "").strip(), (text or "").strip()) if part)


def windows(ids: Sequence[int], size: int, limit: int) -> list[tuple[int, ...]]:
    """Consecutive windows of at most size token ids, the first limit of them, with the rest dropped. A text without
    tokens gives one empty window, so every text gets a score."""
    chunks = [tuple(ids[start:start + size]) for start in range(0, len(ids), size)][:limit]
    return chunks or [()]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Probabilities from one row of logits, in float64."""
    z = np.asarray(logits, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def weighted_mean(rows: np.ndarray, weights: Sequence[int]) -> np.ndarray:
    """The rows averaged with the given weights, each at least one, so an empty window still counts."""
    w = np.maximum(np.asarray(weights, dtype=np.float64), 1.0)
    return (w @ np.asarray(rows, dtype=np.float64)) / w.sum()


def session_options(threads: int) -> ort.SessionOptions:
    """CPU options for bitwise repeatable scores, threads intra op threads, one inter op thread and sequential
    execution."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def open_session(model: bytes, threads: int) -> Session:
    """An onnxruntime CPU session built from the verified model bytes."""
    return ort.InferenceSession(model, sess_options=session_options(threads), providers=PROVIDERS)


def _read(path: Path, what: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise FinbertError(f"cannot read the FinBERT {what} {path} ({exc.strerror}), run {EXPORT_SCRIPT} first, "
                           f"with the owner's approval") from None


def _manifest(data: bytes, path: Path) -> dict[str, Any]:
    """The manifest as a dict, refusing anything that is not one JSON object."""
    try:
        manifest = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FinbertError(f"the FinBERT manifest {path} is not valid JSON, {exc}") from None
    if not isinstance(manifest, dict):
        raise FinbertError(f"the FinBERT manifest {path} is not a JSON object")
    return manifest


def _entry(manifest: dict[str, Any], role: str) -> dict[str, Any]:
    entry = manifest.get(role)
    return entry if isinstance(entry, dict) else {}


def verify(config: NewsConfig) -> Verified:
    """Read the model, tokenizer and manifest named by [finbert] and check every pin, naming each problem, so a hash
    that is unpinned or differs between the config, the manifest and the file raises FinbertError."""
    settings = config.finbert
    problems = [f"[finbert] {key} is not pinned" for key in ("model_sha256", "tokenizer_sha256", "source_revision")
                if not getattr(settings, key)]
    manifest_path = config.path(settings.manifest_file)
    manifest = _manifest(_read(manifest_path, "manifest"), manifest_path)
    files = {"model": (config.path(settings.model_file), settings.model_sha256),
             "tokenizer": (config.path(settings.tokenizer_file), settings.tokenizer_sha256)}
    data = {role: _read(path, role) for role, (path, _) in files.items()}
    for role, (path, pinned) in files.items():
        actual = sha256(data[role])
        if pinned and actual != pinned:
            problems.append(f"the {role} file {path.name} has sha256 {actual}, but [finbert] pins {pinned}")
        if _entry(manifest, role).get("sha256") != actual:
            problems.append(f"the {role} file {path.name} has sha256 {actual}, but the manifest says "
                            f"{_entry(manifest, role).get('sha256')}")
    if settings.source_revision and manifest.get("revision") != settings.source_revision:
        problems.append(f"the manifest revision {manifest.get('revision')} is not the pinned source_revision "
                        f"{settings.source_revision}")
    labels = manifest.get("labels")
    if not (isinstance(labels, list) and all(isinstance(label, str) for label in labels)
            and sorted(labels) == sorted(LABELS)):
        problems.append(f"the manifest labels {labels} are not an order of {', '.join(LABELS)}")
        labels = list(LABELS)
    positions = manifest.get("max_position_embeddings")
    if not isinstance(positions, int) or settings.window_tokens + SPECIAL_TOKENS > positions:
        problems.append(f"[finbert] window_tokens {settings.window_tokens} plus {SPECIAL_TOKENS} special tokens does "
                        f"not fit the manifest max_position_embeddings {positions}")
        positions = 0
    if problems:
        raise FinbertError("FinBERT does not match its pins, so the news process stops\n  " + "\n  ".join(problems))
    return Verified(data["model"], data["tokenizer"], tuple(labels), positions)


class Finbert:
    """One session and tokenizer that score texts. Only one thread may call score at a time."""

    def __init__(self, session: Session, tokenizer: Tokenizer, labels: Sequence[str], window_tokens: int,
                 max_windows: int) -> None:
        tokenizer.no_truncation()
        tokenizer.no_padding()
        cls_id, sep_id = tokenizer.token_to_id(CLS), tokenizer.token_to_id(SEP)
        if cls_id is None or sep_id is None:
            raise FinbertError(f"the FinBERT tokenizer has no {CLS} or {SEP} token")
        names = [node.name for node in session.get_inputs()]
        unknown = [name for name in names if name not in INPUTS]
        if unknown or "input_ids" not in names:
            raise FinbertError(f"the FinBERT model takes inputs {names}, expected input_ids and only {INPUTS}")
        self.session = session
        self.tokenizer = tokenizer
        self.inputs = tuple(names)
        self.cls_id, self.sep_id = cls_id, sep_id
        self.window_tokens = window_tokens
        self.max_windows = max_windows
        self.order = [list(labels).index(label) for label in LABELS]

    @classmethod
    def load(cls, config: NewsConfig, session_factory=open_session) -> Finbert:
        """Verify the files named by [finbert] and build the scorer from their bytes. Tests pass a fake session
        factory."""
        verified = verify(config)
        settings = config.finbert
        try:
            tokenizer = Tokenizer.from_str(verified.tokenizer.decode("utf-8"))
        except Exception as exc:  # the tokenizers library raises its own exception types
            raise FinbertError(f"the FinBERT tokenizer cannot be read ({type(exc).__name__})") from None
        session = session_factory(verified.model, settings.threads)
        return cls(session, tokenizer, verified.labels, settings.window_tokens, settings.max_windows)

    def logits(self, chunk: Sequence[int]) -> np.ndarray:
        """The model's logits for one window, in the model's own label order."""
        ids = np.array([[self.cls_id, *chunk, self.sep_id]], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": np.ones_like(ids), "token_type_ids": np.zeros_like(ids)}
        out = np.asarray(self.session.run(None, {name: feed[name] for name in self.inputs})[0])
        if out.shape != (1, len(LABELS)):
            raise FinbertError(f"the FinBERT model returned logits of shape {out.shape}, expected (1, {len(LABELS)})")
        return out[0]

    def score(self, headline: str, text: str | None = None) -> Probs:
        """The probabilities of one item's headline and text, averaged over its windows by token count."""
        ids = self.tokenizer.encode(join_text(headline, text), add_special_tokens=False).ids
        chunks = windows(ids, self.window_tokens, self.max_windows)
        mean = weighted_mean([softmax(self.logits(chunk)) for chunk in chunks], [len(chunk) for chunk in chunks])
        pos, neg, neu = (float(mean[i]) for i in self.order)
        return Probs(pos, neg, neu, len(chunks), sum(len(chunk) for chunk in chunks))
