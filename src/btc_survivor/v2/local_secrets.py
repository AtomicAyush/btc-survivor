"""The only reader of secrets.env, a file of plain KEY=value lines that only its owner may read.

Errors and the repr name keys, never values, and every value is registered with redact as soon as it is read.
This module imports nothing that can reach the network.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path

from btc_survivor.v2 import redact

FINNHUB = "FINNHUB_API_KEY"
ANTHROPIC = "ANTHROPIC_API_KEY"
CONTACT = "CONTACT_EMAIL"
NAMES = (FINNHUB, ANTHROPIC, CONTACT)
DEFAULT_PATH = Path("~/.config/btc_survivor/secrets.env")

# Every other ANTHROPIC_ variable is dropped, such as ANTHROPIC_AUTH_TOKEN, ANTHROPIC_PROFILE and ANTHROPIC_BASE_URL,
# so the SDK cannot use a credential, profile or endpoint other than the key above.
SCRUBBED_PREFIX = "ANTHROPIC_"
SDK_LOG = "ANTHROPIC_LOG"
# Variables that would send the key through a proxy or trust a certificate authority the owner did not choose.
REFUSED = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "SSL_CERT_FILE", "SSL_CERT_DIR")

_NAME = re.compile(r"[A-Z][A-Z0-9_]*")


class SecretsError(ValueError):
    """Raised for an unsafe, malformed or incomplete secrets file. The message never holds a value."""


@dataclass(frozen=True)
class Secrets:
    path: Path
    values: dict[str, str] = field(compare=False)

    def __repr__(self) -> str:
        return f"Secrets(present={list(self.present)})"

    @property
    def present(self) -> tuple[str, ...]:
        """The known names that have a value, in the fixed order."""
        return tuple(name for name in NAMES if name in self.values)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(name for name in NAMES if name not in self.values)

    def get(self, name: str) -> str:
        """The value for one known name, or an error naming the key and the file."""
        if name not in NAMES:
            raise SecretsError(f"{name} is not one of {', '.join(NAMES)}")
        if name not in self.values:
            raise SecretsError(f"{name} is missing from {self.path}")
        return self.values[name]

    def require(self, *names: str) -> None:
        """Raise one error naming every requested key that has no value."""
        absent = [name for name in names if name not in self.values]
        if absent:
            raise SecretsError(f"missing from {self.path}: {', '.join(absent)}")


def parse(lines: Iterable[str]) -> dict[str, str]:
    """Read KEY=value lines, skipping blanks and comments, and accept only the three names, each once and non empty.

    It takes lines, such as an open file, and never the whole text, so no argument shown in a traceback holds it.
    """
    values: dict[str, str] = {}
    for number, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, sep, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not sep or not _NAME.fullmatch(name):
            raise SecretsError(f"line {number} is not a KEY=value line")
        if name not in NAMES:
            raise SecretsError(f"line {number} has an unknown name, only {', '.join(NAMES)} are allowed")
        if name in values:
            raise SecretsError(f"{name} appears more than once")
        if not value:
            raise SecretsError(f"{name} is empty")
        values[name] = value
    return values


def load_secrets(path: Path = DEFAULT_PATH) -> Secrets:
    """Read the file only when the current user owns it and no one else has access, then register its values."""
    path = Path(path).expanduser()
    try:
        fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:
        raise SecretsError(f"secrets file {path} does not exist") from None
    with os.fdopen(fd, "r", encoding="utf-8") as fh:
        info = os.fstat(fh.fileno())
        if info.st_uid != os.getuid():
            raise SecretsError(f"secrets file {path} is not owned by the current user")
        if info.st_mode & 0o077:
            raise SecretsError(f"secrets file {path} has mode {stat.S_IMODE(info.st_mode):o}, "
                               f"group and others must have no access, so run chmod 600 on it")
        values = parse(fh)
    for value in values.values():
        redact.register(value)
    return Secrets(path, values)


def check(path: Path = DEFAULT_PATH) -> tuple[bool, list[str]]:
    """The check-secrets verdict and its lines, naming the keys present and missing but never a value."""
    try:
        secrets = load_secrets(path)
    except SecretsError as exc:
        return False, [f"fail: {exc}"]
    lines = [f"present: {name}" for name in secrets.present] + [f"missing: {name}" for name in secrets.missing]
    ok = not secrets.missing
    return ok, [f"{'pass' if ok else 'fail'}: {len(secrets.present)} of {len(NAMES)} keys in a file only its owner "
                f"can read", *lines]


def prepare_anthropic_env(secrets: Secrets, environ: MutableMapping[str, str] = os.environ) -> None:
    """For the news process only, refuse while the SDK log, a proxy or a certificate variable is set, drop every
    other ANTHROPIC_ variable, and put the key where the SDK reads it."""
    if SDK_LOG in environ:
        raise SecretsError(f"{SDK_LOG} is set, unset it so the SDK cannot log requests")
    refused = sorted(name for name in environ if name.upper() in REFUSED)
    if refused:
        raise SecretsError(f"{', '.join(refused)} set, unset them so the key goes straight to the Anthropic API")
    key = secrets.get(ANTHROPIC)
    for name in [name for name in environ if name.startswith(SCRUBBED_PREFIX) and name != ANTHROPIC]:
        del environ[name]
    environ[ANTHROPIC] = key
