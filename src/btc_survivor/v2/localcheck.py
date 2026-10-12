"""The local only check of PLAN_V2 section 13.2. Both dashboard ports may listen only on 127.0.0.1, and no
tunnel or remote forward may run, so no page can reach another machine. Sockets come from lsof and from netstat,
which also sees other users' sockets, tunnels from ps and, when Tailscale is installed, from its serve settings.
Results hold counts, pids and program names only, never a full command line, because a tunnel's arguments can
carry its token."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePath

WATCHED_PORTS = (8050, 8060)
LOOPBACK = "127.0.0.1"
LSOF = ("lsof", "-nP", "-iTCP:" + ",".join(map(str, WATCHED_PORTS)), "-sTCP:LISTEN", "-Fpcn")
# lsof run by a normal user lists only that user's sockets, so netstat on macOS, or ss on Linux, lists every user's.
SOCKETS = ("netstat", "-an", "-p", "tcp") if sys.platform == "darwin" else ("ss", "-ltnH")
PS = ("ps", "axww", "-o", "pid=,command=")
TAILSCALE_APP = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
TAILSCALE_STATUS = (("serve", "status", "--json"), ("funnel", "status", "--json"))
TIMEOUT_SECONDS = 30.0
TUNNEL_PROGRAMS = ("cloudflared", "ngrok", "frpc", "zrok", "bore", "lt", "localtunnel", "devtunnel", "code-tunnel")
CODE_EDITORS = ("code", "code-insiders")
TAILSCALE_TUNNELS = ("funnel", "serve")
SSH_FLAGS_WITH_VALUE = frozenset("bceilmopBDEFIJLOPQRSwW")
LOCAL_BINDS = ("localhost", "127.0.0.1", "::1")
DASHBOARD_PORT = re.compile(r":(?:8050|8060)(?!\d)")


class LocalCheckError(RuntimeError):
    """Raised when the socket or process listing cannot be read, so the check fails closed."""


@dataclass(frozen=True)
class Listener:
    """One listening TCP socket on a watched port, from lsof, with the pid and program that hold it."""

    pid: int
    program: str
    address: str
    port: int


@dataclass(frozen=True)
class Socket:
    """One listening TCP socket on a watched port, from netstat or ss, which see every user's sockets."""

    address: str
    port: int


@dataclass(frozen=True)
class Tunnel:
    """One running tunnel or remote forward, named by its kind and never by its arguments."""

    pid: int
    kind: str


@dataclass(frozen=True)
class LocalCheck:
    """The verdict of one check, from every listener on a watched port and every tunnel process."""

    listeners: tuple[Listener, ...]
    tunnels: tuple[Tunnel, ...]
    sockets: tuple[Socket, ...] = ()

    @property
    def exposed(self) -> tuple[Listener, ...]:
        return tuple(s for s in self.listeners if s.address != LOOPBACK)

    @property
    def exposed_sockets(self) -> tuple[Socket, ...]:
        return tuple(s for s in self.sockets if s.address != LOOPBACK)

    @property
    def offending_sockets(self) -> int:
        """The larger of the two listings' counts, since netstat also sees sockets of other users."""
        return max(len(self.exposed), len(self.exposed_sockets))

    @property
    def loopback_listeners(self) -> int:
        return max(len(self.listeners) - len(self.exposed), len(self.sockets) - len(self.exposed_sockets))

    @property
    def passed(self) -> bool:
        return not self.exposed and not self.exposed_sockets and not self.tunnels


def parse_lsof(text: str) -> list[Listener]:
    """Read lsof field output, lines tagged p, c, f and n, into the listeners on the watched ports."""
    found: list[Listener] = []
    pid, program = 0, ""
    for line in text.splitlines():
        tag, value = line[:1], line[1:]
        if tag == "p":
            pid, program = int(value), ""
        elif tag == "c":
            program = value
        elif tag == "n":
            address, _, port = value.rpartition(":")
            if port.isdigit() and int(port) in WATCHED_PORTS:
                found.append(Listener(pid, program, address.strip("[]"), int(port)))
    return found


def parse_sockets(text: str) -> list[Socket]:
    """Read netstat -an -p tcp from macOS, where 127.0.0.1.8060 ends in a dot and the port, or ss -ltnH from
    Linux, where 127.0.0.1:8060 ends in a colon and the port, into the listening sockets on the watched ports."""
    found: list[Socket] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 5 or "LISTEN" not in fields:
            continue
        local = fields[3]
        cut = max(local.rfind("."), local.rfind(":"))
        address, port = local[:cut].strip("[]"), local[cut + 1:]
        if port.isdigit() and int(port) in WATCHED_PORTS:
            found.append(Socket(address, int(port)))
    return found


def _ssh_options(args: list[str]) -> list[tuple[str, str]]:
    """Each ssh flag with its value, from clusters such as -NfR 9000:host:8050, -oName=value or -L spec."""
    options: list[tuple[str, str]] = []
    tokens = iter(args)
    for token in tokens:
        if not token.startswith("-") or token.startswith("--"):
            continue
        for index, flag in enumerate(token[1:], 2):
            if flag in SSH_FLAGS_WITH_VALUE:
                options.append((flag, token[index:] or next(tokens, "")))
                break
            options.append((flag, ""))
    return options


def _open_bind(spec: str) -> bool:
    """True when a -L spec binds an address other than loopback, as 0.0.0.0:9000:host:8060, *:9000 or [::]:9000."""
    if spec.startswith("["):
        return spec[1:spec.find("]")] not in LOCAL_BINDS
    parts = spec.split(":")
    return len(parts) == 4 and parts[0] not in LOCAL_BINDS


def _ssh_kind(args: list[str]) -> str | None:
    """A remote forward, or a forward other machines can reach through -g, GatewayPorts or an open -L bind."""
    for flag, value in _ssh_options(args):
        option = value.lower()
        if flag == "R" or (flag == "o" and option.startswith("remoteforward")):
            return "ssh -R"
        if flag == "g" or (flag == "o" and option.startswith("gatewayports") and not option.endswith("no")) \
                or (flag == "L" and _open_bind(value)):
            return "ssh -g"
    return None


def _named_tunnel(name: str) -> str | None:
    """The tunnel program a file name is, allowing a suffix such as cloudflared-darwin-arm64."""
    return next((p for p in TUNNEL_PROGRAMS if name == p or name.startswith((p + "-", p + "_"))), None)


def tunnel_kind(argv: list[str]) -> str | None:
    """Name the tunnel this command line runs, or None when it runs none. Besides the program itself, any later
    argument that is a path counts, so a tunnel run by an interpreter, as node /usr/local/bin/lt, is found too."""
    if not argv:
        return None
    program, args = PurePath(argv[0]).name.lower(), argv[1:]
    if program == "tailscale":
        mode = next((a for a in args if a in TAILSCALE_TUNNELS), None)
        return f"tailscale {mode}" if mode else None
    if program == "ssh":
        return _ssh_kind(args)
    if program == "socat" and any(DASHBOARD_PORT.search(a) for a in args):
        return "socat"
    for index, token in enumerate(argv):
        if index == 0 or "/" in token:
            name = PurePath(token).name.lower()
            kind = _named_tunnel(name)
            if kind:
                return kind
            if name in CODE_EDITORS and argv[index + 1:index + 2] == ["tunnel"]:
                return "code tunnel"
    return None


def parse_ps(text: str) -> list[Tunnel]:
    """Read ps output, a pid and a command line per row, into the tunnel processes it shows."""
    found: list[Tunnel] = []
    for line in text.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            kind = tunnel_kind(parts[1].split())
            if kind:
                found.append(Tunnel(int(parts[0]), kind))
    return found


def serve_configured(text: str) -> bool:
    """Whether tailscale serve or funnel status --json shows any setting, such as a port served with --bg, which
    keeps serving after the command exits and so never shows in ps."""
    if not text.strip():
        return False
    try:
        config = json.loads(text)
    except ValueError:
        raise LocalCheckError("tailscale status was not JSON") from None
    return any(config.values()) if isinstance(config, dict) else bool(config)


def check(lsof_text: str, ps_text: str, sockets_text: str = "", serve_texts: tuple[str, ...] = ()) -> LocalCheck:
    """Judge the lsof, netstat, ps and tailscale status listings, live or from fixed text in tests."""
    tunnels = parse_ps(ps_text)
    if any(serve_configured(text) for text in serve_texts):
        tunnels.append(Tunnel(0, "tailscale serve config"))
    return LocalCheck(tuple(parse_lsof(lsof_text)), tuple(tunnels), tuple(parse_sockets(sockets_text)))


def tailscale_cli() -> str | None:
    """The tailscale program, on PATH or inside the Mac app, or None when Tailscale is not installed."""
    return shutil.which("tailscale") or (TAILSCALE_APP if Path(TAILSCALE_APP).is_file() else None)


def _listing(command: tuple[str, ...], run: Callable[..., subprocess.CompletedProcess], none_found_ok: bool) -> str:
    """Run one listing command and return its output, accepting lsof's exit code 1 only when it printed nothing."""
    try:
        done = run(list(command), capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError) as exc:
        raise LocalCheckError(f"{PurePath(command[0]).name} could not run ({type(exc).__name__})") from None
    if done.returncode == 0 or (none_found_ok and done.returncode == 1 and not done.stdout and not done.stderr):
        return done.stdout
    raise LocalCheckError(f"{PurePath(command[0]).name} failed with exit code {done.returncode}")


def read_listings(run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                  find_tailscale: Callable[[], str | None] = tailscale_cli) -> tuple[str, str, str, tuple[str, ...]]:
    """The listening sockets with lsof and with netstat or ss, every process with ps, and, when Tailscale is
    installed, its serve and funnel settings."""
    lsof, ps, sockets = (_listing(LSOF, run, none_found_ok=True), _listing(PS, run, none_found_ok=False),
                         _listing(SOCKETS, run, none_found_ok=False))
    cli = find_tailscale()
    serve = tuple(_listing((cli, *args), run, none_found_ok=False) for args in TAILSCALE_STATUS) if cli else ()
    return lsof, ps, sockets, serve


def run_check(run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
              find_tailscale: Callable[[], str | None] = tailscale_cli) -> LocalCheck:
    """The live check, run by check-local and by the stock process before each open."""
    return check(*read_listings(run, find_tailscale))
