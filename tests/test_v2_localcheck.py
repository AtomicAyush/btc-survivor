"""The local only check of PLAN_V2 section 13.2 on fixed lsof and ps listings, so no test reads this machine."""

import json
import subprocess
from pathlib import PurePath

import pytest

from btc_survivor.v2 import localcheck
from btc_survivor.v2.localcheck import (LSOF, PS, SOCKETS, LocalCheckError, Listener, Socket, Tunnel, check,
                                        tunnel_kind)

LOOPBACK_LSOF = """p501
cPython
f7
n127.0.0.1:8050
p502
cPython
f9
n127.0.0.1:8060
"""

QUIET_PS = """    1 /sbin/launchd
  501 /Users/owner/code/btc_survivor/.venv311/bin/python -m btc_survivor dashboard --config configs/run_a.toml
  502 /Users/owner/code/btc_survivor_v2/.venv-v2/bin/python -m btc_survivor.v2 dashboard s1.toml s2.toml
  610 /usr/bin/ssh -o SendEnv=GIT_PROTOCOL git@github.com git-upload-pack 'owner/repo.git'
  611 ssh -L 9000:localhost:9000 -p 2222 -D 1080 build@example.com
  612 /usr/bin/grep cloudflared
  613 vim notes_on_ngrok_and_ssh_-R.txt
  614 /Applications/Tailscale.app/Contents/MacOS/Tailscale
  615 /usr/local/bin/tailscaled
  616 tailscale status
"""

TOKEN = "fake-tunnel-token-canary"

# netstat -an -p tcp on macOS, where the port follows the last dot, with a socket held by another user that lsof
# run by the owner would not list.
MAC_NETSTAT = """Active Internet connections (including servers)
Proto Recv-Q Send-Q  Local Address          Foreign Address        (state)
tcp4       0      0  192.168.1.20.51150     203.0.113.10.443       ESTABLISHED
tcp4       0      0  127.0.0.1.8050         *.*                    LISTEN
tcp4       0      0  127.0.0.1.8060         *.*                    LISTEN
tcp4       0      0  127.0.0.1.18060        *.*                    LISTEN
tcp46      0      0  *.5000                 *.*                    LISTEN
"""

# ss -ltnH on Linux, where the port follows the last colon.
LINUX_SS = """LISTEN 0      4096       127.0.0.1:8060      0.0.0.0:*
LISTEN 0      4096       127.0.0.1:8050      0.0.0.0:*
LISTEN 0      128          0.0.0.0:22        0.0.0.0:*
"""


def with_process(command: str, pid: int = 900) -> str:
    return QUIET_PS + f"  {pid} {command}\n"


def test_loopback_listeners_and_ordinary_processes_pass():
    result = check(LOOPBACK_LSOF, QUIET_PS)
    assert result.passed
    assert result.listeners == (Listener(501, "Python", "127.0.0.1", 8050), Listener(502, "Python", "127.0.0.1", 8060))
    assert result.exposed == () and result.tunnels == ()


def test_no_listener_and_no_process_passes():
    assert check("", "", "", ()).passed


@pytest.mark.parametrize("sockets", [MAC_NETSTAT, LINUX_SS])
def test_every_users_loopback_sockets_pass(sockets):
    result = check(LOOPBACK_LSOF, QUIET_PS, sockets)
    assert result.passed and result.loopback_listeners == 2 and result.offending_sockets == 0
    assert sorted(result.sockets, key=lambda s: s.port) == [Socket("127.0.0.1", 8050), Socket("127.0.0.1", 8060)]


@pytest.mark.parametrize("line, address", [
    ("tcp4       0      0  *.8060                 *.*                    LISTEN", "*"),
    ("tcp46      0      0  *.8050                 *.*                    LISTEN", "*"),
    ("tcp4       0      0  0.0.0.0.8060           *.*                    LISTEN", "0.0.0.0"),
    ("tcp6       0      0  ::1.8060               *.*                    LISTEN", "::1"),
    ("tcp4       0      0  192.168.1.20.8050      *.*                    LISTEN", "192.168.1.20"),
    ("LISTEN 0      4096         0.0.0.0:8060      0.0.0.0:*", "0.0.0.0"),
    ("LISTEN 0      4096            [::]:8050         [::]:*", "::"),
    ("LISTEN 0      4096               *:8060            *:*", "*"),
])
def test_another_users_socket_that_lsof_cannot_see_fails(line, address):
    result = check(LOOPBACK_LSOF, QUIET_PS, MAC_NETSTAT + line + "\n")
    assert not result.passed and result.exposed == () and result.tunnels == ()
    assert [s.address for s in result.exposed_sockets] == [address] and result.offending_sockets == 1


def test_a_socket_seen_by_both_listings_counts_once():
    result = check(LOOPBACK_LSOF + "p777\ncnc\nf3\nn*:8060\n", QUIET_PS,
                   MAC_NETSTAT + "tcp4       0      0  *.8060                 *.*                    LISTEN\n")
    assert not result.passed and result.offending_sockets == 1


@pytest.mark.parametrize("name", ["0.0.0.0:8060", "*:8060", "[::]:8060", "192.168.1.20:8060", "[::1]:8060",
                                  "*:8050"])
def test_a_dashboard_port_on_any_other_address_fails(name):
    result = check(LOOPBACK_LSOF + f"p777\ncnc\nf3\nn{name}\n", QUIET_PS)
    assert not result.passed and result.tunnels == ()
    assert len(result.exposed) == 1 and result.exposed[0].pid == 777 and result.exposed[0].program == "nc"
    assert len(result.listeners) == 3


def test_a_listener_on_8060_at_0_0_0_0_fails_and_other_ports_are_ignored():
    lsof = "p40\ncControlCe\nf9\nn*:5000\nf10\nn*:7000\np41\ncPython\nf4\nn0.0.0.0:18060\nf5\nn0.0.0.0:8060\n"
    result = check(lsof, QUIET_PS)
    assert not result.passed
    assert result.listeners == result.exposed == (Listener(41, "Python", "0.0.0.0", 8060),)


@pytest.mark.parametrize("command", [
    "/opt/homebrew/bin/cloudflared tunnel --url http://127.0.0.1:8060",
    f"cloudflared tunnel run --token {TOKEN}",
])
def test_a_cloudflared_process_fails(command):
    result = check(LOOPBACK_LSOF, with_process(command))
    assert not result.passed and result.exposed == ()
    assert result.tunnels == (Tunnel(900, "cloudflared"),)


def test_a_tunnel_result_never_holds_its_command_line():
    result = check(LOOPBACK_LSOF, with_process(f"cloudflared tunnel run --token {TOKEN}"))
    assert TOKEN not in repr(result)


@pytest.mark.parametrize("command", [
    "ssh -R 80:localhost:8060 serveo.net",
    "/usr/bin/ssh -NfR 9000:127.0.0.1:8050 owner@example.com",
    "ssh -R9000:127.0.0.1:8050 owner@example.com",
    "ssh owner@example.com -R 9000:localhost:8060",
    "ssh -o RemoteForward=9000 127.0.0.1:8060 owner@example.com",
    "ssh -oRemoteForward=9000:127.0.0.1:8060 owner@example.com",
])
def test_an_ssh_remote_forward_fails(command):
    result = check(LOOPBACK_LSOF, with_process(command))
    assert not result.passed and result.tunnels == (Tunnel(900, "ssh -R"),)


@pytest.mark.parametrize("command", [
    "ssh -g -L 9000:localhost:8060 owner@example.com",
    "ssh -NgL 9000:localhost:8060 owner@example.com",
    "ssh -L 0.0.0.0:9000:localhost:8060 owner@example.com",
    "ssh -L*:9000:localhost:8060 owner@example.com",
    "ssh -L [::]:9000:localhost:8060 owner@example.com",
    "ssh -L :9000:localhost:8060 owner@example.com",
    "ssh -o GatewayPorts=yes -L 9000:localhost:8060 owner@example.com",
])
def test_an_ssh_forward_other_machines_can_reach_fails(command):
    result = check(LOOPBACK_LSOF, with_process(command))
    assert not result.passed and result.tunnels == (Tunnel(900, "ssh -g"),)


@pytest.mark.parametrize("command", [
    "ssh -L 9000:localhost:8050 owner@example.com",
    "ssh -L 127.0.0.1:9000:localhost:8050 owner@example.com",
    "ssh -L localhost:9000:localhost:8050 owner@example.com",
    "ssh -L [::1]:9000:localhost:8050 owner@example.com",
    "ssh -o GatewayPorts=no owner@example.com",
    "ssh -iRkey owner@example.com",
    "ssh -i -gkey owner@example.com",
    "ssh -p 2222 -o ServerAliveInterval=60 owner@example.com",
    "sshd: owner@ttys001",
])
def test_ssh_without_a_remote_or_open_forward_passes(command):
    assert check(LOOPBACK_LSOF, with_process(command)).passed


@pytest.mark.parametrize("command, kind", [
    ("tailscale funnel 8060", "tailscale funnel"),
    ("/Applications/Tailscale.app/Contents/MacOS/Tailscale serve --bg 8050", "tailscale serve"),
    ("ngrok http 8060", "ngrok"),
    ("/usr/local/bin/frpc -c frpc.toml", "frpc"),
    ("/opt/homebrew/bin/cloudflared-darwin-arm64 tunnel --url http://127.0.0.1:8060", "cloudflared"),
    ("node /usr/local/bin/lt --port 8060", "lt"),
    ("node /Users/owner/.npm/_npx/0123abcd/node_modules/.bin/lt --port 8060", "lt"),
    ("/usr/local/bin/localtunnel --port 8060", "localtunnel"),
    ("devtunnel host -p 8060", "devtunnel"),
    ("/usr/local/bin/code tunnel --accept-server-license-terms", "code tunnel"),
    ("Code.app/Contents/Resources/app/bin/code-tunnel tunnel", "code-tunnel"),
    ("socat TCP-LISTEN:9000,fork TCP:127.0.0.1:8060", "socat"),
    ("bore local 8060 --to bore.pub", "bore"),
])
def test_other_tunnels_fail(command, kind):
    assert tunnel_kind(command.split()) == kind
    assert not check(LOOPBACK_LSOF, with_process(command)).passed


@pytest.mark.parametrize("command", [
    "socat UNIX-LISTEN:/tmp/fake.sock,fork TCP:127.0.0.1:9000",
    "/usr/bin/grep -r ngrok /usr/local/bin",
    "vim /tmp/notes_on_cloudflared.txt",
    "python -m http.server 18060",
    "/usr/local/bin/code --wait notes.txt",
    "/usr/local/bin/blt --port 8060",
])
def test_ordinary_processes_that_name_a_tunnel_pass(command):
    assert tunnel_kind(command.split()) is None


def test_bad_sockets_and_tunnels_are_counted_together():
    result = check(LOOPBACK_LSOF + "p777\ncPython\nf3\nn*:8060\n",
                   with_process("ngrok http 8060", 901) + "  902 ssh -R 1:localhost:8050 example.com\n")
    assert not result.passed and len(result.exposed) == 1 and len(result.tunnels) == 2


TAILSCALE = "/Applications/Tailscale.app/Contents/MacOS/Tailscale"
SERVE_CONFIG = json.dumps({"TCP": {"443": {"HTTPS": True}},
                           "Web": {"fake-host.example.ts.net:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8060"}}}}})
FUNNEL_CONFIG = json.loads(SERVE_CONFIG) | {"AllowFunnel": {"fake-host.example.ts.net:443": True}}


class FakeRun:
    """Stands in for subprocess.run, answering lsof, netstat, ps and tailscale from fixed results and recording
    each command."""

    def __init__(self, lsof, ps=(0, QUIET_PS, ""), sockets=(0, MAC_NETSTAT, ""), serve=(0, "{}\n", ""),
                 funnel=(0, "{}\n", "")):
        self.answers = {"lsof": lsof, "ps": ps, SOCKETS[0]: sockets, "serve": serve, "funnel": funnel}
        self.commands = []

    def __call__(self, command, **kwargs):
        self.commands.append(tuple(command))
        assert kwargs["timeout"] > 0 and kwargs["capture_output"] and kwargs["text"]
        program = PurePath(command[0]).name
        answer = self.answers[command[1] if program == "Tailscale" else program]
        if isinstance(answer, BaseException):
            raise answer
        code, out, err = answer
        return subprocess.CompletedProcess(command, code, out, err)


def no_tailscale():
    return None


def tailscale_app():
    return TAILSCALE


def test_the_live_check_runs_lsof_netstat_and_ps_without_tailscale():
    run = FakeRun((0, LOOPBACK_LSOF, ""))
    assert localcheck.run_check(run, no_tailscale).passed
    assert run.commands == [LSOF, PS, SOCKETS]
    assert "-iTCP:8050,8060" in LSOF and "-sTCP:LISTEN" in LSOF


def test_the_live_check_reads_tailscale_settings_when_it_is_installed():
    run = FakeRun((0, LOOPBACK_LSOF, ""))
    assert localcheck.run_check(run, tailscale_app).passed
    assert run.commands[3:] == [(TAILSCALE, "serve", "status", "--json"), (TAILSCALE, "funnel", "status", "--json")]


@pytest.mark.parametrize("serve, funnel", [((0, SERVE_CONFIG, ""), (0, "{}", "")),
                                           ((0, "{}", ""), (0, json.dumps(FUNNEL_CONFIG), "")),
                                           ((0, SERVE_CONFIG, ""), (0, json.dumps(FUNNEL_CONFIG), ""))])
def test_a_background_tailscale_serve_or_funnel_fails(serve, funnel):
    result = localcheck.run_check(FakeRun((0, LOOPBACK_LSOF, ""), serve=serve, funnel=funnel), tailscale_app)
    assert not result.passed and result.tunnels == (Tunnel(0, "tailscale serve config"),)
    assert "fake-host" not in repr(result)


@pytest.mark.parametrize("text", ["{}", "{}\n", "", '{"TCP": null, "Web": {}}'])
def test_an_empty_tailscale_setting_passes(text):
    assert not localcheck.serve_configured(text)


def test_lsof_finding_no_listener_passes():
    assert localcheck.run_check(FakeRun((1, "", "")), no_tailscale).listeners == ()


@pytest.mark.parametrize("lsof, ps, sockets", [
    ((1, "", "lsof: unknown option"), (0, QUIET_PS, ""), (0, MAC_NETSTAT, "")),
    ((2, "", ""), (0, QUIET_PS, ""), (0, MAC_NETSTAT, "")),
    (FileNotFoundError("lsof"), (0, QUIET_PS, ""), (0, MAC_NETSTAT, "")),
    (subprocess.TimeoutExpired("lsof", 30), (0, QUIET_PS, ""), (0, MAC_NETSTAT, "")),
    ((0, LOOPBACK_LSOF, ""), (1, "", ""), (0, MAC_NETSTAT, "")),
    ((0, LOOPBACK_LSOF, ""), PermissionError("ps"), (0, MAC_NETSTAT, "")),
    ((0, LOOPBACK_LSOF, ""), (0, QUIET_PS, ""), (1, "", "")),
    ((0, LOOPBACK_LSOF, ""), (0, QUIET_PS, ""), FileNotFoundError("netstat")),
])
def test_a_listing_that_cannot_be_read_fails_closed(lsof, ps, sockets):
    with pytest.raises(LocalCheckError):
        localcheck.run_check(FakeRun(lsof, ps, sockets), no_tailscale)


@pytest.mark.parametrize("serve, funnel", [((1, "", "failed to connect to local Tailscale daemon"), (0, "{}", "")),
                                           ((0, "{}", ""), subprocess.TimeoutExpired("tailscale", 30)),
                                           ((0, "No serve config", ""), (0, "{}", ""))])
def test_an_installed_tailscale_that_cannot_answer_fails_closed(serve, funnel):
    with pytest.raises(LocalCheckError):
        localcheck.run_check(FakeRun((0, LOOPBACK_LSOF, ""), serve=serve, funnel=funnel), tailscale_app)
