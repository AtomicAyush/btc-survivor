"""The secrets reader, on files written here with fake values only. No test reads the real secrets file."""

import os
import subprocess
import sys

import pytest

from btc_survivor.v2 import local_secrets, redact
from btc_survivor.v2.local_secrets import ANTHROPIC, CONTACT, FINNHUB, NAMES, SecretsError, load_secrets
from test_no_orders import NETWORK_MODULES

FAKE = {FINNHUB: "fakefinnhub0001token", ANTHROPIC: "fake-anthropic-0002-value", CONTACT: "fake.owner@example.invalid"}
FULL = f"# fake values for tests\n\n{FINNHUB}={FAKE[FINNHUB]}\n {ANTHROPIC} = {FAKE[ANTHROPIC]} \n{CONTACT}={FAKE[CONTACT]}\n"


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch):
    monkeypatch.setattr(redact, "_values", set())


def write(folder, text, mode=0o600):
    path = folder / "secrets.env"
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)
    return path


def leaked(text):
    return [name for name, value in FAKE.items() if value in text]


def test_parses_plain_lines_skipping_blanks_and_comments(tmp_path):
    secrets = load_secrets(write(tmp_path, FULL))
    assert secrets.present == NAMES
    assert secrets.missing == ()
    for name in NAMES:
        assert secrets.get(name) == FAKE[name]


def test_expands_the_home_folder(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    write(tmp_path, FULL)
    assert load_secrets("~/secrets.env").present == NAMES


@pytest.mark.parametrize("mode", [0o640, 0o604, 0o644, 0o660, 0o620, 0o602, 0o610])
def test_refuses_a_file_others_can_reach(tmp_path, mode):
    with pytest.raises(SecretsError, match="chmod 600") as info:
        load_secrets(write(tmp_path, FULL, mode))
    assert f"{mode:o}" in str(info.value)
    assert not leaked(str(info.value))
    assert redact.redact(FULL) == FULL


def test_accepts_an_owner_read_only_file(tmp_path):
    assert load_secrets(write(tmp_path, FULL, 0o400)).present == NAMES


def test_refuses_a_file_another_user_owns(tmp_path, monkeypatch):
    path = write(tmp_path, FULL)
    owner = os.getuid()
    monkeypatch.setattr(local_secrets.os, "getuid", lambda: owner + 1)
    with pytest.raises(SecretsError, match="not owned by the current user") as info:
        load_secrets(path)
    assert not leaked(str(info.value))
    assert redact.redact(FULL) == FULL


def test_a_missing_file_is_named(tmp_path):
    with pytest.raises(SecretsError, match="does not exist"):
        load_secrets(tmp_path / "absent.env")


def test_a_missing_key_is_named_without_any_value(tmp_path):
    secrets = load_secrets(write(tmp_path, f"{FINNHUB}={FAKE[FINNHUB]}\n"))
    assert secrets.present == (FINNHUB,)
    assert secrets.missing == (ANTHROPIC, CONTACT)
    with pytest.raises(SecretsError, match=CONTACT) as info:
        secrets.get(CONTACT)
    assert not leaked(str(info.value))
    with pytest.raises(SecretsError) as info:
        secrets.require(FINNHUB, ANTHROPIC, CONTACT)
    message = str(info.value)
    assert ANTHROPIC in message and CONTACT in message and FINNHUB not in message
    assert not leaked(message)
    secrets.require(FINNHUB)


@pytest.mark.parametrize("text, expected", [
    ("OTHER_TOKEN=fakefinnhub0001token\n", "line 1 has an unknown name"),
    ("# pasted value\nFAKEFINNHUB0001TOKEN==\n", "line 2 has an unknown name"),
    (f"{FINNHUB}=fakefinnhub0001token\n{FINNHUB}=fakefinnhub0001token\n", "more than once"),
    (f"{FINNHUB}=\n", f"{FINNHUB} is empty"),
    ("fakefinnhub0001token\n", "line 1 is not a KEY=value line"),
    ("# comment\nfake.owner@example.invalid=x\n", "line 2 is not a KEY=value line"),
    ("export FINNHUB_API_KEY=fakefinnhub0001token\n", "line 1 is not a KEY=value line"),
])
def test_unknown_or_malformed_lines_are_refused_without_values(tmp_path, text, expected):
    with pytest.raises(SecretsError, match=expected) as info:
        load_secrets(write(tmp_path, text))
    assert not leaked(str(info.value))
    assert "OTHER_TOKEN" not in str(info.value) and "FAKEFINNHUB" not in str(info.value)


def test_parse_reads_lines_and_never_needs_the_whole_text():
    assert local_secrets.parse(iter([f"{FINNHUB}={FAKE[FINNHUB]}\n", "\n"])) == {FINNHUB: FAKE[FINNHUB]}


def test_an_unknown_name_on_get_is_refused(tmp_path):
    with pytest.raises(SecretsError, match="not one of"):
        load_secrets(write(tmp_path, FULL)).get("OTHER")


def test_repr_and_str_show_only_which_keys_exist(tmp_path):
    secrets = load_secrets(write(tmp_path, f"{FINNHUB}={FAKE[FINNHUB]}\n{CONTACT}={FAKE[CONTACT]}\n"))
    for text in (repr(secrets), str(secrets), f"{secrets}", f"{[secrets]}"):
        assert "Secrets(present=" in text and FINNHUB in text and CONTACT in text and ANTHROPIC not in text
        assert not leaked(text)


def test_every_value_is_registered_with_redact(tmp_path):
    load_secrets(write(tmp_path, FULL))
    masked = redact.redact(" ".join(FAKE.values()))
    assert not leaked(masked)
    assert masked.count(redact.MASK) == 3


def test_check_names_keys_and_never_values(tmp_path):
    ok, lines = local_secrets.check(write(tmp_path, FULL))
    assert ok and lines[0].startswith("pass: 3 of 3")
    assert all(f"present: {name}" in lines for name in NAMES)
    ok, lines = local_secrets.check(write(tmp_path, f"{CONTACT}={FAKE[CONTACT]}\n"))
    assert not ok and f"missing: {FINNHUB}" in lines and f"missing: {ANTHROPIC}" in lines
    ok, lines = local_secrets.check(write(tmp_path, FULL, 0o644))
    assert not ok and len(lines) == 1 and "mode 644" in lines[0]
    assert not leaked("\n".join(lines))


def test_anthropic_environment_is_scrubbed_and_given_only_the_key(tmp_path):
    secrets = load_secrets(write(tmp_path, FULL))
    environ = {"ANTHROPIC_AUTH_TOKEN": "fake-auth", "ANTHROPIC_PROFILE": "fake-profile",
               "ANTHROPIC_BASE_URL": "http://127.0.0.1:9", "ANTHROPIC_CUSTOM_HEADERS": "fake-header",
               "ANTHROPIC_API_KEY": "fake-old-key", "NO_PROXY": "localhost", "PATH": "/usr/bin"}
    local_secrets.prepare_anthropic_env(secrets, environ)
    assert sorted(environ) == [ANTHROPIC, "NO_PROXY", "PATH"]
    assert environ[ANTHROPIC] == FAKE[ANTHROPIC]


@pytest.mark.parametrize("name", ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "https_proxy", "all_proxy",
                                  "SSL_CERT_FILE", "SSL_CERT_DIR"])
def test_a_proxy_or_certificate_variable_refuses_to_start_and_changes_nothing(tmp_path, name):
    secrets = load_secrets(write(tmp_path, FULL))
    environ = {name: "http://fake-proxy.invalid:3128", "ANTHROPIC_BASE_URL": "http://127.0.0.1:9"}
    before = dict(environ)
    with pytest.raises(SecretsError, match=name) as info:
        local_secrets.prepare_anthropic_env(secrets, environ)
    assert environ == before
    assert "fake-proxy" not in str(info.value) and not leaked(str(info.value))


def test_anthropic_log_refuses_to_start_and_changes_nothing(tmp_path):
    secrets = load_secrets(write(tmp_path, FULL))
    environ = {"ANTHROPIC_LOG": "debug", "ANTHROPIC_BASE_URL": "http://127.0.0.1:9"}
    with pytest.raises(SecretsError, match="ANTHROPIC_LOG"):
        local_secrets.prepare_anthropic_env(secrets, environ)
    assert environ == {"ANTHROPIC_LOG": "debug", "ANTHROPIC_BASE_URL": "http://127.0.0.1:9"}


def test_anthropic_environment_needs_the_key(tmp_path):
    secrets = load_secrets(write(tmp_path, f"{FINNHUB}={FAKE[FINNHUB]}\n"))
    environ = {"ANTHROPIC_PROFILE": "fake-profile"}
    with pytest.raises(SecretsError, match=ANTHROPIC):
        local_secrets.prepare_anthropic_env(secrets, environ)
    assert ANTHROPIC not in environ


# pathlib imports urllib.parse for Path.as_uri, and urllib.parse only splits and quotes text.
LOADED_BY_PATHLIB = {"urllib", "urllib.parse"}


def test_imports_no_network_module():
    code = ("import sys\n"
            "import btc_survivor.v2.local_secrets\n"
            f"net = {sorted(NETWORK_MODULES)!r}\n"
            f"allowed = {sorted(LOADED_BY_PATHLIB)!r}\n"
            "print(sorted(m for m in sys.modules if set(m.split('.')) & set(net) and m not in allowed))\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "[]"
