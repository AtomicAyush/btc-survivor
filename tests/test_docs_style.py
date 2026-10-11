import pytest

from docstyle import prose_problems, section_link_problems
from helpers import ROOT

LINKED_DOCS = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *sorted((ROOT / "docs").glob("*.md"))]


@pytest.mark.parametrize("path", [*LINKED_DOCS, ROOT / "PLAN.md"], ids=lambda p: p.name)
def test_prose_rules(path):
    problems = prose_problems(path.read_text(encoding="utf-8"))
    assert not problems, f"{path.name}\n" + "\n".join(problems)


@pytest.mark.parametrize("path", LINKED_DOCS, ids=lambda p: p.name)
def test_every_section_ends_with_links(path):
    problems = section_link_problems(path.read_text(encoding="utf-8"))
    assert not problems, f"{path.name}\n" + "\n".join(problems)


def test_checker_catches_violations():
    bad = "# Title\n\nNote: this is bad; very bad — really. A well-known issue.\n"
    found = " ".join(prose_problems(bad))
    for kind in ("colon", "semicolon", "dash", "hyphen"):
        assert kind in found
    samples = [
        ("Costs -- mostly fees.", "hyphen"),
        ("well‑known", "hyphen"),
        ("a loss of −5 bps", "hyphen"),
        ("Pass the -v flag.", "hyphen"),
        ("Costs ― fees", "dash"),
        ("Done ✅", "emoji"),
        ("Alarm ⏰", "emoji"),
        ("US \U0001F1FA\U0001F1F8", "emoji"),
    ]
    for sample, kind in samples:
        assert kind in " ".join(prose_problems(sample)), sample
    assert prose_problems("- a list item\n* another\n1. numbered\n") == []
    assert prose_problems("Use `a:b;c-d` and [x](https://e.com/a-b:c) freely.\n") == []
    assert prose_problems("```\nkey: value; a-b\n```\n") == []
    assert section_link_problems("# A\n\nText without a link.\n") == ["section '# A' does not end with a link"]
    assert section_link_problems("# A\n\nText with [a link](x.md).\n## B\n\nMore [links](y.md).\n") == []
