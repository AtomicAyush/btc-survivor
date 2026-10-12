import re
from pathlib import Path

import pytest

from docstyle import prose_problems, section_link_problems
from helpers import ROOT

# The version 2 docs of PLAN_V2 section 3, named one by one so a missing file fails both rules.
V2_DOCS = [ROOT / "docs" / name for name in ("v2_overview.md", "v2_data_and_licensing.md", "v2_news.md",
                                              "v2_evaluation.md", "v2_operations.md", "RESULTS_V2_TEMPLATE.md")]
LINKED_DOCS = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *sorted({*(ROOT / "docs").glob("*.md"), *V2_DOCS})]
LINK = re.compile(r"\]\(([^)\s]+)\)")


@pytest.mark.parametrize("path", [*LINKED_DOCS, ROOT / "PLAN.md", ROOT / "PLAN_V2.md"], ids=lambda p: p.name)
def test_prose_rules(path):
    problems = prose_problems(path.read_text(encoding="utf-8"))
    assert not problems, f"{path.name}\n" + "\n".join(problems)


@pytest.mark.parametrize("path", LINKED_DOCS, ids=lambda p: p.name)
def test_every_section_ends_with_links(path):
    problems = section_link_problems(path.read_text(encoding="utf-8"))
    assert not problems, f"{path.name}\n" + "\n".join(problems)


def anchors(text: str) -> set[str]:
    """The heading anchors GitHub gives a markdown file, leaving out lines inside code blocks."""
    found: set[str] = set()
    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        elif not in_code and line.startswith("#"):
            slug = re.sub(r"[^\w\- ]", "", line.lstrip("#").strip().lower()).replace(" ", "-")
            n = 0
            while (slug if n == 0 else f"{slug}-{n}") in found:
                n += 1
            found.add(slug if n == 0 else f"{slug}-{n}")
    return found


def link_problems(path: Path) -> list[str]:
    """Every relative link of a doc must name a file that exists, and an anchor must name one of its headings."""
    text = path.read_text(encoding="utf-8")
    prose = re.sub(r"```.*?```", "", text, flags=re.S)
    problems = []
    for target in LINK.findall(re.sub(r"`[^`]*`", "", prose)):
        if re.match(r"[a-z]+:", target):
            continue
        name, _, anchor = target.partition("#")
        linked = (path.parent / name).resolve() if name else path
        if not linked.exists():
            problems.append(f"{target} names no file")
        elif anchor and linked.suffix == ".md" and anchor not in anchors(linked.read_text(encoding="utf-8")):
            problems.append(f"{target} names no heading")
    return problems


@pytest.mark.parametrize("path", V2_DOCS, ids=lambda p: p.name)
def test_version_2_docs_link_only_to_files_and_headings_that_exist(path):
    problems = link_problems(path)
    assert not problems, f"{path.name}\n" + "\n".join(problems)


def test_the_link_check_finds_missing_files_and_headings(tmp_path):
    (tmp_path / "other.md").write_text("# Other\n\n## Two words\n\n```\n# not a heading\n```\n", encoding="utf-8")
    doc = tmp_path / "doc.md"
    doc.write_text("# Doc\n\n[a](other.md#two-words) [b](other.md#not-a-heading) [c](gone.md) [d](#doc) "
                   "[e](#nothing) [f](https://example.com/x#y) `[g](inline.md)`\n", encoding="utf-8")
    assert link_problems(doc) == ["other.md#not-a-heading names no heading", "gone.md names no file",
                                  "#nothing names no heading"]
    assert anchors("# A b\n## A b\n## 13.1 Why, and how\n") == {"a-b", "a-b-1", "131-why-and-how"}


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
