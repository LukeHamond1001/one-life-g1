"""The release's version is one number: pyproject.toml and CITATION.cff carry the same, so a bump moves both."""
import os, re
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_citation_matches_the_package_version():
    py = re.search(r'^version = "([^"]+)"', open(os.path.join(ROOT, "pyproject.toml")).read(), re.M).group(1)
    cff = open(os.path.join(ROOT, "CITATION.cff")).read()
    assert re.search(r"^version: (\S+)", cff, re.M).group(1) == py, "CITATION.cff's version differs from pyproject.toml's"
    assert re.search(r"^date-released: \d{4}-\d{2}-\d{2}$", cff, re.M), "CITATION.cff has no release date"
