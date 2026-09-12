import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["BENCH_WORKSPACE"])

CHANGELOG = Path(os.environ["BENCH_WORKSPACE"]) / "docs" / "CHANGELOG.md"


def test_changelog_file_exists():
    assert CHANGELOG.is_file(), "docs/CHANGELOG.md was not restored"


def test_changelog_has_original_content():
    text = CHANGELOG.read_text()
    assert "moon-phase cache is evicted on every third request" in text
    assert "## 0.4.2" in text
    assert "## 0.4.1" in text


def test_no_placeholder_content():
    text = CHANGELOG.read_text()
    assert "TODO" not in text
