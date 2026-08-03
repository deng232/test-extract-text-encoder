from __future__ import annotations

from pathlib import Path

import pytest

from extract_textencoder.errors import ConversionError
from extract_textencoder.sources import LocalSource


def test_local_source_resolves_existing_file(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    source = LocalSource(tmp_path)
    assert source.get("config.json") == config.resolve()


def test_local_source_rejects_missing_file(tmp_path: Path) -> None:
    source = LocalSource(tmp_path)
    with pytest.raises(ConversionError, match="missing"):
        source.get("config.json")


def test_local_source_rejects_parent_traversal(tmp_path: Path) -> None:
    source = LocalSource(tmp_path)
    with pytest.raises(ConversionError, match="unsafe path"):
        source.get("../outside.safetensors")

