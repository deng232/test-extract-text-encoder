from __future__ import annotations

from pathlib import Path

import pytest

from extract_textencoder.cli import build_parser, run
from extract_textencoder.errors import ConversionError


def test_convert_requires_one_source() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["convert", "--output", "out.safetensors"])


def test_convert_sources_are_mutually_exclusive() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "convert",
                "--repo-id",
                "owner/model",
                "--local-dir",
                "/models/local",
                "--output",
                "out.safetensors",
            ]
        )


def test_local_source_rejects_hugging_face_options(tmp_path: Path) -> None:
    parser = build_parser()
    arguments = parser.parse_args(
        [
            "convert",
            "--local-dir",
            str(tmp_path),
            "--revision",
            "commit",
            "--output",
            "out.safetensors",
        ]
    )
    with pytest.raises(ConversionError, match="--revision"):
        run(arguments)

