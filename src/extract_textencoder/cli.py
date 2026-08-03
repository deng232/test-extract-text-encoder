from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .converter import convert, human_size
from .errors import ConversionError
from .sources import HuggingFaceSource, LocalSource
from .validation import inspect_saved_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract-textencoder",
        description=(
            "Extract a compatible Mistral Small 3.2 24B checkpoint into a "
            "standalone FLUX.2 text encoder."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    convert_parser = commands.add_parser("convert", help="convert a checkpoint")
    source = convert_parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo-id", help="Hugging Face repository ID")
    source.add_argument(
        "--local-dir", type=Path, help="local Hugging Face snapshot directory"
    )
    convert_parser.add_argument(
        "--revision", default="main", help="Hugging Face branch, tag, or commit"
    )
    convert_parser.add_argument(
        "--cache-dir", type=Path, help="optional Hugging Face cache directory"
    )
    convert_parser.add_argument("--output", required=True, type=Path)
    convert_parser.add_argument("--overwrite", action="store_true")
    convert_parser.add_argument(
        "--allow-non-bf16",
        action="store_true",
        help="preserve non-BF16 source dtypes; this does not quantize weights",
    )

    inspect_parser = commands.add_parser(
        "inspect", help="validate an extracted checkpoint without running it"
    )
    inspect_parser.add_argument("checkpoint", type=Path)
    return parser


def run(arguments: argparse.Namespace) -> int:
    if arguments.command == "inspect":
        result = inspect_saved_file(arguments.checkpoint)
        print("Checkpoint is structurally valid.")
        print(f"File:       {result['path']}")
        print(f"Size:       {human_size(int(result['size']))}")
        print(f"Tensors:    {result['tensor_count']}")
        print(f"Dtypes:     {', '.join(result['model_dtypes'])}")
        print(f"Tekken:     {human_size(int(result['tekken_bytes']))}")
        metadata = result["metadata"]
        if metadata:
            print("Metadata:")
            for key, value in sorted(metadata.items()):
                print(f"  {key}: {value}")
        return 0

    if arguments.command == "convert":
        if arguments.local_dir is not None:
            if arguments.revision != "main":
                raise ConversionError("--revision can only be used with --repo-id.")
            if arguments.cache_dir is not None:
                raise ConversionError("--cache-dir can only be used with --repo-id.")
            source = LocalSource(arguments.local_dir)
        else:
            if arguments.repo_id is None:
                raise AssertionError("argparse did not provide a repository ID")
            source = HuggingFaceSource(
                repo_id=arguments.repo_id,
                revision=arguments.revision,
                cache_dir=arguments.cache_dir,
            )
        convert(
            source=source,
            output=arguments.output,
            overwrite=arguments.overwrite,
            allow_non_bf16=arguments.allow_non_bf16,
        )
        return 0

    raise AssertionError(f"Unhandled command: {arguments.command}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        return run(parser.parse_args(argv))
    except ConversionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
