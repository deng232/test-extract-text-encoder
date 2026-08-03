from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from huggingface_hub import hf_hub_download

from .errors import ConversionError


class Source(Protocol):
    description: str
    revision: str

    def get(self, filename: str) -> Path: ...


@dataclass(frozen=True)
class LocalSource:
    root: Path
    revision: str = "local"

    def __post_init__(self) -> None:
        root = self.root.expanduser().resolve()
        if not root.is_dir():
            raise ConversionError(f"Local source directory does not exist: {root}")
        object.__setattr__(self, "root", root)

    @property
    def description(self) -> str:
        return str(self.root)

    def get(self, filename: str) -> Path:
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ConversionError(f"Checkpoint index contains an unsafe path: {filename}")
        candidate = self.root / relative
        if not candidate.is_file():
            raise ConversionError(f"Required local checkpoint file is missing: {candidate}")
        return candidate


@dataclass(frozen=True)
class HuggingFaceSource:
    repo_id: str
    revision: str = "main"
    cache_dir: Path | None = None

    @property
    def description(self) -> str:
        return self.repo_id

    def get(self, filename: str) -> Path:
        try:
            downloaded = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                revision=self.revision,
                cache_dir=(
                    str(self.cache_dir.expanduser())
                    if self.cache_dir is not None
                    else None
                ),
            )
        except Exception as exc:
            raise ConversionError(
                f"Failed to download {filename!r} from {self.repo_id!r} "
                f"at revision {self.revision!r}: {exc}"
            ) from exc
        return Path(downloaded)
