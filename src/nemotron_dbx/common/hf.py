"""Pinned Hugging Face sources (SPEC design principle 5: model code is pulled from the Hub, so pin it)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HFSource:
    repo_id: str
    revision: str
    repo_type: str = "model"

    def snapshot(self, local_dir: str | Path, allow_patterns: list[str] | None = None) -> Path:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=self.repo_id,
            revision=self.revision,
            repo_type=self.repo_type,
            local_dir=str(local_dir),
            allow_patterns=allow_patterns,
        )
        return Path(path)

    def file(self, filename: str, local_dir: str | Path | None = None) -> Path:
        """Download one file. With `local_dir`, the file is a real copy there (safe to delete)."""
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                self.repo_id, filename, revision=self.revision, repo_type=self.repo_type,
                local_dir=str(local_dir) if local_dir else None,
            )
        )


# Revisions observed on 2026-11-10 (SPEC §11).
PAGE_ELEMENTS_V3 = HFSource("nvidia/nemotron-page-elements-v3", "df62dbb631502575ac4d43b44d700b1674ab1d56")
DOCLAYNET_V1_2 = HFSource(
    "docling-project/DocLayNet-v1.2", "0daf93102e2efce76c3e11a274a5e0d0969391d3", repo_type="dataset"
)
NEMOTRON_OCR_V1 = HFSource("nvidia/nemotron-ocr-v1", "8657d08d3279f4864002d5fd3fdcd47ad8c96bcb")
CORD_V2 = HFSource("naver-clova-ix/cord-v2", "7f0115a4b758a71d6473b8d085751692da2fef98", repo_type="dataset")
