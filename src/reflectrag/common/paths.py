from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    src: Path
    scripts: Path
    configs: Path
    docs: Path
    data: Path
    outputs: Path
    experiments: Path
    paper: Path
    secrets: Path
    source_root: Path

    @property
    def marco_runs(self) -> Path:
        return self.experiments / "marco" / "runs"

    @property
    def nq_runs(self) -> Path:
        return self.experiments / "nq" / "runs"

    @property
    def tqa_runs(self) -> Path:
        return self.experiments / "tqa" / "runs"

    @property
    def shared_outputs(self) -> Path:
        return self.outputs / "shared"

    @property
    def shared_init_lora(self) -> Path:
        return self.shared_outputs / "gemma-2-2b-lora-init"

    @property
    def source_marco(self) -> Path:
        return self.source_root / "MARCO"

    @property
    def source_nq(self) -> Path:
        return self.source_root / "NQ"

    @property
    def source_tqa(self) -> Path:
        return self.source_root / "TQA"

    @property
    def default_keys_file(self) -> Path:
        return self.secrets / "openai_api_keys.txt"


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_paths() -> ProjectPaths:
    root = project_root()
    source_root = Path(os.getenv("REFLECTRAG_SOURCE_ROOT", root / "source")).expanduser().resolve()
    return ProjectPaths(
        root=root,
        src=root / "src",
        scripts=root / "scripts",
        configs=root / "configs",
        docs=root / "docs",
        data=root / "data",
        outputs=root / "outputs",
        experiments=root / "experiments",
        paper=root / "paper",
        secrets=root / ".secrets",
        source_root=source_root,
    )


def ensure_runtime_dirs() -> ProjectPaths:
    paths = get_paths()
    for directory in (
        paths.outputs,
        paths.shared_outputs,
        paths.marco_runs,
        paths.nq_runs,
        paths.tqa_runs,
        paths.secrets,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths
