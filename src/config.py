"""Configuration resolution for the Warsaw EV Charging pipeline.

Precedence: command-line arguments > environment variables > optional JSON
config file > built-in defaults. This keeps local paths out of source code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = Path.home() / "Downloads" / "OneDrive_2026-10-14" / "Warsaw EV Charging"
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.json"


class ConfigError(RuntimeError):
    """Raised when configuration is invalid or incomplete."""


@dataclass(frozen=True)
class PipelineConfig:
    data_dir: Path
    model_dir: Path
    reports_dir: Path
    random_state: int = 42
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    n_jobs: int = -1

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"

    @property
    def model_path(self) -> Path:
        return self.model_dir / "best_model.joblib"

    @property
    def metadata_path(self) -> Path:
        return self.model_dir / "model_metadata.json"

    @property
    def metrics_path(self) -> Path:
        return self.reports_dir / "model_metrics.json"

    def ensure_output_dirs(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "model_dir": str(self.model_dir),
            "reports_dir": str(self.reports_dir),
            "random_state": self.random_state,
            "validation_fraction": self.validation_fraction,
            "test_fraction": self.test_fraction,
            "n_jobs": self.n_jobs,
        }


def _read_json_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError(f"Config file {path} must contain a JSON object.")
    return payload


def _as_path(value: Any) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def _value_from_args(args: Any, name: str) -> Any:
    return getattr(args, name, None) if args is not None else None


def load_config(args: Any | None = None) -> PipelineConfig:
    """Build a validated configuration from CLI args, env vars and JSON."""

    cli_config = _as_path(_value_from_args(args, "config"))
    env_config = _as_path(os.environ.get("WARSAW_EV_CONFIG"))
    file_path = cli_config or env_config or (CONFIG_FILE if CONFIG_FILE.exists() else None)
    file_cfg = _read_json_config(file_path) if file_path else {}

    data_dir = (
        _as_path(_value_from_args(args, "data_dir"))
        or _as_path(os.environ.get("WARSAW_EV_DATA_DIR"))
        or _as_path(file_cfg.get("data_dir"))
        or DEFAULT_DATA_DIR
    )
    model_dir = (
        _as_path(_value_from_args(args, "model_dir"))
        or _as_path(os.environ.get("WARSAW_EV_MODEL_DIR"))
        or _as_path(file_cfg.get("model_dir"))
        or (PROJECT_ROOT / "models")
    )
    reports_dir = (
        _as_path(_value_from_args(args, "reports_dir"))
        or _as_path(os.environ.get("WARSAW_EV_REPORTS_DIR"))
        or _as_path(file_cfg.get("reports_dir"))
        or (PROJECT_ROOT / "reports")
    )
    cli_seed = _value_from_args(args, "seed")
    random_state = int(cli_seed if cli_seed is not None else file_cfg.get("random_state", 42))

    try:
        validation_fraction = float(file_cfg.get("validation_fraction", 0.15))
        test_fraction = float(file_cfg.get("test_fraction", 0.15))
    except (TypeError, ValueError) as exc:
        raise ConfigError("validation_fraction and test_fraction must be numeric.") from exc

    if not 0.0 < validation_fraction < 1.0:
        raise ConfigError("validation_fraction must be between 0 and 1.")
    if not 0.0 < test_fraction < 1.0:
        raise ConfigError("test_fraction must be between 0 and 1.")
    if validation_fraction + test_fraction >= 1.0:
        raise ConfigError("validation_fraction + test_fraction must be less than 1.")

    config = PipelineConfig(
        data_dir=data_dir,
        model_dir=model_dir,
        reports_dir=reports_dir,
        random_state=random_state,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        n_jobs=int(file_cfg.get("n_jobs", -1)),
    )
    if not config.data_dir.exists():
        raise ConfigError(
            f"Dataset directory not found: {config.data_dir}\n"
            "Set it with --data-dir, WARSAW_EV_DATA_DIR, or configs/config.json."
        )
    if not config.data_dir.is_dir():
        raise ConfigError(f"Dataset path is not a directory: {config.data_dir}")
    return config


def add_common_arguments(parser: Any) -> None:
    """Attach the shared CLI options used by train/evaluate/predict."""

    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config file.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Directory containing the four source CSVs.")
    parser.add_argument("--model-dir", type=Path, default=None, help="Directory for trained model artifacts.")
    parser.add_argument("--reports-dir", type=Path, default=None, help="Directory for reports and figures.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: 42).")
