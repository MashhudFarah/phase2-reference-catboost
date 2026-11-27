"""Compatibility entry point for the canonical Phase 2 CatBoost trainer.

The former standalone reference runner wrote a different metadata/artifact
format. It now delegates to ``src.train`` so every prediction backend consumes
the same 34-feature, chronological CatBoost bundles. This module is not run by
the application; invoking its CLI performs the explicitly requested training.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import SUPPORTED_HORIZONS
from src.train import DEFAULT_INPUT, run_training as run_canonical_training

INPUT = DEFAULT_INPUT


def run_training(input_path: Path = INPUT, output_parent: Path | None = None) -> Path:
    """Delegate to the one authoritative three-horizon 50-iteration runner."""
    output_dir = None
    if output_parent is not None:
        output_dir = Path(output_parent) / datetime.now(timezone.utc).strftime(
            "phase2_catboost_%Y%m%dT%H%M%S_%fZ"
        )
    return run_canonical_training(
        input_path=Path(input_path), output_dir=output_dir,
        horizons=SUPPORTED_HORIZONS, iterations=50,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the canonical Phase 2 CatBoost models (2/3/4 weeks, 50 iterations each)."
    )
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output-parent", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    output = run_training(args.input, args.output_parent)
    print(f"Training run completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
