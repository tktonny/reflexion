#!/usr/bin/env python3
"""Train a custom "Hey Aria" openWakeWord detector and export ONNX for the mirror.

This wraps openWakeWord's OFFICIAL automatic-training flow (config-driven). It:
  1. writes a training config for the phrase "Hey Aria",
  2. lets openWakeWord synthesize positives (Piper TTS) + pull negative/background sets,
  3. trains the small wakeword head and exports `output/hey_aria.onnx`.

openWakeWord's training API has shifted across versions. This script targets the config-driven
entrypoint documented in openwakeword/notebooks/automatic_model_training.ipynb. If your installed
version exposes a different entrypoint, run that notebook with the config produced here (printed at the
end) — the config is the stable, portable part.

Usage:  python train_hey_aria.py [--phrase "Hey Aria"] [--samples 30000] [--out output]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build_config(phrase: str, samples: int, out_dir: Path) -> dict:
    # Keys follow openWakeWord's automatic-training config. Augmentation is tuned for an elderly-home,
    # far-field, TV-noise deployment (the mirror's real environment).
    return {
        "target_phrase": [phrase],
        "model_name": "hey_aria",
        "n_samples": samples,
        "n_samples_val": max(2000, samples // 15),
        "output_dir": str(out_dir),
        "tts_batch_size": 50,
        "augmentation_batch_size": 16,
        # Realistic acoustics: room impulse responses + background noise (kitchen/TV/street).
        "augmentation_rounds": 1,
        "background_paths_duplication_rate": [1],
        "rir_paths": ["mit_rirs"],
        "background_paths": ["audioset_16k", "fma"],
        # Feature front-end MUST match the models the app ships (melspectrogram + embedding).
        "feature_data_files": {},
        "target_accuracy": 0.7,
        "target_recall": 0.5,
        "max_negative_weight": 1500,
        "steps": 12000,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phrase", default="Hey Aria")
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--out", default=str(HERE / "output"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(args.phrase, args.samples, out_dir)
    config_path = out_dir / "hey_aria.yaml"

    try:
        import yaml  # type: ignore
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    except Exception:
        config_path.write_text(json.dumps(config, indent=2))
    print(f"[train] wrote config -> {config_path}")

    # Preflight: openWakeWord must be importable.
    try:
        import openwakeword  # noqa: F401
    except Exception as error:  # pragma: no cover - environment dependent
        print(f"[train] openwakeword not importable ({error}).", file=sys.stderr)
        print("[train] `pip install -r requirements.txt`, then re-run. If the automatic trainer's API "
              "differs in your version, open notebooks/automatic_model_training.ipynb and feed it "
              f"{config_path}.", file=sys.stderr)
        return 2

    # Drive the documented config-based trainer. Kept as a subprocess so a version whose module path
    # differs fails loudly with an actionable message rather than a stack trace mid-import.
    cmd = [sys.executable, "-m", "openwakeword.train", "--training_config", str(config_path)]
    print(f"[train] running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[train] automatic trainer exited non-zero. Fall back to the official notebook with the "
              f"config at {config_path} (it is version-portable).", file=sys.stderr)
        return result.returncode

    model = out_dir / "hey_aria.onnx"
    if model.exists():
        print(f"[train] DONE -> {model}")
        print(f"[train] Next: read {out_dir/'hey_aria_metrics.json'} for the ROC, pick a threshold, then")
        print("[train]   cp output/hey_aria.onnx ../assets/wakeword/wakeword.onnx && (cd .. && npx expo run:android)")
        return 0
    print("[train] trainer finished but output/hey_aria.onnx was not found — check trainer logs.", file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
