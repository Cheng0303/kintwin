#!/usr/bin/env python3
"""Run curriculum training from a JSON config.

Usage:
  python kintwin/run_train_from_json.py
  python kintwin/run_train_from_json.py --config kintwin/train_tuning.json --profile baseline
  python kintwin/run_train_from_json.py --profile smoke_test --dry_run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch kintwin/train.py from JSON config")
    parser.add_argument("--config", type=str, default="kintwin/train_tuning.json")
    parser.add_argument("--profile", type=str, default="")
    parser.add_argument("--dry_run", action="store_true", help="Print command only, do not execute")
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_profile(cfg: Dict[str, Any], profile_arg: str) -> Dict[str, Any]:
    profiles = cfg.get("profiles", {})
    if not profiles:
        raise ValueError("Config must contain a non-empty 'profiles' object")

    profile_name = profile_arg or cfg.get("default_profile", "")
    if not profile_name:
        raise ValueError("No profile selected. Set --profile or 'default_profile' in JSON")
    if profile_name not in profiles:
        raise ValueError(f"Profile '{profile_name}' not found. Available: {sorted(profiles.keys())}")

    common = cfg.get("common", {})
    selected = dict(common)
    selected.update(profiles[profile_name])
    selected["_profile_name"] = profile_name
    return selected


def _build_cmd(repo_root: Path, params: Dict[str, Any]) -> list[str]:
    train_script = repo_root / "kintwin" / "train.py"

    # Map JSON keys to CLI flags in kintwin/train.py.
    keys = [
        "hdf5_dir",
        "xml_path",
        "npz_dir",
        "save_dir",
        "init_model",
        "n_envs",
        "episode_length",
        "seed",
        "device",
        "racket_mass_scale",
        "racket_body_name",
        "balance_steps",
        "track_steps",
        "racket_steps",
        "control_penalty",
        "vel_penalty",
        "alive_bonus",
        "root_height_w",
        "com_w",
        "qpos_track_w",
        "qvel_track_w",
        "root_track_w",
        "wrist_track_w",
        "racket_tip_w",
        "racket_orient_w",
        "racket_tip_err_scale",
    ]

    cmd = [sys.executable, str(train_script)]
    for k in keys:
        if k not in params:
            continue
        cmd.extend([f"--{k}", str(params[k])])
    return cmd


def main() -> None:
    args = parse_args()

    # Script lives at humenv/kintwin/run_train_from_json.py -> repo root is humenv
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = repo_root / cfg_path

    cfg = _load_json(cfg_path)
    params = _resolve_profile(cfg, args.profile)

    profile_name = params.pop("_profile_name")
    print(f"Using profile: {profile_name}")
    print(f"Config file: {cfg_path}")

    cmd = _build_cmd(repo_root, params)
    print("Command:")
    print(" ".join(cmd))

    if args.dry_run:
        return

    subprocess.run(cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()
