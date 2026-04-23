#!/usr/bin/env python3
"""Batch evaluate models on a fixed clip with optional MP4 rendering.

This script automates:
1) rollout generation with rollout_policy_to_h5.py
2) basic stability metrics extraction from generated HDF5
3) optional MP4 rendering with play_hdf5.py

Example:
  python kintwin/auto_eval_fixed_clip.py \
    --models_glob "kintwin/models_recommended_v5_9_balance_recover_continue/racket/*.zip" \
    --fixed_clip 0-NewRacket_250111_1_2_01_00_0.hdf5 \
    --steps 240 --render_mp4
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto evaluate fixed-clip rollouts for multiple models")
    p.add_argument("--models_glob", type=str, required=True, help="Glob for model zip files")
    p.add_argument("--fixed_clip", type=str, required=True, help="Fixed clip name in HDF5 dataset")
    p.add_argument("--out_dir", type=str, default="kintwin/eval_outputs/auto_eval")
    p.add_argument("--steps", type=int, default=240)
    p.add_argument("--stage", type=str, default="track", choices=["balance", "track", "racket"])
    p.add_argument("--render_mp4", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--hdf5_dir", type=str, default="kintwin/humenv_amass")
    p.add_argument("--npz_dir", type=str, default="data_preparation/AMASS/datasets/NewRacket")
    p.add_argument("--xml_path", type=str, default="humenv/assets/robot.xml")
    p.add_argument("--min_pelvis_z", type=float, default=0.55)
    p.add_argument("--min_knee_z", type=float, default=0.06)
    p.add_argument("--min_hand_z", type=float, default=0.03)
    p.add_argument("--max_sample_factor", type=int, default=20)
    return p.parse_args()


def _resolve_path(p: str, expect_dir: bool = False) -> Path:
    path = Path(p)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(REPO_ROOT / path)
        candidates.append(SCRIPT_DIR / path)
    for c in candidates:
        if expect_dir and c.is_dir():
            return c.resolve()
        if (not expect_dir) and c.is_file():
            return c.resolve()
    return path


def _collect_models(models_glob: str) -> List[Path]:
    pattern = Path(models_glob)
    candidates: List[Path] = []
    if pattern.is_absolute():
        candidates = sorted(Path("/").glob(str(pattern).lstrip("/")))
    else:
        candidates.extend(sorted(REPO_ROOT.glob(models_glob)))
        candidates.extend(sorted(SCRIPT_DIR.glob(models_glob)))
        candidates.extend(sorted(Path.cwd().glob(models_glob)))

    unique: Dict[str, Path] = {}
    for c in candidates:
        if c.is_file() and c.suffix == ".zip":
            unique[str(c.resolve())] = c.resolve()
    return [unique[k] for k in sorted(unique.keys())]


def _safe_name(model_path: Path) -> str:
    try:
        rel = model_path.resolve().relative_to(REPO_ROOT)
        s = str(rel)
    except Exception:
        s = str(model_path.resolve())
    return s.replace("/", "__").replace(".zip", "")


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _metrics_from_h5(h5_path: Path) -> Dict[str, Any]:
    with h5py.File(h5_path, "r") as f:
        ep = f["ep_0"]
        qpos = ep["qpos"][:]
        reward = ep["reward"][:]

    root = qpos[:, 2]
    return {
        "frames": int(len(root)),
        "root_min": float(np.min(root)),
        "root_mean": float(np.mean(root)),
        "root_max": float(np.max(root)),
        "reward_mean": float(np.mean(reward)),
    }


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hdf5_dir = _resolve_path(args.hdf5_dir, expect_dir=True)
    npz_dir = _resolve_path(args.npz_dir, expect_dir=True)
    xml_path = _resolve_path(args.xml_path, expect_dir=False)

    if not hdf5_dir.is_dir():
        raise FileNotFoundError(f"HDF5 directory not found: {args.hdf5_dir}")
    if not npz_dir.is_dir():
        raise FileNotFoundError(f"NPZ directory not found: {args.npz_dir}")
    if not xml_path.is_file():
        raise FileNotFoundError(f"XML file not found: {args.xml_path}")

    models = _collect_models(args.models_glob)
    if not models:
        raise FileNotFoundError(f"No model zip found for glob: {args.models_glob}")

    rows: List[Dict[str, Any]] = []

    for model_path in models:
        tag = _safe_name(model_path)
        out_h5 = out_dir / f"{tag}.hdf5"
        out_mp4 = out_dir / f"{tag}.mp4"

        row: Dict[str, Any] = {
            "model": str(model_path),
            "fixed_clip": args.fixed_clip,
            "stage": args.stage,
            "h5": str(out_h5),
            "mp4": str(out_mp4),
            "status": "ok",
            "error": "",
        }

        try:
            rollout_cmd = [
                sys.executable,
                str(SCRIPT_DIR / "rollout_policy_to_h5.py"),
                "--model_path",
                str(model_path),
                "--out_h5",
                str(out_h5),
                "--hdf5_dir",
                str(hdf5_dir),
                "--npz_dir",
                str(npz_dir),
                "--xml_path",
                str(xml_path),
                "--stage",
                args.stage,
                "--fixed_clip",
                args.fixed_clip,
                "--single_episode_only",
                "--allow_short",
                "--steps",
                str(args.steps),
                "--min_pelvis_z",
                str(args.min_pelvis_z),
                "--min_knee_z",
                str(args.min_knee_z),
                "--min_hand_z",
                str(args.min_hand_z),
                "--max_sample_factor",
                str(args.max_sample_factor),
            ]
            if args.deterministic:
                rollout_cmd.append("--deterministic")

            _run(rollout_cmd)

            metrics = _metrics_from_h5(out_h5)
            row.update(metrics)

            if args.render_mp4:
                render_cmd = [
                    sys.executable,
                    str(SCRIPT_DIR / "play_hdf5.py"),
                    "--h5_file",
                    str(out_h5),
                    "--output",
                    str(out_mp4),
                ]
                _run(render_cmd)

        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
            row.setdefault("frames", 0)
            row.setdefault("root_min", np.nan)
            row.setdefault("root_mean", np.nan)
            row.setdefault("root_max", np.nan)
            row.setdefault("reward_mean", np.nan)

        rows.append(row)

    csv_path = out_dir / "summary.csv"
    json_path = out_dir / "summary.json"

    fieldnames = [
        "model",
        "fixed_clip",
        "stage",
        "status",
        "frames",
        "root_min",
        "root_mean",
        "root_max",
        "reward_mean",
        "h5",
        "mp4",
        "error",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Done. Models evaluated: {len(rows)}")
    print(f"Summary CSV: {csv_path}")
    print(f"Summary JSON: {json_path}")


if __name__ == "__main__":
    main()
