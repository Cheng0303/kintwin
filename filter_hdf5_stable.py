#!/usr/bin/env python3
"""Filter HDF5 clips to keep stable, upright segments.

Example:
  python kintwin/filter_hdf5_stable.py \
    --in_dir kintwin/humenv_amass \
    --out_dir kintwin/humenv_amass_stable \
    --min_pelvis_z 0.7 --min_len 240 --use_fk
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import mujoco
    from humenv.env import HumEnv
except Exception:
    mujoco = None
    HumEnv = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter HDF5 clips to stable segments.")
    p.add_argument("--in_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--min_pelvis_z", type=float, default=0.7)
    p.add_argument("--min_knee_z", type=float, default=0.06)
    p.add_argument("--min_hand_z", type=float, default=0.03)
    p.add_argument("--min_len", type=int, default=240)
    p.add_argument(
        "--prefer_start_segment",
        action="store_true",
        help="Prefer the first stable segment near the clip start.",
    )
    p.add_argument(
        "--start_max_idx",
        type=int,
        default=60,
        help="Max start index for the preferred start segment.",
    )
    p.add_argument("--use_fk", action="store_true", help="Use MuJoCo FK to filter knees/hands.")
    p.add_argument("--xml_path", type=str, default="humenv/assets/robot.xml")
    return p.parse_args()


def _find_body_id(model: "mujoco.MjModel", names: list[str]) -> Optional[int]:
    for name in names:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if idx != -1:
            return idx
    return None


def _longest_true_segment(mask: np.ndarray) -> tuple[int, int]:
    best_start = 0
    best_len = 0
    cur_start = 0
    cur_len = 0
    for i, ok in enumerate(mask):
        if ok:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
            cur_len = 0
    if cur_len > best_len:
        best_len = cur_len
        best_start = cur_start
    return best_start, best_start + best_len


def _first_true_segment(mask: np.ndarray, max_start: int, min_len: int) -> Optional[tuple[int, int]]:
    cur_start = 0
    cur_len = 0
    for i, ok in enumerate(mask):
        if ok:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len >= min_len and cur_start <= max_start:
                return cur_start, cur_start + cur_len
            cur_len = 0
    if cur_len >= min_len and cur_start <= max_start:
        return cur_start, cur_start + cur_len
    return None


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*.hdf5"))
    if not files:
        raise RuntimeError(f"No .hdf5 files found under {in_dir}")

    env = None
    pelvis_bid = None
    lknee_bid = None
    rknee_bid = None
    lhand_bid = None
    rhand_bid = None

    if args.use_fk:
        if HumEnv is None or mujoco is None:
            raise RuntimeError("MuJoCo/HumEnv not available; rerun without --use_fk.")
        xml_path = Path(args.xml_path)
        if not xml_path.is_absolute():
            xml_path = REPO_ROOT / xml_path
        env = HumEnv(task=None, xml=str(xml_path), render_mode=None, state_init="Default")
        pelvis_bid = _find_body_id(env.model, ["Pelvis", "pelvis"])
        lknee_bid = _find_body_id(env.model, ["L_Knee"])
        rknee_bid = _find_body_id(env.model, ["R_Knee"])
        lhand_bid = _find_body_id(env.model, ["L_Hand", "L_Wrist"])
        rhand_bid = _find_body_id(env.model, ["R_Hand", "R_Wrist"])

    kept = 0
    skipped = 0
    for f in files:
        with h5py.File(f, "r") as hf:
            ep0 = hf["ep_0"]
            qpos = ep0["qpos"][:]
            qvel = ep0["qvel"][:]
            ep0_attrs = dict(ep0.attrs.items())
            root_z = qpos[:, 2]

        if not args.use_fk:
            mask = root_z >= args.min_pelvis_z
        else:
            mask = np.ones(len(qpos), dtype=bool)
            for i in range(len(qpos)):
                env.set_physics(qpos=qpos[i], qvel=qvel[i])
                pelvis_z = float(env.data.xpos[pelvis_bid, 2]) if pelvis_bid is not None else float(qpos[i, 2])
                if pelvis_z < args.min_pelvis_z:
                    mask[i] = False
                    continue
                if lknee_bid is not None and float(env.data.xpos[lknee_bid, 2]) < args.min_knee_z:
                    mask[i] = False
                    continue
                if rknee_bid is not None and float(env.data.xpos[rknee_bid, 2]) < args.min_knee_z:
                    mask[i] = False
                    continue
                if lhand_bid is not None and float(env.data.xpos[lhand_bid, 2]) < args.min_hand_z:
                    mask[i] = False
                    continue
                if rhand_bid is not None and float(env.data.xpos[rhand_bid, 2]) < args.min_hand_z:
                    mask[i] = False
                    continue

        if args.prefer_start_segment:
            start_end = _first_true_segment(mask, args.start_max_idx, args.min_len)
        else:
            start_end = None

        if start_end is None:
            start, end = _longest_true_segment(mask)
        else:
            start, end = start_end
        if end - start < args.min_len:
            skipped += 1
            continue

        out_path = out_dir / f.name
        with h5py.File(out_path, "w") as hf_out:
            hf_out.attrs["num_episodes"] = 1
            ep = hf_out.create_group("ep_0")
            for key, value in ep0_attrs.items():
                ep.attrs[key] = value
            ep.attrs["filtered_start"] = int(start)
            ep.attrs["filtered_end"] = int(end)
            ep.create_dataset("qpos", data=qpos[start:end], compression="gzip")
            ep.create_dataset("qvel", data=qvel[start:end], compression="gzip")

        kept += 1

    print(f"Filtered clips saved to: {out_dir}")
    print(f"Kept: {kept} | Skipped: {skipped} | Total: {len(files)}")


if __name__ == "__main__":
    main()
