#!/usr/bin/env python3
"""Analyze reference foot contacts and posture from HDF5 clips."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import h5py
import mujoco
import numpy as np


def _get_body_id(model: mujoco.MjModel, names: List[str]) -> int | None:
    for name in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            return bid
    return None


def _lin_vel_from_body(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> np.ndarray:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    return jacp @ data.qvel


def _summ_update(stats: Dict[str, float], key: str, values: np.ndarray) -> None:
    if values.size == 0:
        return
    stats[f"{key}_min"] = float(np.min(values))
    stats[f"{key}_mean"] = float(np.mean(values))
    stats[f"{key}_max"] = float(np.max(values))


def analyze(hdf5_dir: Path, xml_path: Path, stride: int, vel_thresh: float, max_files: int) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    pelvis_bid = _get_body_id(model, ["Pelvis", "pelvis"])
    lfoot_bid = _get_body_id(model, ["L_Foot", "left_foot", "l_foot", "LeftFoot", "L_Ankle"])
    rfoot_bid = _get_body_id(model, ["R_Foot", "right_foot", "r_foot", "RightFoot", "R_Ankle"])
    lknee_bid = _get_body_id(model, ["L_Knee"])
    rknee_bid = _get_body_id(model, ["R_Knee"])

    if pelvis_bid is None:
        raise RuntimeError("Pelvis body not found in model.")

    h5_files = sorted(hdf5_dir.glob("*.hdf5"))
    if max_files > 0:
        h5_files = h5_files[:max_files]

    total_frames = 0
    contact_frames = 0
    contact_fast_frames = 0

    pelvis_zs: List[float] = []
    foot_min_zs: List[float] = []
    knee_min_zs: List[float] = []
    ref_clearances: List[float] = []
    foot_vels: List[float] = []
    ref_contact_counts: List[float] = []

    for fpath in h5_files:
        with h5py.File(fpath, "r") as hf:
            qpos = hf["ep_0"]["qpos"][:]
            qvel = hf["ep_0"]["qvel"][:]

        for idx in range(0, len(qpos), stride):
            data.qpos[:] = qpos[idx]
            data.qvel[:] = qvel[idx]
            mujoco.mj_forward(model, data)

            pelvis_z = float(data.xpos[pelvis_bid, 2])
            pelvis_zs.append(pelvis_z)

            foot_zs = []
            foot_vel_norms = []
            ref_contact = 0.0

            if lfoot_bid is not None:
                lz = float(data.xpos[lfoot_bid, 2])
                foot_zs.append(lz)
                lv = _lin_vel_from_body(model, data, lfoot_bid)
                foot_vel_norms.append(float(np.linalg.norm(lv)))
                if lz < 0.15:
                    ref_contact += 1.0

            if rfoot_bid is not None:
                rz = float(data.xpos[rfoot_bid, 2])
                foot_zs.append(rz)
                rv = _lin_vel_from_body(model, data, rfoot_bid)
                foot_vel_norms.append(float(np.linalg.norm(rv)))
                if rz < 0.15:
                    ref_contact += 1.0

            ref_contact_counts.append(ref_contact)

            if foot_zs:
                foot_min = float(np.min(foot_zs))
                foot_min_zs.append(foot_min)
                ref_clearances.append(pelvis_z - foot_min)

            if foot_vel_norms:
                foot_vels.extend(foot_vel_norms)

            knee_zs = []
            if lknee_bid is not None:
                knee_zs.append(float(data.xpos[lknee_bid, 2]))
            if rknee_bid is not None:
                knee_zs.append(float(data.xpos[rknee_bid, 2]))
            if knee_zs:
                knee_min_zs.append(float(np.min(knee_zs)))

            if ref_contact >= 1.0:
                contact_frames += 1
                if any(v > vel_thresh for v in foot_vel_norms):
                    contact_fast_frames += 1

            total_frames += 1

    stats: Dict[str, float] = {
        "total_frames": float(total_frames),
        "contact_frames": float(contact_frames),
        "contact_fast_frames": float(contact_fast_frames),
    }
    if total_frames > 0:
        stats["contact_ratio"] = float(contact_frames / total_frames)
        stats["contact_fast_ratio"] = float(contact_fast_frames / total_frames)

    _summ_update(stats, "pelvis_z", np.asarray(pelvis_zs))
    _summ_update(stats, "foot_min_z", np.asarray(foot_min_zs))
    _summ_update(stats, "knee_min_z", np.asarray(knee_min_zs))
    _summ_update(stats, "ref_clearance", np.asarray(ref_clearances))
    _summ_update(stats, "foot_vel", np.asarray(foot_vels))
    _summ_update(stats, "ref_contact_count", np.asarray(ref_contact_counts))

    print("[ref-scan] files:", len(h5_files))
    for key in sorted(stats.keys()):
        print(f"{key}: {stats[key]:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan reference foot contacts in HDF5 clips")
    parser.add_argument("--hdf5_dir", type=str, required=True)
    parser.add_argument("--xml_path", type=str, default="humenv/assets/robot.xml")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--vel_thresh", type=float, default=1.0)
    parser.add_argument("--max_files", type=int, default=0, help="0 means all files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hdf5_dir = Path(args.hdf5_dir)
    xml_path = Path(args.xml_path)
    if not hdf5_dir.is_dir():
        raise FileNotFoundError(f"HDF5 dir not found: {hdf5_dir}")
    if not xml_path.is_file():
        raise FileNotFoundError(f"XML not found: {xml_path}")
    analyze(hdf5_dir, xml_path, args.stride, args.vel_thresh, args.max_files)


if __name__ == "__main__":
    main()
