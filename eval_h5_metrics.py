#!/usr/bin/env python3
"""Evaluate converted H5 trajectories with retarget-like metrics.

Example:
  python kintwin/eval_h5_metrics.py \
    --pred_h5 kintwin/humenv_amass/0-NewRacket_241217_1_1_00_01_0.hdf5 \
        --ref_h5  kintwin/humenv_amass/0-NewRacket_241217_1_1_00_01_0.hdf5

    python kintwin/eval_h5_metrics.py \
        --pred_h5 kintwin/humenv_amass/0-NewRacket_241217_1_1_00_01_0.hdf5 \
        --ref_npz ../processed_kintwin_data/241217_1/1_00_01.npz
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import mujoco
import numpy as np

from humenv.env import HumEnv


DEFAULT_TRACK_BODIES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "L_Knee",
    "R_Knee",
    "L_Ankle",
    "R_Ankle",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Hand",
    "R_Hand",
]

BONE_PAIRS = [
    ("Pelvis", "L_Hip"),
    ("Pelvis", "R_Hip"),
    ("L_Hip", "L_Knee"),
    ("R_Hip", "R_Knee"),
    ("L_Knee", "L_Ankle"),
    ("R_Knee", "R_Ankle"),
    ("L_Shoulder", "L_Elbow"),
    ("R_Shoulder", "R_Elbow"),
    ("L_Elbow", "L_Hand"),
    ("R_Elbow", "R_Hand"),
]

FOOT_BODIES = ["L_Ankle", "R_Ankle"]

# Mapping from source states joint index (processed_kintwin_data npz) to HumEnv body names.
NPZ_STATES_BODY_MAP = [
    ("Pelvis", 0),
    ("L_Hip", 1),
    ("R_Hip", 2),
    ("L_Knee", 4),
    ("R_Knee", 5),
    ("L_Ankle", 7),
    ("R_Ankle", 8),
    ("L_Shoulder", 16),
    ("R_Shoulder", 17),
    ("L_Elbow", 18),
    ("R_Elbow", 19),
    ("L_Hand", 20),
    ("R_Hand", 21),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate H5 conversion quality with retarget-like metrics")
    p.add_argument("--pred_h5", type=str, required=True)
    p.add_argument("--ref_h5", type=str, default="", help="Reference H5 (post-conversion vs post-conversion)")
    p.add_argument("--ref_npz", type=str, default="", help="Reference npz with states (pre-conversion vs post-conversion)")
    p.add_argument("--ref_npz_key", type=str, default="states", help="Key in ref npz for source joints")
    p.add_argument("--pck_thresholds_mm", type=str, default="50,100,150")
    p.add_argument("--dt", type=float, default=1 / 30, help="Frame interval in seconds")
    p.add_argument("--max_frames", type=int, default=None, help="Optional max frame count")
    p.add_argument("--root_align", action="store_true", help="Also report root-aligned MPJPE/PCK")
    p.add_argument("--foot_contact_offset_m", type=float, default=0.03, help="Contact threshold offset over min ref z")
    p.add_argument("--csv_out", type=str, default="", help="Optional csv output")
    p.add_argument("--out_json", type=str, default="")
    return p.parse_args()


def load_h5_qpos(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return f["ep_0"]["qpos"][:]


def load_npz_states(path: Path, key: str) -> np.ndarray:
    with np.load(path) as npz:
        if key not in npz.files:
            raise KeyError(f"Key '{key}' not found in {path}")
        arr = np.asarray(npz[key], dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected [T, J, >=3] array in npz key '{key}', got {arr.shape}")
    return arr


def get_body_ids(model: mujoco.MjModel, names: List[str]) -> Tuple[List[str], List[int]]:
    kept_names: List[str] = []
    ids: List[int] = []
    for n in names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        if bid != -1:
            kept_names.append(n)
            ids.append(bid)
    if not ids:
        raise RuntimeError("No requested body names found in model")
    return kept_names, ids


def fk_positions(env: HumEnv, qpos_seq: np.ndarray, body_ids: List[int]) -> np.ndarray:
    out = np.zeros((len(qpos_seq), len(body_ids), 3), dtype=np.float64)
    qvel_zero = np.zeros(env.model.nv, dtype=np.float64)
    for i, qpos in enumerate(qpos_seq):
        env.set_physics(qpos=qpos, qvel=qvel_zero)
        out[i] = env.data.xpos[body_ids]
    return out


def root_align_positions(pred_xyz: np.ndarray, ref_xyz: np.ndarray, root_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    pred_rel = pred_xyz - pred_xyz[:, root_idx : root_idx + 1, :]
    ref_rel = ref_xyz - ref_xyz[:, root_idx : root_idx + 1, :]
    return pred_rel, ref_rel


def similarity_transform_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Align src to dst with a similarity transform using Umeyama method.

    src, dst: [J, 3]
    """
    if src.shape != dst.shape:
        raise ValueError(f"Shape mismatch: {src.shape} vs {dst.shape}")

    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean

    var_src = np.sum(src_c * src_c) / max(src.shape[0], 1)
    if var_src < 1e-12:
        return src.copy()

    cov = (dst_c.T @ src_c) / max(src.shape[0], 1)
    u, s, vt = np.linalg.svd(cov)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = u @ vt

    scale = float(np.sum(s) / var_src)
    t = dst_mean - scale * (r @ src_mean)
    aligned = (scale * (r @ src.T)).T + t
    return aligned


def procrustes_align_sequence(pred_xyz: np.ndarray, ref_xyz: np.ndarray) -> np.ndarray:
    """Per-frame similarity alignment from pred to ref, returns aligned pred."""
    aligned = np.zeros_like(pred_xyz)
    for i in range(pred_xyz.shape[0]):
        aligned[i] = similarity_transform_points(pred_xyz[i], ref_xyz[i])
    return aligned


def foot_skating(
    pred_xyz: np.ndarray,
    ref_xyz: np.ndarray,
    body_names: List[str],
    dt: float,
    contact_offset_m: float,
) -> float:
    name_to_col = {name: i for i, name in enumerate(body_names)}
    vals: List[float] = []

    for foot_name in FOOT_BODIES:
        if foot_name not in name_to_col:
            continue
        c = name_to_col[foot_name]
        pred = pred_xyz[:, c, :]
        ref = ref_xyz[:, c, :]

        ref_z = ref[:, 2]
        z_th = float(np.min(ref_z) + contact_offset_m)
        contact = ref_z <= z_th
        if np.sum(contact) < 2:
            continue

        vel = np.linalg.norm(np.diff(pred, axis=0) / dt, axis=1)
        contact_vel = vel[contact[:-1]]
        if contact_vel.size > 0:
            vals.append(float(np.mean(contact_vel)))

    if not vals:
        return 0.0
    return float(np.mean(vals))


def bone_length_mae(pred_xyz: np.ndarray, ref_xyz: np.ndarray, body_names: List[str]) -> float:
    name_to_col = {name: i for i, name in enumerate(body_names)}
    errs = []
    for a, b in BONE_PAIRS:
        if a not in name_to_col or b not in name_to_col:
            continue
        ca = name_to_col[a]
        cb = name_to_col[b]
        pred_len = np.linalg.norm(pred_xyz[:, ca] - pred_xyz[:, cb], axis=-1)
        ref_len = np.linalg.norm(ref_xyz[:, ca] - ref_xyz[:, cb], axis=-1)
        errs.append(np.abs(pred_len - ref_len))
    if not errs:
        return 0.0
    return float(np.mean(np.concatenate(errs)))


def compute_metrics(
    pred_xyz: np.ndarray,
    ref_xyz: np.ndarray,
    body_names: List[str],
    pck_th_mm: List[float],
    dt: float,
    root_align: bool,
    contact_offset_m: float,
) -> Dict[str, float]:
    # pred/ref: [T, J, 3]
    err = np.linalg.norm(pred_xyz - ref_xyz, axis=-1)  # [T, J], meters
    mpjpe_m = float(np.mean(err))

    pred_rel, ref_rel = root_align_positions(pred_xyz, ref_xyz, root_idx=0)
    err_rel = np.linalg.norm(pred_rel - ref_rel, axis=-1)
    r_mpjpe_m = float(np.mean(err_rel))

    pred_pa = procrustes_align_sequence(pred_xyz, ref_xyz)
    err_pa = np.linalg.norm(pred_pa - ref_xyz, axis=-1)
    pa_mpjpe_m = float(np.mean(err_pa))

    if pred_xyz.shape[0] >= 2:
        pred_vel = np.diff(pred_xyz, axis=0) / dt
        ref_vel = np.diff(ref_xyz, axis=0) / dt
        vel_err_mps = float(np.mean(np.linalg.norm(pred_vel - ref_vel, axis=-1)))
    else:
        vel_err_mps = 0.0

    bone_mae_m = bone_length_mae(pred_xyz, ref_xyz, body_names)
    skating_mps = foot_skating(pred_xyz, ref_xyz, body_names, dt=dt, contact_offset_m=contact_offset_m)
    per_joint_mm = np.mean(err, axis=0) * 1000.0

    metrics: Dict[str, float] = {
        "mpjpe_mm": mpjpe_m * 1000.0,
        "root_mpjpe_mm": r_mpjpe_m * 1000.0,
        "pa_mpjpe_mm": pa_mpjpe_m * 1000.0,
        "velocity_error_mps": vel_err_mps,
        "bone_length_mae_mm": bone_mae_m * 1000.0,
        "foot_skating_mps": skating_mps,
        "num_frames": float(pred_xyz.shape[0]),
        "num_joints": float(pred_xyz.shape[1]),
    }

    for th in pck_th_mm:
        hit = (err * 1000.0) <= th
        metrics[f"pck@{int(th)}mm"] = float(np.mean(hit))
        if root_align:
            hit_rel = (err_rel * 1000.0) <= th
            metrics[f"root_pck@{int(th)}mm"] = float(np.mean(hit_rel))
        hit_pa = (err_pa * 1000.0) <= th
        metrics[f"pa_pck@{int(th)}mm"] = float(np.mean(hit_pa))

    for i, name in enumerate(body_names):
        metrics[f"joint_{name}_mm"] = float(per_joint_mm[i])

    return metrics


def write_csv(path: Path, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def xyz_from_npz_states(states: np.ndarray, body_names: List[str]) -> np.ndarray:
    name_to_src = {name: src_idx for name, src_idx in NPZ_STATES_BODY_MAP}
    valid_names = [n for n in body_names if n in name_to_src]
    if not valid_names:
        raise RuntimeError("No overlapping joints between model bodies and NPZ_STATES_BODY_MAP")

    src_ids = [name_to_src[n] for n in valid_names]
    max_needed = max(src_ids)
    if states.shape[1] <= max_needed:
        raise ValueError(
            f"ref npz states has only {states.shape[1]} joints, but mapping requires index {max_needed}"
        )
    xyz = states[:, src_ids, :3]
    return xyz.astype(np.float64), valid_names


def main() -> None:
    args = parse_args()
    pred_h5 = Path(args.pred_h5)

    if bool(args.ref_h5) == bool(args.ref_npz):
        raise ValueError("Specify exactly one reference: --ref_h5 or --ref_npz")

    pred_qpos = load_h5_qpos(pred_h5)

    env = HumEnv(task=None, render_mode=None, state_init="Default")
    body_names, body_ids = get_body_ids(env.model, DEFAULT_TRACK_BODIES)

    if args.ref_h5:
        ref_qpos = load_h5_qpos(Path(args.ref_h5))
        T = min(len(pred_qpos), len(ref_qpos))
        if args.max_frames is not None:
            T = min(T, args.max_frames)
        pred_qpos = pred_qpos[:T]
        ref_qpos = ref_qpos[:T]

        pred_xyz = fk_positions(env, pred_qpos, body_ids)
        ref_xyz = fk_positions(env, ref_qpos, body_ids)
        eval_body_names = body_names
    else:
        states = load_npz_states(Path(args.ref_npz), args.ref_npz_key)
        T = min(len(pred_qpos), len(states))
        if args.max_frames is not None:
            T = min(T, args.max_frames)
        pred_qpos = pred_qpos[:T]
        states = states[:T]

        pred_xyz_all = fk_positions(env, pred_qpos, body_ids)
        ref_xyz, eval_body_names = xyz_from_npz_states(states, body_names)
        keep_cols = [i for i, n in enumerate(body_names) if n in set(eval_body_names)]
        pred_xyz = pred_xyz_all[:, keep_cols, :]

        mode_tag = "h5_vs_npz"
        print(f"Mode: {mode_tag}, using {len(eval_body_names)} mapped joints")

    pck_th = [float(x) for x in args.pck_thresholds_mm.split(",") if x.strip()]
    metrics = compute_metrics(
        pred_xyz,
        ref_xyz,
        eval_body_names,
        pck_th,
        dt=args.dt,
        root_align=args.root_align,
        contact_offset_m=args.foot_contact_offset_m,
    )

    print("Evaluation summary")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args.csv_out:
        out_csv = Path(args.csv_out)
        write_csv(out_csv, metrics)
        print(f"Saved metrics to {out_csv}")

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {out}")


if __name__ == "__main__":
    main()
