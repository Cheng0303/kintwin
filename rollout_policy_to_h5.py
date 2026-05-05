#!/usr/bin/env python3
"""Roll out a trained PPO policy and save trajectory to HDF5.

Example:
  python kintwin/rollout_policy_to_h5.py \
    --model_path kintwin/models_recommended_v5_7_continue/ppo_curriculum_final.zip \
    --out_h5 kintwin/eval_outputs/policy_rollout.hdf5 \
    --steps 900
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import h5py
import numpy as np
from stable_baselines3 import PPO

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train import CurriculumBadmintonEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Roll out PPO policy and export qpos/qvel HDF5")
    p.add_argument("--model_path", type=str, required=True, help="Path to PPO .zip model")
    p.add_argument("--out_h5", type=str, required=True, help="Output HDF5 path")
    p.add_argument("--hdf5_dir", type=str, default="kintwin/humenv_amass")
    p.add_argument("--xml_path", type=str, default="humenv/assets/robot.xml")
    p.add_argument("--npz_dir", type=str, default="data_preparation/AMASS/datasets/NewRacket")
    p.add_argument("--stage", type=str, default="racket", choices=["balance", "track", "racket"])
    p.add_argument("--fixed_clip", type=str, default="", help="Force rollout to sample this clip name")
    p.add_argument("--fixed_start_idx", type=int, default=-1, help="If >=0, force clip start index")
    p.add_argument("--single_episode_only", action="store_true", help="Do not stitch multiple episodes")
    p.add_argument("--allow_short", action="store_true", help="Allow saving fewer than --steps frames")
    p.add_argument("--steps", type=int, default=900, help="Total rollout steps to save")
    p.add_argument("--min_pelvis_z", type=float, default=0.55, help="Skip frames below this pelvis height")
    p.add_argument("--min_knee_z", type=float, default=0.06, help="Skip frames if knee height is below this")
    p.add_argument("--min_hand_z", type=float, default=0.03, help="Skip frames if hand height is below this")
    p.add_argument("--max_sample_factor", type=int, default=20, help="Max simulated steps = steps * factor")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    p.add_argument("--episode_length", type=int, default=300)
    p.add_argument("--racket_mass_scale", type=float, default=1.0)
    p.add_argument("--racket_body_name", type=str, default="Racket")
    p.add_argument("--fall_pelvis_h_th", type=float, default=0.55)
    p.add_argument("--fall_head_margin", type=float, default=0.20)
    p.add_argument("--fall_penalty", type=float, default=20.0)
    p.add_argument("--upright_tilt_cos", type=float, default=0.86)
    p.add_argument("--foot_height_margin", type=float, default=-1.0)
    p.add_argument("--foot_height_penalty", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    def _resolve_path(p: str, expect_dir: bool = False) -> Path:
        path = Path(p)
        candidates = [path]
        if not path.is_absolute():
            candidates.append(REPO_ROOT / path)
            candidates.append(SCRIPT_DIR / path)
            if not expect_dir:
                # Common convenience fallbacks when running inside kintwin/.
                if path.name == "robot.xml":
                    candidates.append(REPO_ROOT / "humenv" / "assets" / "robot.xml")
                if path.name == "humenv_amass":
                    candidates.append(REPO_ROOT / "kintwin" / "humenv_amass")
                if path.name == "NewRacket":
                    candidates.append(REPO_ROOT / "data_preparation" / "AMASS" / "datasets" / "NewRacket")
        for c in candidates:
            if expect_dir and c.is_dir():
                return c.resolve()
            if (not expect_dir) and c.is_file():
                return c.resolve()
        return path

    model_path = Path(args.model_path)
    if not model_path.exists():
        # Friendly fallback: if user passes only a filename, search common model dirs.
        if model_path.name == str(model_path):
            repo_root = SCRIPT_DIR.parent
            candidates = list(repo_root.glob(f"kintwin/models_*/{model_path.name}"))
            if candidates:
                model_path = max(candidates, key=lambda p: p.stat().st_mtime)
                print(f"[model-resolve] using latest candidate: {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}. "
            "Use an absolute path or a path relative to the current working directory."
        )

    xml_path = _resolve_path(args.xml_path, expect_dir=False)
    if not xml_path.is_file():
        raise FileNotFoundError(
            f"XML file not found: {args.xml_path}. "
            "When XML path is invalid, MuJoCo may raise an XML parse error."
        )

    hdf5_dir = _resolve_path(args.hdf5_dir, expect_dir=True)
    if not hdf5_dir.is_dir():
        raise FileNotFoundError(f"HDF5 directory not found: {args.hdf5_dir}")

    npz_dir = _resolve_path(args.npz_dir, expect_dir=True)
    if not npz_dir.is_dir():
        raise FileNotFoundError(f"NPZ directory not found: {args.npz_dir}")

    out_h5 = Path(args.out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)

    env = CurriculumBadmintonEnv(
        hdf5_dir=str(hdf5_dir),
        stage=args.stage,
        xml_path=str(xml_path),
        npz_dir=str(npz_dir),
        episode_length=args.episode_length,
        seed=args.seed,
        racket_mass_scale=args.racket_mass_scale,
        racket_body_name=args.racket_body_name,
        fall_pelvis_h_th=args.fall_pelvis_h_th,
        fall_head_margin=args.fall_head_margin,
        fall_penalty=args.fall_penalty,
        upright_tilt_cos=args.upright_tilt_cos,
        foot_height_margin=args.foot_height_margin,
        foot_height_penalty=args.foot_height_penalty,
    )

    def _bid(names: list[str]) -> int | None:
        for n in names:
            b = env._find_body_id([n])
            if b is not None:
                return b
        return None

    lknee_bid = _bid(["L_Knee"])
    rknee_bid = _bid(["R_Knee"])
    lhand_bid = _bid(["L_Hand", "L_Wrist"])
    rhand_bid = _bid(["R_Hand", "R_Wrist"])

    model = PPO.load(str(model_path), device=args.device)

    def _reset_constrained(seed: int | None = None):
        max_tries = 5000 if args.fixed_clip else 1
        obs_local = None
        info_local = None
        for i in range(max_tries):
            obs_local, info_local = env.reset(seed=seed if i == 0 else None)
            clip_name = str(info_local.get("clip", ""))
            if args.fixed_clip and clip_name != args.fixed_clip:
                continue

            if args.fixed_start_idx >= 0:
                env.start_idx = int(args.fixed_start_idx)
                env.t = 0
                idx0 = env._target_index()
                env.base_env.reset(options={"qpos": env.clip_qpos[idx0], "qvel": env.clip_qvel[idx0]})
                obs_local = env.base_env.get_obs()["proprio"]
                info_local = {
                    "clip": env.clip_name,
                    "start_idx": env.start_idx,
                    "stage": env.stage,
                }
            return obs_local, info_local

        raise RuntimeError(
            f"Could not sample fixed clip '{args.fixed_clip}' after {max_tries} resets. "
            "Check --fixed_clip value and --hdf5_dir contents."
        )

    obs, info = _reset_constrained(seed=args.seed)
    qpos_seq: List[np.ndarray] = []
    qvel_seq: List[np.ndarray] = []
    action_seq: List[np.ndarray] = []
    actuator_force_seq: List[np.ndarray] = []
    qfrc_actuator_seq: List[np.ndarray] = []
    reward_seq: List[float] = []
    done_seq: List[int] = []
    clip_seq: List[str] = []

    def _append_frame(
        qpos: np.ndarray,
        qvel: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: int,
        clip: str,
    ) -> None:
        qpos_seq.append(np.asarray(qpos, dtype=np.float64).copy())
        qvel_seq.append(np.asarray(qvel, dtype=np.float64).copy())
        action_seq.append(np.asarray(action, dtype=np.float64).copy())
        actuator_force_seq.append(np.asarray(env.data.actuator_force, dtype=np.float64).copy())
        qfrc_actuator_seq.append(np.asarray(env.data.qfrc_actuator, dtype=np.float64).copy())
        reward_seq.append(float(reward))
        done_seq.append(int(done))
        clip_seq.append(str(clip))

    sampled_steps = 0
    max_sampled_steps = max(args.steps * args.max_sample_factor, args.steps)
    while len(qpos_seq) < args.steps and sampled_steps < max_sampled_steps:
        sampled_steps += 1
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        pelvis_z = float(info["qpos"][2])
        knee_low = False
        hand_low = False
        if lknee_bid is not None:
            knee_low = knee_low or float(env.data.xpos[lknee_bid, 2]) < args.min_knee_z
        if rknee_bid is not None:
            knee_low = knee_low or float(env.data.xpos[rknee_bid, 2]) < args.min_knee_z
        if lhand_bid is not None:
            hand_low = hand_low or float(env.data.xpos[lhand_bid, 2]) < args.min_hand_z
        if rhand_bid is not None:
            hand_low = hand_low or float(env.data.xpos[rhand_bid, 2]) < args.min_hand_z

        fell = bool(terminated or truncated) or pelvis_z < args.min_pelvis_z or knee_low or hand_low
        if fell:
            if args.single_episode_only and args.allow_short:
                _append_frame(
                    info["qpos"],
                    info["qvel"],
                    action,
                    reward,
                    1,
                    info.get("clip", ""),
                )
                break
            if args.single_episode_only:
                break
            obs, info = _reset_constrained()
            continue

        _append_frame(
            info["qpos"],
            info["qvel"],
            action,
            reward,
            0,
            info.get("clip", ""),
        )

    if len(qpos_seq) < args.steps and not (args.allow_short or args.single_episode_only):
        raise RuntimeError(
            f"Only collected {len(qpos_seq)} valid frames out of requested {args.steps}. "
            f"Try reducing --min_pelvis_z or increasing --max_sample_factor."
        )
    if len(qpos_seq) == 0:
        raise RuntimeError("No valid rollout frames collected.")
    if len(qpos_seq) < args.steps:
        print(f"Warning: saved short rollout ({len(qpos_seq)} < requested {args.steps})")

    qpos = np.stack(qpos_seq, axis=0)
    qvel = np.stack(qvel_seq, axis=0)
    action_arr = np.stack(action_seq, axis=0)
    actuator_force_arr = np.stack(actuator_force_seq, axis=0)
    qfrc_actuator_arr = np.stack(qfrc_actuator_seq, axis=0)
    rewards = np.asarray(reward_seq, dtype=np.float32)
    dones = np.asarray(done_seq, dtype=np.int8)
    clips = np.asarray(clip_seq, dtype="S128")

    with h5py.File(out_h5, "w") as hf:
        hf.attrs["num_episodes"] = 1
        ep0 = hf.create_group("ep_0")
        ep0.attrs["length"] = int(len(qpos))
        ep0.attrs["source_model"] = str(model_path)
        ep0.attrs["stage"] = args.stage
        ep0.attrs["fixed_clip"] = args.fixed_clip
        ep0.attrs["fixed_start_idx"] = int(args.fixed_start_idx)
        ep0.attrs["single_episode_only"] = int(bool(args.single_episode_only))
        ep0.create_dataset("qpos", data=qpos, compression="gzip")
        ep0.create_dataset("qvel", data=qvel, compression="gzip")
        ep0.create_dataset("action", data=action_arr, compression="gzip")
        ep0.create_dataset("actuator_force", data=actuator_force_arr, compression="gzip")
        ep0.create_dataset("qfrc_actuator", data=qfrc_actuator_arr, compression="gzip")
        ep0.create_dataset("reward", data=rewards, compression="gzip")
        ep0.create_dataset("done", data=dones, compression="gzip")
        ep0.create_dataset("clip", data=clips, compression="gzip")

    print(f"Saved rollout HDF5: {out_h5}")
    print(f"Frames: {len(qpos)}")


if __name__ == "__main__":
    main()
