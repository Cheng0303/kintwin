#!/usr/bin/env python3
"""Curriculum PPO training for badminton humanoid tracking.

This script trains a policy in 3 stages:
1) balance
2) kinematic tracking
3) racket integration (if racket bodies/sites exist in the model)

Data source:
- HDF5 clips under --hdf5_dir, each with ep_0/qpos and ep_0/qvel.

Example:
    python kintwin/train.py \
      --hdf5_dir kintwin/humenv_amass \
      --save_dir kintwin/models_curriculum \
      --balance_steps 1000000 \
      --track_steps 4000000 \
      --racket_steps 4000000
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import h5py
import humenv
import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from humenv.env import HumEnv


@dataclass
class RewardWeights:
    # Shared
    control_penalty: float = 0.01
    vel_penalty: float = 0.002
    alive_bonus: float = 0.2

    # Balance terms
    root_height_w: float = 0.35
    com_w: float = 0.35

    # Tracking terms
    qpos_track_w: float = 0.32
    qvel_track_w: float = 0.18
    root_track_w: float = 0.20
    wrist_track_w: float = 0.20

    # Racket terms (optional if racket is not present)
    racket_tip_w: float = 0.20
    racket_orient_w: float = 0.20
    racket_tip_err_scale: float = 12.0


class CurriculumBadmintonEnv(gym.Env):
    """A thin wrapper over HumEnv with curriculum rewards using HDF5 references."""

    metadata = {"render_modes": []}
    _printed_racket_info: bool = False

    @staticmethod
    def _append_startup_check_line(line: str) -> None:
        log_path = os.environ.get("KINTWIN_STARTUP_CHECK_LOG", "").strip()
        if not log_path:
            return
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def __init__(
        self,
        hdf5_dir: str,
        stage: str,
        xml_path: str = "humenv/assets/robot.xml",
        npz_dir: str = "data_preparation/AMASS/datasets/NewRacket",
        episode_length: int = 300,
        track_stride: int = 1,
        seed: int = 0,
        reward_weights: Optional[Dict[str, float]] = None,
        racket_mass_scale: float = 1.0,
        racket_body_name: str = "Racket",
    ) -> None:
        super().__init__()
        self.stage = stage
        self.npz_dir = Path(npz_dir)
        self.episode_length = episode_length
        self.track_stride = track_stride
        self.rng = np.random.default_rng(seed)

        self.base_env = HumEnv(task=None, xml=xml_path, render_mode=None, state_init="Default")
        self.model = self.base_env.model
        self.data = self.base_env.data

        humenv_module_path = Path(humenv.__file__).resolve()
        xml_arg = str(self.base_env.xml)
        xml_path = Path(xml_arg)
        if xml_path.exists():
            resolved_xml_path = str(xml_path.resolve())
        else:
            pkg_xml_path = humenv_module_path.parent / xml_arg
            resolved_xml_path = str(pkg_xml_path.resolve()) if pkg_xml_path.exists() else "<xml-string-or-missing>"

        # Reference forward-kinematics buffer for target endpoint computation.
        self.ref_data = mujoco.MjData(self.model)

        self.action_space = self.base_env.action_space
        proprio = self.base_env.get_obs()["proprio"]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=proprio.shape,
            dtype=np.float64,
        )

        self.weights = RewardWeights(**(reward_weights or {}))

        self.files = sorted(Path(hdf5_dir).glob("*.hdf5"))
        if not self.files:
            raise RuntimeError(f"No .hdf5 files found under {hdf5_dir}")

        self.clip_qpos: Optional[np.ndarray] = None
        self.clip_qvel: Optional[np.ndarray] = None
        self.clip_name: str = ""
        self.clip_racket_tip: Optional[np.ndarray] = None
        self.t = 0
        self.start_idx = 0

        self.pelvis_bid = self._find_body_id(["Pelvis", "pelvis"])
        self.lwrist_bid = self._find_body_id(["L_Hand", "hand_l", "left_hand", "L_Wrist"])
        self.rwrist_bid = self._find_body_id(["R_Hand", "hand_r", "right_hand", "R_Wrist"])

        self.racket_sid = self._find_site_id(["racket_tip", "RacketTip", "racket_site"])
        self.racket_bid = self._find_body_id([racket_body_name, "racket", "Racket"])

        if not CurriculumBadmintonEnv._printed_racket_info:
            xml_msg = (
                "[xml-check]"
                f" ts={datetime.now().isoformat(timespec='seconds')}"
                f" humenv_module={humenv_module_path}"
                f" xml_arg={xml_arg}"
                f" resolved_xml={resolved_xml_path}"
            )
            print(xml_msg)
            CurriculumBadmintonEnv._append_startup_check_line(xml_msg)

            msg = (
                "[racket-check]"
                f" ts={datetime.now().isoformat(timespec='seconds')}"
                f" stage={self.stage}"
                f" racket_body_name={racket_body_name}"
                f" racket_bid={self.racket_bid}"
                f" racket_sid={self.racket_sid}"
            )
            print(msg)
            CurriculumBadmintonEnv._append_startup_check_line(msg)
            CurriculumBadmintonEnv._printed_racket_info = True

        # Optionally scale racket inertia/mass so phase-3 learns heavier swing dynamics.
        if self.racket_bid is not None and racket_mass_scale > 0 and abs(racket_mass_scale - 1.0) > 1e-8:
            self.model.body_mass[self.racket_bid] *= racket_mass_scale
            self.model.body_inertia[self.racket_bid, :] *= racket_mass_scale
            mujoco.mj_setConst(self.model, self.data)

    def _find_body_id(self, names: List[str]) -> Optional[int]:
        for n in names:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            if idx != -1:
                return idx
        return None

    def _find_site_id(self, names: List[str]) -> Optional[int]:
        for n in names:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, n)
            if idx != -1:
                return idx
        return None

    def _sample_clip(self) -> Tuple[np.ndarray, np.ndarray, str]:
        f = self.files[self.rng.integers(0, len(self.files))]
        with h5py.File(f, "r") as hf:
            qpos = hf["ep_0"]["qpos"][:]
            qvel = hf["ep_0"]["qvel"][:]
        return qpos, qvel, f.name

    def _npz_from_h5_name(self, h5_name: str) -> Optional[Path]:
        # Expected pattern: 0-NewRacket_<session>_<clip>.hdf5
        stem = Path(h5_name).stem
        prefix = "0-NewRacket_"
        if not stem.startswith(prefix):
            return None
        body = stem[len(prefix):]
        parts = body.split("_")
        if len(parts) < 3:
            return None
        session = "_".join(parts[:2])
        clip = "_".join(parts[2:])
        p = self.npz_dir / session / f"{clip}.npz"
        return p if p.exists() else None

    def _target_index(self) -> int:
        assert self.clip_qpos is not None
        return min(self.start_idx + self.t * self.track_stride, len(self.clip_qpos) - 1)

    def _set_ref_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.ref_data.qpos[:] = qpos
        self.ref_data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.ref_data)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.clip_qpos, self.clip_qvel, self.clip_name = self._sample_clip()
        self.clip_racket_tip = None
        npz_path = self._npz_from_h5_name(self.clip_name)
        if npz_path is not None:
            try:
                with np.load(npz_path) as npz:
                    if "racket_tip" in npz.files:
                        self.clip_racket_tip = np.asarray(npz["racket_tip"], dtype=np.float64)
            except Exception:
                self.clip_racket_tip = None
        max_start = max(0, len(self.clip_qpos) - self.episode_length * self.track_stride - 1)
        self.start_idx = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0
        self.t = 0

        # Start from reference pose for easier stabilization and faster curriculum convergence.
        idx0 = self._target_index()
        self.base_env.reset(options={"qpos": self.clip_qpos[idx0], "qvel": self.clip_qvel[idx0]})
        obs = self.base_env.get_obs()["proprio"]
        info = {"clip": self.clip_name, "start_idx": self.start_idx, "stage": self.stage}
        return obs, info

    def step(self, action: np.ndarray):
        assert self.clip_qpos is not None and self.clip_qvel is not None
        tidx = self._target_index()
        tqpos = self.clip_qpos[tidx]
        tqvel = self.clip_qvel[tidx]

        try:
            obs_dict, _, _, _, base_info = self.base_env.step(action)
            obs = obs_dict["proprio"]
        except ValueError as exc:
            # HumEnv raises on MuJoCo divergence (e.g., mjWARN_BADQACC).
            # Treat it as an early terminal transition instead of crashing training.
            if "UNSTABLE MUJOCO" not in str(exc):
                raise

            try:
                self.base_env.reset(options={"qpos": tqpos, "qvel": tqvel})
                obs = self.base_env.get_obs()["proprio"]
            except Exception:
                obs, _ = self.reset()

            self.t += 1
            info = {
                "clip": self.clip_name,
                "stage": self.stage,
                "unstable_mujoco": 1.0,
                "r_balance": 0.0,
                "r_track": 0.0,
                "r_racket": 0.0,
            }
            return obs, -10.0, True, False, info

        reward, terms = self._compute_reward(tqpos=tqpos, tqvel=tqvel, action=action, tidx=tidx)

        self.t += 1
        done_by_len = self.t >= self.episode_length

        # A simple fall detector: pelvis too low.
        pelvis_h = float(self.data.xpos[self.pelvis_bid, 2]) if self.pelvis_bid is not None else float(self.data.qpos[2])
        done_by_fall = pelvis_h < 0.55
        if done_by_fall:
            reward -= 5.0

        terminated = bool(done_by_len or done_by_fall)
        truncated = False
        info = {**base_info, **terms, "clip": self.clip_name, "stage": self.stage}
        return obs, float(reward), terminated, truncated, info

    def _compute_reward(
        self,
        tqpos: np.ndarray,
        tqvel: np.ndarray,
        action: np.ndarray,
        tidx: int,
    ) -> Tuple[float, Dict[str, float]]:
        w = self.weights

        # Shared regularization terms.
        control_cost = w.control_penalty * float(np.mean(np.square(action)))
        vel_cost = w.vel_penalty * float(np.mean(np.square(self.data.qvel)))
        alive = w.alive_bonus

        # Balance: root height + COM stability.
        if self.pelvis_bid is not None:
            root_h = float(self.data.xpos[self.pelvis_bid, 2])
        else:
            root_h = float(self.data.qpos[2])
        root_h_ref = float(tqpos[2])
        r_root_h = float(np.exp(-12.0 * abs(root_h - root_h_ref)))

        com = self.data.subtree_com[0]
        com_xy = com[:2]
        pelvis_xy = self.data.xpos[self.pelvis_bid, :2] if self.pelvis_bid is not None else self.data.qpos[:2]
        r_com = float(np.exp(-10.0 * np.linalg.norm(com_xy - pelvis_xy)))

        r_balance = w.root_height_w * r_root_h + w.com_w * r_com + alive - control_cost - vel_cost

        # Tracking terms.
        qpos_err = float(np.mean(np.abs(self.data.qpos - tqpos)))
        qvel_err = float(np.mean(np.abs(self.data.qvel - tqvel)))
        root_err = float(np.linalg.norm(self.data.qpos[:7] - tqpos[:7]))

        r_qpos = float(np.exp(-8.0 * qpos_err))
        r_qvel = float(np.exp(-2.5 * qvel_err))
        r_root = float(np.exp(-6.0 * root_err))

        # Wrist endpoint tracking uses FK on target qpos/qvel.
        r_wrist = 0.0
        has_wrists = self.lwrist_bid is not None and self.rwrist_bid is not None
        if has_wrists:
            self._set_ref_state(tqpos, tqvel)
            l_curr = self.data.xpos[self.lwrist_bid]
            r_curr = self.data.xpos[self.rwrist_bid]
            l_ref = self.ref_data.xpos[self.lwrist_bid]
            r_ref = self.ref_data.xpos[self.rwrist_bid]
            wrist_err = 0.5 * (np.linalg.norm(l_curr - l_ref) + np.linalg.norm(r_curr - r_ref))
            r_wrist = float(np.exp(-10.0 * wrist_err))

        r_track = (
            w.qpos_track_w * r_qpos
            + w.qvel_track_w * r_qvel
            + w.root_track_w * r_root
            + w.wrist_track_w * r_wrist
            + 0.10 * r_balance
        )

        # Racket integration (optional if model includes racket body/site).
        r_racket_tip = 0.0
        r_racket_orient = 0.0
        racket_tip_err = 0.0
        tip_scale = max(w.racket_tip_err_scale, 1e-6)
        if self.stage == "racket" and (self.clip_racket_tip is not None or self.racket_sid is not None or self.racket_bid is not None):
            self._set_ref_state(tqpos, tqvel)

            # Prefer explicit racket_tip target from converted NPZ.
            if self.clip_racket_tip is not None:
                ridx = min(tidx, len(self.clip_racket_tip) - 1)
                tip_ref = self.clip_racket_tip[ridx]
                if self.racket_sid is not None:
                    tip_curr = self.data.site_xpos[self.racket_sid]
                elif self.racket_bid is not None:
                    tip_curr = self.data.xpos[self.racket_bid]
                elif self.rwrist_bid is not None:
                    # Fallback when model has no racket object: use right wrist as proxy.
                    tip_curr = self.data.xpos[self.rwrist_bid]
                else:
                    tip_curr = None

                if tip_curr is not None:
                    tip_err = float(np.linalg.norm(tip_curr - tip_ref))
                    racket_tip_err = tip_err
                    r_racket_tip = float(np.exp(-tip_scale * tip_err))

            if r_racket_tip == 0.0 and self.racket_sid is not None:
                tip_curr = self.data.site_xpos[self.racket_sid]
                tip_ref = self.ref_data.site_xpos[self.racket_sid]
                tip_err = float(np.linalg.norm(tip_curr - tip_ref))
                racket_tip_err = tip_err
                r_racket_tip = float(np.exp(-tip_scale * tip_err))
            elif r_racket_tip == 0.0 and self.racket_bid is not None:
                tip_curr = self.data.xpos[self.racket_bid]
                tip_ref = self.ref_data.xpos[self.racket_bid]
                tip_err = float(np.linalg.norm(tip_curr - tip_ref))
                racket_tip_err = tip_err
                r_racket_tip = float(np.exp(-tip_scale * tip_err))

            if self.racket_bid is not None:
                # Compare one racket local axis (x-axis) as a lightweight orientation reward.
                xmat_curr = self.data.xmat[self.racket_bid].reshape(3, 3)
                xmat_ref = self.ref_data.xmat[self.racket_bid].reshape(3, 3)
                dot = float(np.clip(np.dot(xmat_curr[:, 0], xmat_ref[:, 0]), -1.0, 1.0))
                r_racket_orient = 0.5 * (dot + 1.0)

        r_racket = w.racket_tip_w * r_racket_tip + w.racket_orient_w * r_racket_orient + 0.10 * r_track

        if self.stage == "balance":
            reward = r_balance
        elif self.stage == "track":
            reward = r_track
        elif self.stage == "racket":
            reward = r_racket
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

        terms = {
            "r_balance": float(r_balance),
            "r_track": float(r_track),
            "r_racket": float(r_racket),
            "r_qpos": float(r_qpos),
            "r_qvel": float(r_qvel),
            "r_root": float(r_root),
            "r_wrist": float(r_wrist),
            "r_racket_tip": float(r_racket_tip),
            "r_racket_orient": float(r_racket_orient),
            "r_racket_tip_err": float(racket_tip_err),
        }
        return reward, terms


class RewardTermsCallback(BaseCallback):
    """Logs reward sub-terms from env info dict to TensorBoard.

    The callback reads keys with prefix "r_" returned in info and logs
    running means every ``log_freq`` environment steps.
    """

    def __init__(self, log_freq: int = 2048, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._buffer: Dict[str, List[float]] = {}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            for key, value in info.items():
                if key.startswith("r_"):
                    self._buffer.setdefault(key, []).append(float(value))

        if self.num_timesteps % self.log_freq == 0 and self._buffer:
            for key, values in self._buffer.items():
                self.logger.record(f"reward_terms/{key}", float(np.mean(values)))
            self._buffer.clear()

        return True


def build_vec_env(
    hdf5_dir: str,
    stage: str,
    xml_path: str,
    npz_dir: str,
    n_envs: int,
    episode_length: int,
    seed: int,
    reward_weights: Dict[str, float],
    racket_mass_scale: float,
    racket_body_name: str,
) -> DummyVecEnv:
    def _make(rank: int):
        def _thunk():
            return CurriculumBadmintonEnv(
                hdf5_dir=hdf5_dir,
                stage=stage,
                xml_path=xml_path,
                npz_dir=npz_dir,
                episode_length=episode_length,
                seed=seed + rank,
                reward_weights=reward_weights,
                racket_mass_scale=racket_mass_scale,
                racket_body_name=racket_body_name,
            )

        return _thunk

    return DummyVecEnv([_make(i) for i in range(n_envs)])


def train_stage(
    model: Optional[PPO],
    stage: str,
    hdf5_dir: str,
    xml_path: str,
    npz_dir: str,
    save_dir: Path,
    timesteps: int,
    n_envs: int,
    episode_length: int,
    seed: int,
    device: str,
    reward_weights: Dict[str, float],
    racket_mass_scale: float,
    racket_body_name: str,
    init_model: str = "",
) -> PPO:
    vec_env = build_vec_env(
        hdf5_dir,
        stage,
        xml_path,
        npz_dir,
        n_envs,
        episode_length,
        seed,
        reward_weights,
        racket_mass_scale,
        racket_body_name,
    )

    if model is None:
        if init_model:
            model = PPO.load(init_model, env=vec_env, device=device)
            model.verbose = 1
            model.tensorboard_log = str(save_dir / "tb")
        else:
            # Base PPO config that is usually stable for muscle-control tasks.
            model = PPO(
                policy="MlpPolicy",
                env=vec_env,
                verbose=1,
                tensorboard_log=str(save_dir / "tb"),
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=256,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                device=device,
                seed=seed,
            )
    else:
        model.set_env(vec_env)

    ckpt = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 1),
        save_path=str(save_dir / stage),
        name_prefix=f"ppo_{stage}",
    )
    reward_terms_cb = RewardTermsCallback(log_freq=max(10_000 // n_envs, 1))
    callback = CallbackList([ckpt, reward_terms_cb])

    model.learn(total_timesteps=timesteps, callback=callback, reset_num_timesteps=False)
    model.save(str(save_dir / f"ppo_{stage}_final"))
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curriculum PPO training for badminton humanoid.")
    parser.add_argument("--hdf5_dir", type=str, default="kintwin/humenv_amass")
    parser.add_argument("--xml_path", type=str, default="humenv/assets/robot.xml")
    parser.add_argument("--npz_dir", type=str, default="data_preparation/AMASS/datasets/NewRacket")
    parser.add_argument("--save_dir", type=str, default="kintwin/models_curriculum")
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--episode_length", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--racket_mass_scale", type=float, default=1.0)
    parser.add_argument("--racket_body_name", type=str, default="Racket")
    parser.add_argument("--init_model", type=str, default="", help="Path to a .zip PPO model to resume training from")

    # Reward weights (tunable from CLI/JSON launcher)
    parser.add_argument("--control_penalty", type=float, default=0.01)
    parser.add_argument("--vel_penalty", type=float, default=0.002)
    parser.add_argument("--alive_bonus", type=float, default=0.2)
    parser.add_argument("--root_height_w", type=float, default=0.35)
    parser.add_argument("--com_w", type=float, default=0.35)
    parser.add_argument("--qpos_track_w", type=float, default=0.32)
    parser.add_argument("--qvel_track_w", type=float, default=0.18)
    parser.add_argument("--root_track_w", type=float, default=0.20)
    parser.add_argument("--wrist_track_w", type=float, default=0.20)
    parser.add_argument("--racket_tip_w", type=float, default=0.20)
    parser.add_argument("--racket_orient_w", type=float, default=0.20)
    parser.add_argument("--racket_tip_err_scale", type=float, default=12.0)

    # Curriculum stages (set 0 to skip a stage)
    parser.add_argument("--balance_steps", type=int, default=1_000_000)
    parser.add_argument("--track_steps", type=int, default=4_000_000)
    parser.add_argument("--racket_steps", type=int, default=4_000_000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KINTWIN_STARTUP_CHECK_LOG"] = str(save_dir / "startup_checks.log")

    reward_weights = {
        "control_penalty": args.control_penalty,
        "vel_penalty": args.vel_penalty,
        "alive_bonus": args.alive_bonus,
        "root_height_w": args.root_height_w,
        "com_w": args.com_w,
        "qpos_track_w": args.qpos_track_w,
        "qvel_track_w": args.qvel_track_w,
        "root_track_w": args.root_track_w,
        "wrist_track_w": args.wrist_track_w,
        "racket_tip_w": args.racket_tip_w,
        "racket_orient_w": args.racket_orient_w,
        "racket_tip_err_scale": args.racket_tip_err_scale,
    }

    stages = [
        ("balance", args.balance_steps),
        ("track", args.track_steps),
        ("racket", args.racket_steps),
    ]

    model: Optional[PPO] = None
    init_model_path = args.init_model.strip()
    if init_model_path and not Path(init_model_path).exists():
        raise FileNotFoundError(f"init model not found: {init_model_path}")

    for stage, steps in stages:
        if steps <= 0:
            continue
        print(f"\n=== Training stage: {stage} ({steps} steps) ===")
        model = train_stage(
            model=model,
            stage=stage,
            hdf5_dir=args.hdf5_dir,
            xml_path=args.xml_path,
            npz_dir=args.npz_dir,
            save_dir=save_dir,
            timesteps=steps,
            n_envs=args.n_envs,
            episode_length=args.episode_length,
            seed=args.seed,
            device=args.device,
            reward_weights=reward_weights,
            racket_mass_scale=args.racket_mass_scale,
            racket_body_name=args.racket_body_name,
            init_model=init_model_path,
        )
        # Only use init_model for the first executed stage.
        init_model_path = ""

    if model is None:
        raise ValueError("All stage timesteps are 0; nothing to train.")

    model.save(str(save_dir / "ppo_curriculum_final"))
    print("Training finished. Final model saved to:", save_dir / "ppo_curriculum_final")


if __name__ == "__main__":
    main()
