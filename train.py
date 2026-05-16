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
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
        fall_pelvis_h_th: float = 0.55,
        fall_head_margin: float = 0.20,
        fall_penalty: float = 20.0,
        upright_pelvis_h_th: float = -1.0,
        upright_head_margin: float = -1.0,
        upright_tilt_cos: float = 0.86,
        foot_height_margin: float = -1.0,
        foot_height_penalty: float = 0.0,
        upright_bonus: float = 0.0,
        upright_track_scale: float = 1.0,
        upright_penalty_scale: float = 1.0,
        low_pose_margin: float = -1.0,
        low_pose_penalty: float = 0.0,
        foot_slip_penalty: float = 0.0,
        body_pos_log_path: str = "",
        body_pos_debug_every: int = 10,
    ) -> None:
        super().__init__()
        self.stage = stage
        self.episode_length = episode_length
        self.track_stride = track_stride
        self.rng = np.random.default_rng(seed)
        self.fall_pelvis_h_th = float(fall_pelvis_h_th)
        self.fall_head_margin = float(fall_head_margin)
        self.fall_penalty = float(fall_penalty)
        self.upright_pelvis_h_th = float(upright_pelvis_h_th)
        self.upright_head_margin = float(upright_head_margin)
        self.upright_tilt_cos = float(upright_tilt_cos)
        self.foot_height_margin = float(foot_height_margin)
        self.foot_height_penalty = float(foot_height_penalty)
        self.upright_bonus = float(upright_bonus)
        self.upright_track_scale = float(upright_track_scale)
        self.upright_penalty_scale = float(upright_penalty_scale)
        self.low_pose_margin = float(low_pose_margin)
        self.low_pose_penalty = float(low_pose_penalty)
        self.foot_slip_penalty = float(foot_slip_penalty)
        self.body_pos_log_path = str(body_pos_log_path).strip()
        self.body_pos_debug_every = int(body_pos_debug_every)
        

        xml_p = Path(xml_path)
        if not xml_p.is_absolute():
            cand = REPO_ROOT / xml_p
            if cand.exists():
                xml_p = cand
        if not xml_p.is_file():
            raise FileNotFoundError(
                f"XML file not found: {xml_path}. "
                "Invalid xml path can surface as MuJoCo XML parse errors."
            )

        self.base_env = HumEnv(task=None, xml=str(xml_p), render_mode=None, state_init="Default")
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
        self.future_offsets = [1, 5, 10]
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float64)

        self.weights = RewardWeights(**(reward_weights or {}))

        self.files = sorted(Path(hdf5_dir).glob("*.hdf5"))
        if not self.files:
            raise RuntimeError(f"No .hdf5 files found under {hdf5_dir}")

        self.clip_qpos: Optional[np.ndarray] = None
        self.clip_qvel: Optional[np.ndarray] = None
        self.clip_name: str = ""
        self.t = 0
        self.start_idx = 0

        self.pelvis_bid = self._find_body_id(["Pelvis", "pelvis"])
        self.head_bid = self._find_body_id(["Head", "head"])
        self.torso_bid = self._find_body_id(["Torso", "torso"])
        self.lwrist_bid = self._find_body_id(["L_Hand", "hand_l", "left_hand", "L_Wrist"])
        self.rwrist_bid = self._find_body_id(["R_Hand", "hand_r", "right_hand", "R_Wrist"])
        self.lfoot_bid = self._find_body_id(["L_Foot", "left_foot", "l_foot", "LeftFoot", "L_Ankle"])
        self.rfoot_bid = self._find_body_id(["R_Foot", "right_foot", "r_foot", "RightFoot", "R_Ankle"])
        self.foot_bids = [bid for bid in (self.lfoot_bid, self.rfoot_bid) if bid is not None]

        self.track_body_names = [
            "Pelvis",
            "Torso",
            "Spine",
            "Chest",
            "Head",
            "L_Shoulder",
            "L_Elbow",
            "L_Wrist",
            "L_Hand",
            "R_Shoulder",
            "R_Elbow",
            "R_Wrist",
            "R_Hand",
            "L_Hip",
            "L_Knee",
            "L_Ankle",
            "L_Toe",
            "R_Hip",
            "R_Knee",
            "R_Ankle",
            "R_Toe",
        ]
        self.track_bids = []
        for name in self.track_body_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self.track_bids.append(bid)

        self.orient_body_names = [
            "Pelvis",
            "Torso",
            "Chest",
            "Head",
            "L_Shoulder",
            "R_Shoulder",
        ]
        self.orient_bids = []
        for name in self.orient_body_names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self.orient_bids.append(bid)

        self.racket_sid = self._find_site_id(["racket_tip", "RacketTip", "racket_site"])
        self.racket_bid = self._find_body_id([racket_body_name, "racket", "Racket"])

        proprio_dim = int(self.base_env.get_obs()["proprio"].shape[0])
        per_future_dim = self.model.nv + self.model.nv + 6 + 3
        goal_dim = len(self.future_offsets) * per_future_dim
        prev_action_dim = int(np.prod(self.action_space.shape))
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(proprio_dim + goal_dim + prev_action_dim,),
            dtype=np.float32,
        )

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

    def _target_index(self) -> int:
        assert self.clip_qpos is not None
        return min(self.start_idx + self.t * self.track_stride, len(self.clip_qpos) - 1)

    def _set_ref_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        self.ref_data.qpos[:] = qpos
        self.ref_data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.ref_data)

    def _get_ref_racket_tip(self) -> Optional[np.ndarray]:
        if self.racket_sid is not None:
            return self.ref_data.site_xpos[self.racket_sid].copy()
        if self.racket_bid is not None:
            return self.ref_data.xpos[self.racket_bid].copy()
        if self.rwrist_bid is not None:
            return self.ref_data.xpos[self.rwrist_bid].copy()
        return None

    def _get_obs(self) -> np.ndarray:
        proprio = self.base_env.get_obs()["proprio"].astype(np.float64)

        if self.clip_qpos is None or self.clip_qvel is None:
            goal_dim = len(self.future_offsets) * (2 * self.model.nv + 6 + 3)
            goal_zeros = np.zeros(goal_dim, dtype=np.float64)
            return np.concatenate([proprio, goal_zeros, self.prev_action.ravel()]).astype(np.float32)

        goal_parts: List[np.ndarray] = []
        for off in self.future_offsets:
            ref_idx = min(
                self.start_idx + (self.t + off) * self.track_stride,
                len(self.clip_qpos) - 1,
            )
            tqpos = self.clip_qpos[ref_idx]
            tqvel = self.clip_qvel[ref_idx]

            qpos_diff = np.zeros(self.model.nv, dtype=np.float64)
            mujoco.mj_differentiatePos(
                self.model,
                qpos_diff,
                1.0,
                self.data.qpos,
                tqpos,
            )
            qpos_diff = np.clip(qpos_diff, -5.0, 5.0)
            qvel_diff = np.clip(tqvel - self.data.qvel, -20.0, 20.0)

            self._set_ref_state(tqpos, tqvel)

            wrist_delta = np.zeros(6, dtype=np.float64)
            if self.pelvis_bid is not None:
                root_curr = self.data.xpos[self.pelvis_bid].copy()
                R_curr = self.data.xmat[self.pelvis_bid].reshape(3, 3).copy()

                if self.lwrist_bid is not None:
                    wrist_delta[:3] = R_curr.T @ (
                        self.ref_data.xpos[self.lwrist_bid] - self.data.xpos[self.lwrist_bid]
                    )
                if self.rwrist_bid is not None:
                    wrist_delta[3:6] = R_curr.T @ (
                        self.ref_data.xpos[self.rwrist_bid] - self.data.xpos[self.rwrist_bid]
                    )
            wrist_delta = np.clip(wrist_delta, -2.0, 2.0)

            racket_delta = np.zeros(3, dtype=np.float64)
            if self.pelvis_bid is not None:
                R_curr = self.data.xmat[self.pelvis_bid].reshape(3, 3).copy()
                tip_curr = None
                if self.racket_sid is not None:
                    tip_curr = self.data.site_xpos[self.racket_sid].copy()
                elif self.racket_bid is not None:
                    tip_curr = self.data.xpos[self.racket_bid].copy()
                elif self.rwrist_bid is not None:
                    tip_curr = self.data.xpos[self.rwrist_bid].copy()

                tip_ref = self._get_ref_racket_tip()
                if tip_curr is not None and tip_ref is not None:
                    racket_delta = R_curr.T @ (tip_ref - tip_curr)
            racket_delta = np.clip(racket_delta, -2.0, 2.0)

            goal_parts.extend([qpos_diff, qvel_diff, wrist_delta, racket_delta])

        return np.concatenate([proprio, *goal_parts, self.prev_action.ravel()]).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.clip_qpos, self.clip_qvel, self.clip_name = self._sample_clip()
        max_start = max(0, len(self.clip_qpos) - self.episode_length * self.track_stride - 1)
        self.start_idx = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0
        self.t = 0
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float64)

        # Start from reference pose for easier stabilization and faster curriculum convergence.
        idx0 = self._target_index()
        self.base_env.reset(options={"qpos": self.clip_qpos[idx0], "qvel": self.clip_qvel[idx0]})
        obs = self._get_obs()
        info = {"clip": self.clip_name, "start_idx": self.start_idx, "stage": self.stage}
        return obs, info

    def step(self, action: np.ndarray):
        assert self.clip_qpos is not None and self.clip_qvel is not None
        target_t = self.t + 1
        tidx = min(
            self.start_idx + target_t * self.track_stride,
            len(self.clip_qpos) - 1,
        )
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
                self.prev_action = np.zeros(self.action_space.shape, dtype=np.float64)
                obs = self._get_obs()
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
        self.prev_action = np.asarray(action, dtype=np.float64)

        self.t += 1
        done_by_len = self.t >= self.episode_length

        # Head-aware fall detector: pelvis too low or head drops near pelvis level.
        pelvis_h = float(self.data.xpos[self.pelvis_bid, 2]) if self.pelvis_bid is not None else float(self.data.qpos[2])
        upper_h = pelvis_h + 0.5
        upper_margin = self.fall_head_margin
        if self.head_bid is not None:
            upper_h = float(self.data.xpos[self.head_bid, 2])
        elif self.torso_bid is not None:
            upper_h = float(self.data.xpos[self.torso_bid, 2])
            # Torso is naturally lower than head, so use a softer threshold.
            upper_margin = 0.3 * self.fall_head_margin

        done_by_fall = (pelvis_h < self.fall_pelvis_h_th) or (upper_h < pelvis_h + upper_margin)
        if done_by_fall:
            reward -= self.fall_penalty

        terminated = bool(done_by_len or done_by_fall)
        truncated = False
        obs = self._get_obs()
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

        pelvis_h = float(self.data.xpos[self.pelvis_bid, 2]) if self.pelvis_bid is not None else float(self.data.qpos[2])
        upper_h = pelvis_h + 0.5
        upper_margin = self.upright_head_margin
        if self.head_bid is not None:
            upper_h = float(self.data.xpos[self.head_bid, 2])
        elif self.torso_bid is not None:
            upper_h = float(self.data.xpos[self.torso_bid, 2])
            if upper_margin >= 0:
                upper_margin = 0.3 * upper_margin

        pelvis_ok = True if self.upright_pelvis_h_th < 0 else pelvis_h >= self.upright_pelvis_h_th
        head_ok = True if upper_margin < 0 else upper_h >= pelvis_h + upper_margin
        # tilt gating (torso local z-axis should align with world z)
        torso_tilt_ok = True
        if self.torso_bid is not None and self.upright_tilt_cos >= 0:
            torso_xmat = self.data.xmat[self.torso_bid].reshape(3, 3)
            torso_tilt_ok = float(torso_xmat[2, 2]) >= self.upright_tilt_cos
        upright = bool(pelvis_ok and head_ok and torso_tilt_ok)

        # Shared regularization terms.
        penalty_scale = self.upright_penalty_scale if upright else 0.0
        control_cost = penalty_scale * w.control_penalty * float(np.mean(np.square(action)))
        action_delta_cost = 0.003 * float(np.mean(np.square(action - self.prev_action)))
        vel_cost = penalty_scale * w.vel_penalty * float(np.mean(np.square(self.data.qvel)))
        alive = w.alive_bonus

        # Balance: root height + COM stability.
        if self.pelvis_bid is not None:
            root_h = float(self.data.xpos[self.pelvis_bid, 2])
        else:
            root_h = float(self.data.qpos[2])
        root_h_ref = float(tqpos[2])
        r_root_h = float(np.exp(-12.0 * abs(root_h - root_h_ref)))

        low_pose_err = 0.0
        low_pose_cost = 0.0
        if self.low_pose_margin >= 0 and self.low_pose_penalty > 0:
            low_pose_err = max(0.0, root_h_ref - root_h - self.low_pose_margin)
            low_pose_cost = self.low_pose_penalty * (low_pose_err ** 2)

        com = self.data.subtree_com[0]
        com_xy = com[:2]
        pelvis_xy = self.data.xpos[self.pelvis_bid, :2] if self.pelvis_bid is not None else self.data.qpos[:2]
        r_com = float(np.exp(-10.0 * np.linalg.norm(com_xy - pelvis_xy)))

        foot_over = 0.0
        foot_penalty = 0.0
        if self.foot_height_margin >= 0 and self.foot_height_penalty > 0:
            foot_zs = []
            if self.lfoot_bid is not None:
                foot_zs.append(float(self.data.xpos[self.lfoot_bid, 2]))
            if self.rfoot_bid is not None:
                foot_zs.append(float(self.data.xpos[self.rfoot_bid, 2]))
            if foot_zs:
                foot_limit = pelvis_h + self.foot_height_margin
                foot_over = max(0.0, max(foot_zs) - foot_limit)
                foot_penalty = self.foot_height_penalty * foot_over

        contact_h = 0.15
        foot_raw_slip = 0.0
        foot_contact_count = 0.0
        foot_zs: List[float] = []

        if self.foot_bids:
            for foot_bid in self.foot_bids:
                foot_z = float(self.data.xpos[foot_bid, 2])
                foot_xy_vel = self.data.cvel[foot_bid, 3:5]
                foot_zs.append(foot_z)

                if foot_z < contact_h:
                    foot_raw_slip += float(np.sum(foot_xy_vel ** 2))
                    foot_contact_count += 1.0

        foot_min_z = float(np.min(foot_zs)) if foot_zs else 0.0
        foot_mean_z = float(np.mean(foot_zs)) if foot_zs else 0.0

        foot_slip_cost = self.foot_slip_penalty * foot_raw_slip

        upright_bonus = self.upright_bonus if upright else 0.0
        r_balance_base = (
            w.root_height_w * r_root_h
            + w.com_w * r_com
            + alive
            + upright_bonus
            - control_cost
            - action_delta_cost
            - vel_cost
            - foot_penalty
            - low_pose_cost
        )
        r_balance = r_balance_base - foot_slip_cost

        # freejoint: qpos has 7 dims, qvel/differentiate result has 6 root dims
        qpos_diff = np.zeros(self.model.nv, dtype=np.float64)
        mujoco.mj_differentiatePos(
            self.model,
            qpos_diff,
            1.0,
            tqpos,
            self.data.qpos,
        )

        root_pos_err = float(np.linalg.norm(self.data.qpos[:3] - tqpos[:3]))
        root_rot_err = float(np.linalg.norm(qpos_diff[3:6]))
        joint_err = float(np.mean(np.abs(qpos_diff[6:]))) if qpos_diff.shape[0] > 6 else 0.0

        qpos_err = joint_err
        qvel_err = float(np.mean(np.abs(self.data.qvel - tqvel)))
        root_err = root_pos_err + 0.5 * root_rot_err

        r_qpos = float(np.exp(-4.0 * qpos_err))
        r_qvel = float(np.exp(-2.5 * qvel_err))
        r_root = float(np.exp(-4.0 * root_err))


        # Wrist endpoint tracking uses FK on target qpos/qvel.
        r_wrist = 0.0
        has_wrists = self.lwrist_bid is not None and self.rwrist_bid is not None
        if has_wrists:
            self._set_ref_state(tqpos, tqvel)
            l_curr = self.data.xpos[self.lwrist_bid].copy()
            r_curr = self.data.xpos[self.rwrist_bid].copy()
            l_ref = self.ref_data.xpos[self.lwrist_bid].copy()
            r_ref = self.ref_data.xpos[self.rwrist_bid].copy()

            root_curr = self.data.xpos[self.pelvis_bid].copy()
            root_ref = self.ref_data.xpos[self.pelvis_bid].copy()

            R_curr = self.data.xmat[self.pelvis_bid].reshape(3, 3).copy()
            R_ref = self.ref_data.xmat[self.pelvis_bid].reshape(3, 3).copy()

            l_curr_local = R_curr.T @ (l_curr - root_curr)
            r_curr_local = R_curr.T @ (r_curr - root_curr)

            l_ref_local = R_ref.T @ (l_ref - root_ref)
            r_ref_local = R_ref.T @ (r_ref - root_ref)

            wrist_err = 0.5 * (
                np.linalg.norm(l_curr_local - l_ref_local)
                + np.linalg.norm(r_curr_local - r_ref_local)
            )
            r_wrist = float(np.exp(-10.0 * wrist_err))

        r_body_pos = 0.0
        body_pos_err = 0.0
        if self.pelvis_bid is not None and self.track_bids:
            self._set_ref_state(tqpos, tqvel)
            root_curr = self.data.xpos[self.pelvis_bid].copy()
            root_ref = self.ref_data.xpos[self.pelvis_bid].copy()

            R_curr = self.data.xmat[self.pelvis_bid].reshape(3, 3).copy()
            R_ref = self.ref_data.xmat[self.pelvis_bid].reshape(3, 3).copy()

            errs = []
            for bid in self.track_bids:
                p_curr = self.data.xpos[bid].copy()
                p_ref = self.ref_data.xpos[bid].copy()

                p_curr_local = R_curr.T @ (p_curr - root_curr)
                p_ref_local = R_ref.T @ (p_ref - root_ref)

                errs.append(np.linalg.norm(p_curr_local - p_ref_local))

            body_pos_err = float(np.mean(errs))
            r_body_pos = float(np.exp(-3.0 * body_pos_err))

            debug_enabled = os.environ.get("KINTWIN_BODY_POS_DEBUG", "0") != "0"
            debug_every = int(
                os.environ.get(
                    "KINTWIN_BODY_POS_DEBUG_EVERY",
                    str(self.body_pos_debug_every),
                )
            )
            debug_every = max(1, debug_every)
            if debug_enabled and self.clip_qpos is not None and (self.t % debug_every == 0):
                debug_lines = []
                for offset in range(-5, 6):
                    ref_idx = int(np.clip(tidx + offset, 0, len(self.clip_qpos) - 1))
                    self._set_ref_state(self.clip_qpos[ref_idx], self.clip_qvel[ref_idx])

                    root_ref_dbg = self.ref_data.xpos[self.pelvis_bid].copy()
                    R_ref_dbg = self.ref_data.xmat[self.pelvis_bid].reshape(3, 3).copy()

                    debug_errs = []
                    for bid in self.track_bids:
                        p_curr = self.data.xpos[bid].copy()
                        p_ref = self.ref_data.xpos[bid].copy()

                        p_curr_local = R_curr.T @ (p_curr - root_curr)
                        p_ref_local = R_ref_dbg.T @ (p_ref - root_ref_dbg)
                        debug_errs.append(np.linalg.norm(p_curr_local - p_ref_local))

                    debug_err = float(np.mean(debug_errs))
                    debug_lines.append(f"{offset}:{debug_err:.4f}")

                print(
                    "[body-pos-debug]"
                    f" clip={self.clip_name}"
                    f" tidx={tidx}"
                    f" offsets={' '.join(debug_lines)}"
                )
                if self.body_pos_log_path:
                    try:
                        with open(self.body_pos_log_path, "a", encoding="utf-8") as f:
                            f.write(
                                "[body-pos-debug]"
                                f" clip={self.clip_name}"
                                f" tidx={tidx}"
                                f" offsets={' '.join(debug_lines)}\n"
                            )
                    except OSError:
                        pass

        r_upper_orient = 0.0
        if self.orient_bids:
            self._set_ref_state(tqpos, tqvel)
            orient_rewards = []
            for bid in self.orient_bids:
                R_curr = self.data.xmat[bid].reshape(3, 3).copy()
                R_ref = self.ref_data.xmat[bid].reshape(3, 3).copy()

                R_rel = R_curr.T @ R_ref
                cos_angle = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
                orient_rewards.append(0.5 * (cos_angle + 1.0))

            r_upper_orient = float(np.mean(orient_rewards))

        pose_num = (
            3.0 * r_body_pos
            + 0.6 * r_upper_orient
            + 0.25 * w.qpos_track_w * r_qpos
            + 0.5 * w.qvel_track_w * r_qvel
            + w.root_track_w * r_root
            + w.wrist_track_w * r_wrist
        )
        pose_den = (
            3.0
            + 0.6
            + 0.25 * w.qpos_track_w
            + 0.5 * w.qvel_track_w
            + w.root_track_w
            + w.wrist_track_w
        )
        r_pose = pose_num / max(pose_den, 1e-6)

        balance_den = max(
            w.root_height_w + w.com_w + w.alive_bonus + self.upright_bonus,
            1e-6,
        )

        # 用於 reward 組合的 balance，不含 foot slip，避免 foot_slip 被 normalize 吃掉
        r_balance_base_norm = float(np.clip(r_balance_base / balance_den, -1.0, 1.0))

        # 純 logging 用：含 foot slip 後的 balance
        r_balance_norm = float(np.clip(r_balance / balance_den, -1.0, 1.0))

        r_track_balance_part = 0.15 * r_balance_base_norm
        r_track = 0.85 * r_pose + r_track_balance_part

        # Racket integration (optional if model includes racket body/site).
        r_racket_tip = 0.0
        r_racket_orient = 0.0
        racket_tip_err = 0.0

        r_racket_pure = 0.0
        r_racket_task = 0.0
        r_racket_track_part = 0.0
        r_racket_balance_part = 0.0

        tip_scale = max(w.racket_tip_err_scale, 1e-6)

        if self.racket_sid is not None or self.racket_bid is not None:
            self._set_ref_state(tqpos, tqvel)

            if self.racket_sid is not None:
                tip_curr = self.data.site_xpos[self.racket_sid].copy()
                tip_ref = self.ref_data.site_xpos[self.racket_sid].copy()
            elif self.racket_bid is not None:
                tip_curr = self.data.xpos[self.racket_bid].copy()
                tip_ref = self.ref_data.xpos[self.racket_bid].copy()
            else:
                tip_curr = None
                tip_ref = None

            if tip_curr is not None and tip_ref is not None:
                # World-space racket tip tracking:
                # compare current racket tip with HDF5 FK reference racket tip directly.
                tip_err = float(np.linalg.norm(tip_curr - tip_ref))

                racket_tip_err = tip_err
                r_racket_tip = float(np.exp(-tip_scale * tip_err))

            if self.racket_bid is not None:
                xmat_curr = self.data.xmat[self.racket_bid].reshape(3, 3)
                xmat_ref = self.ref_data.xmat[self.racket_bid].reshape(3, 3)
                shaft_curr = -xmat_curr[:, 0]
                shaft_ref = -xmat_ref[:, 0]

                dot = float(np.clip(np.dot(shaft_curr, shaft_ref), -1.0, 1.0))
                r_racket_orient = 0.5 * (dot + 1.0)
        # r_racket = w.racket_tip_w * r_racket_tip + w.racket_orient_w * r_racket_orient + 0.10 * r_track
        tip_w = max(float(w.racket_tip_w), 0.0)
        orient_w = max(float(w.racket_orient_w), 0.0)

        task_den = max(tip_w + orient_w, 1e-6)

        # normalized racket task score, roughly 0~1
        r_racket_task = (
            tip_w * r_racket_tip
            + orient_w * r_racket_orient
        ) / task_den

        r_racket_track_part = 0.55 * r_pose
        r_racket_task_part = 0.30 * r_racket_task
        r_racket_balance_part = 0.15 * r_balance_base_norm

        r_racket = (
            r_racket_track_part
            + r_racket_task_part
            + r_racket_balance_part
        )

        # for logging
        r_racket_pure = r_racket_task
        if not upright:
            r_racket *= self.upright_track_scale

        if self.stage == "balance":
            reward_before_slip = r_balance_base_norm
        elif self.stage == "track":
            reward_before_slip = r_track
        elif self.stage == "racket":
            reward_before_slip = r_racket
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

        # foot slip 直接扣 final reward，避免只藏在 balance 裡被稀釋
        reward = reward_before_slip - foot_slip_cost

        terms = {
            "r_balance": float(r_balance),
            "r_track": float(r_track),
            "r_racket": float(r_racket),
            "r_balance_norm": float(r_balance_norm),
            "r_balance_base": float(r_balance_base),
            "r_balance_base_norm": float(r_balance_base_norm),
            "r_reward_before_slip": float(reward_before_slip),

            "r_qpos": float(r_qpos),
            "r_qvel": float(r_qvel),
            "r_root": float(r_root),
            "r_wrist": float(r_wrist),
            "r_body_pos": float(r_body_pos),
            "r_body_pos_err": float(body_pos_err),
            "r_upper_orient": float(r_upper_orient),

            "r_racket_tip": float(r_racket_tip),
            "r_racket_orient": float(r_racket_orient),
            "r_racket_tip_err": float(racket_tip_err),
            "r_racket_pure": float(r_racket_pure),
            "r_racket_track_part": float(r_racket_track_part),
            "r_racket_balance_part": float(r_racket_balance_part),
            "r_racket_task": float(r_racket_task),

            "r_upright": 1.0 if upright else 0.0,
            "r_foot_penalty": float(foot_penalty),
            "r_foot_raw_slip": float(foot_raw_slip),
            "r_foot_slip_cost": float(foot_slip_cost),
            "r_foot_contact_count": float(foot_contact_count),
            "r_foot_min_z": float(foot_min_z),
            "r_foot_mean_z": float(foot_mean_z),
            "r_foot_over": float(foot_over),
            "r_low_pose_err": float(low_pose_err),
            "r_low_pose_cost": float(low_pose_cost),
            "r_action_delta_cost": float(action_delta_cost),
            "r_pose": float(r_pose),
            "r_track_balance_part": float(r_track_balance_part),
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
    fall_pelvis_h_th: float,
    fall_head_margin: float,
    fall_penalty: float,
    upright_pelvis_h_th: float,
    upright_head_margin: float,
    upright_tilt_cos: float,
    foot_height_margin: float,
    foot_height_penalty: float,
    low_pose_margin: float,
    low_pose_penalty: float,
    foot_slip_penalty: float,
    upright_bonus: float,
    upright_track_scale: float,
    upright_penalty_scale: float,
    body_pos_log_path: str = "",
    body_pos_debug_every: int = 10,
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
                fall_pelvis_h_th=fall_pelvis_h_th,
                fall_head_margin=fall_head_margin,
                fall_penalty=fall_penalty,
                upright_pelvis_h_th=upright_pelvis_h_th,
                upright_head_margin=upright_head_margin,
                upright_tilt_cos=upright_tilt_cos,
                foot_height_margin=foot_height_margin,
                foot_height_penalty=foot_height_penalty,
                low_pose_margin=low_pose_margin,
                low_pose_penalty=low_pose_penalty,
                foot_slip_penalty=foot_slip_penalty,
                upright_bonus=upright_bonus,
                upright_track_scale=upright_track_scale,
                upright_penalty_scale=upright_penalty_scale,
                body_pos_log_path=body_pos_log_path,
                body_pos_debug_every=body_pos_debug_every,
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
    fall_pelvis_h_th: float,
    fall_head_margin: float,
    fall_penalty: float,
    upright_pelvis_h_th: float,
    upright_head_margin: float,
    upright_tilt_cos: float,
    foot_height_margin: float,
    foot_height_penalty: float,
    low_pose_margin: float,
    low_pose_penalty: float,
    foot_slip_penalty: float,
    upright_bonus: float,
    upright_track_scale: float,
    upright_penalty_scale: float,
    init_model: str = "",
    body_pos_log_path: str = "",
    body_pos_debug_every: int = 10,
) -> PPO:
    if not body_pos_log_path:
        body_pos_log_path = str(save_dir / f"body_pos_debug_{stage}.log")
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
        fall_pelvis_h_th,
        fall_head_margin,
        fall_penalty,
        upright_pelvis_h_th,
        upright_head_margin,
        upright_tilt_cos,
        foot_height_margin,
        foot_height_penalty,
        low_pose_margin,
        low_pose_penalty,
        foot_slip_penalty,
        upright_bonus,
        upright_track_scale,
        upright_penalty_scale,
        body_pos_log_path,
        body_pos_debug_every,
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
    parser.add_argument("--fall_pelvis_h_th", type=float, default=0.55)
    parser.add_argument("--fall_head_margin", type=float, default=0.20)
    parser.add_argument("--fall_penalty", type=float, default=20.0)
    parser.add_argument("--upright_pelvis_h_th", type=float, default=-1.0)
    parser.add_argument("--upright_head_margin", type=float, default=-1.0)
    parser.add_argument("--upright_tilt_cos", type=float, default=0.86)
    parser.add_argument("--foot_height_margin", type=float, default=-1.0)
    parser.add_argument("--foot_height_penalty", type=float, default=0.0)
    parser.add_argument("--low_pose_margin", type=float, default=-1.0)
    parser.add_argument("--low_pose_penalty", type=float, default=0.0)
    parser.add_argument("--foot_slip_penalty", type=float, default=0.0)
    parser.add_argument("--upright_bonus", type=float, default=0.0)
    parser.add_argument("--upright_track_scale", type=float, default=1.0)
    parser.add_argument("--upright_penalty_scale", type=float, default=1.0)
    parser.add_argument("--body_pos_debug_every", type=int, default=10)

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
            fall_pelvis_h_th=args.fall_pelvis_h_th,
            fall_head_margin=args.fall_head_margin,
            fall_penalty=args.fall_penalty,
            upright_pelvis_h_th=args.upright_pelvis_h_th,
            upright_head_margin=args.upright_head_margin,
            upright_tilt_cos=args.upright_tilt_cos,
            foot_height_margin=args.foot_height_margin,
            foot_height_penalty=args.foot_height_penalty,
            low_pose_margin=args.low_pose_margin,
            low_pose_penalty=args.low_pose_penalty,
            foot_slip_penalty=args.foot_slip_penalty,
            upright_bonus=args.upright_bonus,
            upright_track_scale=args.upright_track_scale,
            upright_penalty_scale=args.upright_penalty_scale,
            init_model=init_model_path,
            body_pos_debug_every=args.body_pos_debug_every,
        )
        # Only use init_model for the first executed stage.
        init_model_path = ""

    if model is None:
        raise ValueError("All stage timesteps are 0; nothing to train.")

    model.save(str(save_dir / "ppo_curriculum_final"))
    print("Training finished. Final model saved to:", save_dir / "ppo_curriculum_final")


if __name__ == "__main__":
    main()
