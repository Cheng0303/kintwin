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
        reward_mode: str = "exp",
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
        self.reward_mode = str(reward_mode)

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
        self.future_offsets = [1, 3, 5, 10, 20]
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self.prev_prev_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self.knee_contact_frames = 0
        self.crouch_frames = 0

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

        self.lknee_bid = self._find_body_id(["L_Knee"])
        self.rknee_bid = self._find_body_id(["R_Knee"])
        self.knee_bids = [bid for bid in (self.lknee_bid, self.rknee_bid) if bid is not None]

        if self.knee_bids:
            print(
                "[KINTWIN] knee_bids:",
                [
                    (bid, mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid))
                    for bid in self.knee_bids
                ],
            )
        else:
            print("[KINTWIN][WARN] No knee bodies found; knee_ground_cost will be disabled.")

        self.lknee_geom_id = self._find_geom_id(["L_Knee"])
        self.rknee_geom_id = self._find_geom_id(["R_Knee"])
        self.knee_geom_ids = [gid for gid in (self.lknee_geom_id, self.rknee_geom_id) if gid is not None]

        if self.knee_geom_ids:
            print(
                "[KINTWIN] knee_geom_ids:",
                [
                    (gid, mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid))
                    for gid in self.knee_geom_ids
                ],
            )
        else:
            print("[KINTWIN][WARN] No knee geoms found; knee contact penalty will be disabled.")

        self.floor_gid = self._find_geom_id(["floor"])

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

        dummy_obs = self._get_obs()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=dummy_obs.shape,
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

    def _find_geom_id(self, names: List[str]) -> Optional[int]:
        for n in names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
            if gid >= 0:
                return gid
        return None

    def _site_linear_velocity(self, data: mujoco.MjData, site_id: int) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacSite(self.model, data, jacp, jacr, site_id)
        return jacp @ data.qvel

    def _body_linear_velocity(self, data: mujoco.MjData, body_id: int) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        mujoco.mj_jacBody(self.model, data, jacp, jacr, body_id)
        return jacp @ data.qvel

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

        phase = 0.0
        if self.clip_qpos is not None and len(self.clip_qpos) > 1:
            phase = float(
                (self.start_idx + self.t * self.track_stride)
                / max(len(self.clip_qpos) - 1, 1)
            )
        phase = float(np.clip(phase, 0.0, 1.0))
        phase_feat = np.array(
            [
                np.sin(2.0 * np.pi * phase),
                np.cos(2.0 * np.pi * phase),
            ],
            dtype=np.float64,
        )

        if self.clip_qpos is None or self.clip_qvel is None:
            per_offset_dim = (
                2 * self.model.nv
                + 6
                + 3
                + 15
                + 3
                + 3
                + 3
            )
            goal_dim = len(self.future_offsets) * per_offset_dim
            goal_zeros = np.zeros(goal_dim, dtype=np.float64)
            return np.concatenate(
                [proprio, np.zeros(2, dtype=np.float64), goal_zeros, self.prev_action.ravel()]
            ).astype(np.float32)

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

            root_curr = None
            R_curr = None
            if self.pelvis_bid is not None:
                root_curr = self.data.xpos[self.pelvis_bid].copy()
                R_curr = self.data.xmat[self.pelvis_bid].reshape(3, 3).copy()

            wrist_delta = np.zeros(6, dtype=np.float64)
            if R_curr is not None:
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
            racket_tip_traj_delta = np.zeros(3, dtype=np.float64)
            if R_curr is not None:
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
                    racket_tip_traj_delta = R_curr.T @ (tip_ref - tip_curr)
            racket_delta = np.clip(racket_delta, -2.0, 2.0)
            racket_tip_traj_delta = np.clip(racket_tip_traj_delta, -2.0, 2.0)

            body_traj_parts = []
            for bid in [self.pelvis_bid, self.torso_bid, self.head_bid, self.lwrist_bid, self.rwrist_bid]:
                if bid is None or R_curr is None:
                    body_traj_parts.append(np.zeros(3, dtype=np.float64))
                    continue
                delta = R_curr.T @ (self.ref_data.xpos[bid] - self.data.xpos[bid])
                body_traj_parts.append(np.clip(delta, -2.0, 2.0))
            body_traj_delta = np.concatenate(body_traj_parts) if body_traj_parts else np.zeros(15, dtype=np.float64)

            rwrist_vel_ref_local = np.zeros(3, dtype=np.float64)
            if self.rwrist_bid is not None and R_curr is not None:
                rwrist_vel_ref = self._body_linear_velocity(self.ref_data, self.rwrist_bid)
                rwrist_vel_ref_local = R_curr.T @ rwrist_vel_ref
            rwrist_vel_ref_local = np.clip(rwrist_vel_ref_local, -10.0, 10.0)

            racket_tip_vel_ref_local = np.zeros(3, dtype=np.float64)
            if R_curr is not None:
                if self.racket_sid is not None:
                    racket_tip_vel_ref = self._site_linear_velocity(self.ref_data, self.racket_sid)
                elif self.racket_bid is not None:
                    racket_tip_vel_ref = self._body_linear_velocity(self.ref_data, self.racket_bid)
                elif self.rwrist_bid is not None:
                    racket_tip_vel_ref = self._body_linear_velocity(self.ref_data, self.rwrist_bid)
                else:
                    racket_tip_vel_ref = None

                if racket_tip_vel_ref is not None:
                    racket_tip_vel_ref_local = R_curr.T @ racket_tip_vel_ref
            racket_tip_vel_ref_local = np.clip(racket_tip_vel_ref_local, -10.0, 10.0)

            goal_parts.extend(
                [
                    qpos_diff,
                    qvel_diff,
                    wrist_delta,
                    racket_delta,
                    body_traj_delta,
                    rwrist_vel_ref_local,
                    racket_tip_vel_ref_local,
                    racket_tip_traj_delta,
                ]
            )

        return np.concatenate([proprio, phase_feat, *goal_parts, self.prev_action.ravel()]).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.clip_qpos, self.clip_qvel, self.clip_name = self._sample_clip()
        max_start = max(0, len(self.clip_qpos) - self.episode_length * self.track_stride - 1)
        self.start_idx = int(self.rng.integers(0, max_start + 1)) if max_start > 0 else 0
        self.t = 0
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self.prev_prev_action = np.zeros(self.action_space.shape, dtype=np.float64)
        self.knee_contact_frames = 0
        self.crouch_frames = 0

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
                self.prev_prev_action = np.zeros(self.action_space.shape, dtype=np.float64)
                self.knee_contact_frames = 0
                self.crouch_frames = 0
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
        self.prev_prev_action = self.prev_action.copy()
        self.prev_action = np.asarray(action, dtype=np.float64)

        self.t += 1
        done_by_len = self.t >= self.episode_length

        knee_floor_contacts = float(terms.get("r_knee_floor_contacts", 0.0))
        if knee_floor_contacts > 0:
            self.knee_contact_frames += 1
        else:
            self.knee_contact_frames = max(0, self.knee_contact_frames - 1)

        knee_terminated = self.knee_contact_frames >= 3
        terms["r_knee_contact_frames"] = float(self.knee_contact_frames)
        terms["r_knee_terminated"] = float(1.0 if knee_terminated else 0.0)

        pelvis_foot_clearance = float(terms.get("r_pelvis_foot_clearance", 999.0))
        low_pose_cost = float(terms.get("r_low_pose_cost", 0.0))
        r_upright = float(terms.get("r_upright", 1.0))

        # Bad collapse: allow athletic crouch; terminate only on collapse evidence.
        # V10_13 failure mode: foot support lost (low contact) despite tracking gains.
        foot_contact_count = float(terms.get("r_foot_contact_count", 0.0))
        collapse_knee = knee_floor_contacts > 0.0
        collapse_clearance_lowpose = (
            pelvis_foot_clearance < 0.50
            and low_pose_cost > 0.010
        )
        collapse_upright_clearance = (
            r_upright < 0.04
            and pelvis_foot_clearance < 0.58
            and low_pose_cost > 0.006
        )
        collapse_foot_support = (
            foot_contact_count < 0.5
            and r_upright < 0.05
        )
        bad_collapse = (
            collapse_knee
            or collapse_clearance_lowpose
            or collapse_upright_clearance
            or collapse_foot_support
        )

        terms["r_bad_collapse_knee_contact"] = float(1.0 if collapse_knee else 0.0)
        terms["r_bad_collapse_clearance_lowpose"] = float(1.0 if collapse_clearance_lowpose else 0.0)
        terms["r_bad_collapse_upright_clearance"] = float(1.0 if collapse_upright_clearance else 0.0)
        terms["r_bad_collapse_foot_support"] = float(1.0 if collapse_foot_support else 0.0)

        # Backward-compatible aliases for existing dashboards.
        terms["r_bad_crouch_clearance"] = float(1.0 if collapse_clearance_lowpose else 0.0)
        terms["r_bad_crouch_low_pose"] = float(1.0 if collapse_clearance_lowpose else 0.0)
        terms["r_bad_crouch_upright"] = float(1.0 if collapse_upright_clearance else 0.0)

        if bad_collapse:
            self.crouch_frames += 1
        else:
            self.crouch_frames = max(0, self.crouch_frames - 1)

        collapse_terminated = self.crouch_frames >= 5
        terms["r_collapse_frames"] = float(self.crouch_frames)
        terms["r_collapse_terminated"] = float(1.0 if collapse_terminated else 0.0)
        terms["r_crouch_frames"] = float(self.crouch_frames)
        terms["r_crouch_terminated"] = float(1.0 if collapse_terminated else 0.0)

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
        terminated = bool(terminated or knee_terminated or collapse_terminated)
        truncated = False
        obs = self._get_obs()
        if knee_terminated:
            reward -= 8.0
        if collapse_terminated:
            reward -= 8.0
        info = {
            **base_info,
            **terms,
            "clip": self.clip_name,
            "stage": self.stage,
        }
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
        r_upright = 1.0 if upright else 0.0

        # Shared regularization terms.
        penalty_scale = self.upright_penalty_scale if upright else 0.5 * self.upright_penalty_scale
        control_cost = penalty_scale * w.control_penalty * float(np.mean(np.square(action)))
        action_delta_cost = 0.003 * float(np.mean(np.square(action - self.prev_action)))
        action_accel_cost = 0.002 * float(
            np.mean(np.square(action - 2.0 * self.prev_action + self.prev_prev_action))
        )
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
            - action_accel_cost
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

        qpos_mse = float(np.mean(np.square(qpos_diff[6:]))) if qpos_diff.shape[0] > 6 else 0.0
        qvel_diff = self.data.qvel - tqvel
        qvel_mse = float(np.mean(np.square(qvel_diff)))
        qvel_mse = float(np.clip(qvel_mse, 0.0, 10.0))
        root_mse = (root_pos_err ** 2) + 0.25 * (root_rot_err ** 2)

        qpos_err = joint_err
        qvel_err = float(np.mean(np.abs(qvel_diff)))
        root_err = root_pos_err + 0.5 * root_rot_err

        r_qpos = float(np.exp(-4.0 * qpos_err))
        r_qvel = float(np.exp(-2.5 * qvel_err))
        r_root = float(np.exp(-4.0 * root_err))


        # Wrist endpoint tracking uses FK on target qpos/qvel.
        r_wrist = 0.0
        wrist_err = 0.0
        wrist_mse = 0.0
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
            wrist_mse = wrist_err ** 2
            r_wrist = float(np.exp(-10.0 * wrist_err))

        r_body_pos = 0.0
        body_pos_err = 0.0
        body_pos_mse = 0.0
        if self.pelvis_bid is not None and self.track_bids:
            self._set_ref_state(tqpos, tqvel)
            root_curr = self.data.xpos[self.pelvis_bid].copy()
            root_ref = self.ref_data.xpos[self.pelvis_bid].copy()

            R_curr = self.data.xmat[self.pelvis_bid].reshape(3, 3).copy()
            R_ref = self.ref_data.xmat[self.pelvis_bid].reshape(3, 3).copy()

            errs = []
            mses = []
            for bid in self.track_bids:
                p_curr = self.data.xpos[bid].copy()
                p_ref = self.ref_data.xpos[bid].copy()

                p_curr_local = R_curr.T @ (p_curr - root_curr)
                p_ref_local = R_ref.T @ (p_ref - root_ref)

                diff = p_curr_local - p_ref_local
                errs.append(np.linalg.norm(diff))
                mses.append(float(np.sum(diff ** 2)))

            body_pos_err = float(np.mean(errs))
            body_pos_mse = float(np.mean(mses))
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
        upper_orient_mse = 0.0
        if self.orient_bids:
            self._set_ref_state(tqpos, tqvel)
            orient_rewards = []
            orient_mses = []
            for bid in self.orient_bids:
                R_curr = self.data.xmat[bid].reshape(3, 3).copy()
                R_ref = self.ref_data.xmat[bid].reshape(3, 3).copy()

                R_rel = R_curr.T @ R_ref
                cos_angle = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
                orient_rewards.append(0.5 * (cos_angle + 1.0))
                orient_mses.append(float(np.arccos(cos_angle) ** 2))

            r_upper_orient = float(np.mean(orient_rewards))
            upper_orient_mse = float(np.mean(orient_mses))

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
        racket_tip_mse = 0.0
        racket_orient_mse = 0.0
        wrist_vel_mse = 0.0
        racket_tip_vel_mse = 0.0

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
                racket_tip_mse = tip_err ** 2
                r_racket_tip = float(np.exp(-tip_scale * tip_err))

            if self.racket_bid is not None:
                xmat_curr = self.data.xmat[self.racket_bid].reshape(3, 3)
                xmat_ref = self.ref_data.xmat[self.racket_bid].reshape(3, 3)
                shaft_curr = -xmat_curr[:, 0]
                shaft_ref = -xmat_ref[:, 0]

                dot = float(np.clip(np.dot(shaft_curr, shaft_ref), -1.0, 1.0))
                r_racket_orient = 0.5 * (dot + 1.0)
                racket_orient_mse = float(np.arccos(dot) ** 2)
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

        if self.rwrist_bid is not None:
            wrist_vel_curr = self._body_linear_velocity(self.data, self.rwrist_bid)
            wrist_vel_ref = self._body_linear_velocity(self.ref_data, self.rwrist_bid)
            wrist_vel_diff = np.clip(wrist_vel_curr - wrist_vel_ref, -10.0, 10.0)
            wrist_vel_mse = float(np.mean(np.square(wrist_vel_diff)))

        if self.racket_sid is not None:
            tip_vel_curr = self._site_linear_velocity(self.data, self.racket_sid)
            tip_vel_ref = self._site_linear_velocity(self.ref_data, self.racket_sid)
            tip_vel_diff = np.clip(tip_vel_curr - tip_vel_ref, -10.0, 10.0)
            racket_tip_vel_mse = float(np.mean(np.square(tip_vel_diff)))
        elif self.racket_bid is not None:
            tip_vel_curr = self._body_linear_velocity(self.data, self.racket_bid)
            tip_vel_ref = self._body_linear_velocity(self.ref_data, self.racket_bid)
            tip_vel_diff = np.clip(tip_vel_curr - tip_vel_ref, -10.0, 10.0)
            racket_tip_vel_mse = float(np.mean(np.square(tip_vel_diff)))

        pose_core_cost_raw = (
            4.0 * body_pos_mse
            + 0.45 * upper_orient_mse
            + 0.25 * qpos_mse
            + 0.03 * qvel_mse
            + 0.90 * root_mse
            + 0.30 * wrist_mse
        )

        pose_core_cost = float(np.clip(pose_core_cost_raw, 0.0, 2.0))

        posture_cost = 0.0
        if not upright:
            posture_cost += 0.80

        if self.torso_bid is not None:
            torso_xmat = self.data.xmat[self.torso_bid].reshape(3, 3)
            torso_up_cos = float(torso_xmat[2, 2])
            posture_cost += 2.00 * max(0.0, 0.84 - torso_up_cos) ** 2

        if self.head_bid is not None and self.pelvis_bid is not None:
            head_h = float(self.data.xpos[self.head_bid, 2])
            pelvis_h = float(self.data.xpos[self.pelvis_bid, 2])
            posture_cost += 1.00 * max(0.0, 0.50 - (head_h - pelvis_h)) ** 2

        foot_ground_cost = 0.0
        if self.foot_bids:
            for foot_bid in self.foot_bids:
                foot_z = float(self.data.xpos[foot_bid, 2])
                ref_foot_z = float(self.ref_data.xpos[foot_bid, 2])

                if ref_foot_z < 0.15:
                    target_z = min(0.13, ref_foot_z + 0.05)
                    foot_ground_cost += max(0.0, foot_z - target_z) ** 2

        foot_ground_cost *= 8.0

        crouch_cost = 0.0
        pelvis_foot_clearance = 0.0

        if self.pelvis_bid is not None and self.foot_bids:
            pelvis_h_for_crouch = float(self.data.xpos[self.pelvis_bid, 2])
            foot_min_z_for_crouch = min(float(self.data.xpos[bid, 2]) for bid in self.foot_bids)
            pelvis_foot_clearance = pelvis_h_for_crouch - foot_min_z_for_crouch
            crouch_cost = 0.8 * max(0.0, 0.55 - pelvis_foot_clearance) ** 2

        knee_ground_cost = 0.0
        knee_min_z = 0.0

        if hasattr(self, "knee_bids") and self.knee_bids:
            knee_zs = [float(self.data.xpos[bid, 2]) for bid in self.knee_bids]
            knee_min_z = min(knee_zs)

            for knee_z in knee_zs:
                knee_ground_cost += max(0.0, 0.22 - knee_z) ** 2

            knee_ground_cost *= 3.0

        knee_contact_cost = 0.0
        knee_floor_contacts = 0

        if (
            hasattr(self, "knee_geom_ids")
            and self.knee_geom_ids
            and self.floor_gid is not None
        ):
            knee_set = set(self.knee_geom_ids)
            for ci in range(self.data.ncon):
                c = self.data.contact[ci]
                g1 = int(c.geom1)
                g2 = int(c.geom2)

                if (g1 in knee_set and g2 == self.floor_gid) or (
                    g2 in knee_set and g1 == self.floor_gid
                ):
                    knee_floor_contacts += 1

            if knee_floor_contacts > 0:
                knee_contact_cost = 1.0 * knee_floor_contacts

        pose_cost = (
            pose_core_cost
            + posture_cost
            + foot_ground_cost
            + crouch_cost
            + knee_ground_cost
            + knee_contact_cost
        )

        swing_vel_cost_raw = (
            0.010 * wrist_vel_mse
            + 0.004 * racket_tip_vel_mse
        )
        swing_vel_cost = float(np.clip(swing_vel_cost_raw, 0.0, 0.20))

        # Allow athletic crouch; only gate when support looks physically unhealthy.
        # Foot contact can drop slip cost because the feet are off ground, so gate it explicitly.
        support_gate = 1.0
        if pelvis_foot_clearance < 0.52 and low_pose_cost > 0.006:
            support_gate *= 0.6
        if r_upright < 0.04 and pelvis_foot_clearance < 0.58:
            support_gate *= 0.6
        if knee_floor_contacts > 0:
            support_gate *= 0.3
        if foot_contact_count < 0.5:
            support_gate *= 0.4
        elif foot_contact_count < 1.0:
            support_gate *= 0.7
        support_mult = 0.30 + 0.70 * support_gate

        r_track_mse_raw = (
            1.00
            + 0.45 * r_balance_base_norm
            - pose_cost
            - 0.20 * swing_vel_cost
        )
        r_track_mse = (
            support_mult * max(0.0, r_track_mse_raw)
            + min(0.0, r_track_mse_raw)
            - 0.20 * (1.0 - support_gate)
        )

        racket_cost_raw = (
            0.12 * racket_tip_mse
            + 0.10 * racket_orient_mse
        )
        racket_cost = float(np.clip(racket_cost_raw, 0.0, 1.5))

        r_racket_mse_raw = (
            0.65 * r_track_mse_raw
            + 0.35 * r_balance_base_norm
            + 0.15 * r_racket_task
            - racket_cost
            - 0.50 * swing_vel_cost
        )
        r_racket_mse = (
            support_mult * max(0.0, r_racket_mse_raw)
            + min(0.0, r_racket_mse_raw)
            - 0.20 * (1.0 - support_gate)
        )

        reward_mode_mse = self.reward_mode == "mse_hybrid"

        if self.stage == "balance":
            reward_before_slip = r_balance_base_norm
        elif self.stage == "track":
            reward_before_slip = r_track_mse if reward_mode_mse else r_track
        elif self.stage == "racket":
            reward_before_slip = r_racket_mse if reward_mode_mse else r_racket
        else:
            raise ValueError(f"Unknown stage: {self.stage}")

        # foot slip 直接扣 final reward，避免只藏在 balance 裡被稀釋
        reward = reward_before_slip - foot_slip_cost

        terms = {
            "r_balance": float(r_balance),
            "r_track": float(r_track),
            "r_racket": float(r_racket),
            "r_track_mse": float(r_track_mse),
            "r_track_mse_raw": float(r_track_mse_raw),
            "r_racket_mse": float(r_racket_mse),
            "r_racket_mse_raw": float(r_racket_mse_raw),
            "r_support_gate": float(support_gate),
            "r_support_mult": float(support_mult),
            "r_reward_mode_mse": 1.0 if reward_mode_mse else 0.0,
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
            "r_pose_cost": float(pose_cost),
            "r_pose_core_cost": float(pose_core_cost),
            "r_pose_core_cost_raw": float(pose_core_cost_raw),
            "r_posture_cost": float(posture_cost),
            "r_foot_ground_cost": float(foot_ground_cost),
            "r_crouch_cost": float(crouch_cost),
            "r_pelvis_foot_clearance": float(pelvis_foot_clearance),
            "r_knee_ground_cost": float(knee_ground_cost),
            "r_knee_min_z": float(knee_min_z),
            "r_knee_contact_cost": float(knee_contact_cost),
            "r_knee_floor_contacts": float(knee_floor_contacts),
            "r_swing_vel_cost": float(swing_vel_cost),
            "r_swing_vel_cost_raw": float(swing_vel_cost_raw),
            "r_qpos_mse": float(qpos_mse),
            "r_qvel_mse": float(qvel_mse),
            "r_root_mse": float(root_mse),
            "r_wrist_mse": float(wrist_mse),
            "r_wrist_vel_mse": float(wrist_vel_mse),
            "r_body_pos_mse": float(body_pos_mse),
            "r_upper_orient_mse": float(upper_orient_mse),

            "r_racket_tip": float(r_racket_tip),
            "r_racket_orient": float(r_racket_orient),
            "r_racket_tip_err": float(racket_tip_err),
            "r_racket_tip_mse": float(racket_tip_mse),
            "r_racket_orient_mse": float(racket_orient_mse),
            "r_racket_tip_vel_mse": float(racket_tip_vel_mse),
            "r_racket_cost": float(racket_cost),
            "r_racket_cost_raw": float(racket_cost_raw),
            "r_racket_pure": float(r_racket_pure),
            "r_racket_track_part": float(r_racket_track_part),
            "r_racket_balance_part": float(r_racket_balance_part),
            "r_racket_task": float(r_racket_task),

            "r_upright": float(r_upright),
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
            "r_action_accel_cost": float(action_accel_cost),
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
    reward_mode: str = "exp",
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
                reward_mode=reward_mode,
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
    reward_mode: str = "exp",
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
        reward_mode,
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
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="exp",
        choices=["exp", "mse_hybrid"],
    )

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
            reward_mode=args.reward_mode,
        )
        # Only use init_model for the first executed stage.
        init_model_path = ""

    if model is None:
        raise ValueError("All stage timesteps are 0; nothing to train.")

    model.save(str(save_dir / "ppo_curriculum_final"))
    print("Training finished. Final model saved to:", save_dir / "ppo_curriculum_final")


if __name__ == "__main__":
    main()
