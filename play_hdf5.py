import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np # 🌟 新增 numpy
from scipy.spatial.transform import Rotation as R # 🌟 新增 Scipy 的旋轉庫

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None

if not os.environ.get("DISPLAY") and not os.environ.get("MUJOCO_GL"):
    os.environ["MUJOCO_GL"] = "egl"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
from humenv.env import HumEnv

parser = argparse.ArgumentParser(description="Play one HDF5 motion in MuJoCo viewer")
parser.add_argument(
    "--h5_file",
    type=str,
    default="data_preparation/one_test/0-NewRacket_241217_1_1_00_01_1.hdf5",
)
parser.add_argument("--fps", type=float, default=30.0)
parser.add_argument(
    "--output",
    type=str,
    default="",
    help="Optional mp4 output path. If set, frames are rendered offscreen and written to video.",
)
parser.add_argument(
    "--save_h5",
    type=str,
    default="",
    help="Optional output HDF5 path to save transformed qpos/qvel for later reuse.",
)
parser.add_argument(
    "--apply_playback_rotation",
    action="store_true",
    help="Apply the playback-time root rotation correction (disable when data is already rotated).",
)
parser.add_argument(
    "--floor_clearance",
    type=float,
    default=0.01,
    help="Minimum allowed foot/body height above ground after any playback-time transform.",
)
parser.add_argument(
    "--no_floor_align",
    action="store_true",
    help="Disable automatic vertical lift used to keep feet above the floor.",
)
args = parser.parse_args()

h5_file = Path(args.h5_file)
fps = args.fps

if not h5_file.exists():
    raise FileNotFoundError(f"HDF5 file not found: {h5_file}")

env = HumEnv(render_mode=None)
dt = 1.0 / fps

print(f"Loading {h5_file}...")

with h5py.File(h5_file, "r") as hf:
    num_episodes = int(hf.attrs.get("num_episodes", 1))
    ep0 = hf["ep_0"]
    ep0_attrs = dict(ep0.attrs.items())
    episode_data = {k: ep0[k][:] for k in ep0.keys()}

qpos = episode_data["qpos"]
qvel = episode_data["qvel"]

print(f"Loaded {len(qpos)} frames")


def _candidate_floor_body_ids(model: mujoco.MjModel) -> list[int]:
    ids: list[int] = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        lname = (name or "").lower()
        if any(k in lname for k in ("foot", "toe", "ankle", "heel")):
            ids.append(i)
    return ids

if args.apply_playback_rotation:
    print("Applying playback-time root rotation correction...")
    rot_z_90 = R.from_euler("x", -90, degrees=True)

    for i in range(len(qpos)):
        qpos[i, 0:3] = rot_z_90.apply(qpos[i, 0:3])

        orig_quat = qpos[i, 3:7]
        scipy_quat = [orig_quat[1], orig_quat[2], orig_quat[3], orig_quat[0]]

        new_rot = rot_z_90 * R.from_quat(scipy_quat)
        new_scipy_quat = new_rot.as_quat()
        qpos[i, 3:7] = [new_scipy_quat[3], new_scipy_quat[0], new_scipy_quat[1], new_scipy_quat[2]]

        if len(qvel) > 0:
            qvel[i, 0:3] = rot_z_90.apply(qvel[i, 0:3])
            qvel[i, 3:6] = rot_z_90.apply(qvel[i, 3:6])

if not args.no_floor_align:
    body_ids = _candidate_floor_body_ids(env.model)
    if not body_ids:
        body_ids = list(range(1, env.model.nbody))

    min_z = np.inf
    for i in range(len(qpos)):
        env.set_physics(qpos=qpos[i], qvel=qvel[i])
        body_z = env.data.xpos[body_ids, 2]
        frame_min = float(np.min(body_z))
        if frame_min < min_z:
            min_z = frame_min

    if np.isfinite(min_z) and min_z < args.floor_clearance:
        lift = args.floor_clearance - min_z
        qpos[:, 2] += lift
        print(f"Applied floor lift: +{lift:.4f} m (min body z was {min_z:.4f})")

if args.save_h5:
    save_h5 = Path(args.save_h5)
    save_h5.parent.mkdir(parents=True, exist_ok=True)
    episode_data["qpos"] = qpos
    episode_data["qvel"] = qvel
    with h5py.File(save_h5, "w") as hf_out:
        hf_out.attrs["num_episodes"] = num_episodes
        grp = hf_out.create_group("ep_0")
        for key, value in ep0_attrs.items():
            grp.attrs[key] = value
        for key, value in episode_data.items():
            grp.create_dataset(key, data=value, compression="gzip")
    print(f"Saved transformed h5 to {save_h5}")

if args.save_h5 and not args.output and not os.environ.get("DISPLAY"):
    print("No DISPLAY found; finished after saving transformed HDF5.")
    sys.exit(0)

headless = bool(args.output) or not os.environ.get("DISPLAY")

if headless and not args.output:
    raise RuntimeError("No DISPLAY found. Re-run with --output out.mp4, or use xvfb-run.")

if args.output and imageio is None:
    raise RuntimeError("imageio is not installed, so mp4 output is unavailable.")

if args.output:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing video to {output_path}...")
    writer = imageio.get_writer(output_path, fps=fps)
    try:
        renderer = mujoco.Renderer(env.model, width=1280, height=720)
        for i in range(len(qpos)):
            env.set_physics(qpos=qpos[i], qvel=qvel[i])
            renderer.update_scene(env.data, camera=env.camera)
            writer.append_data(renderer.render())
            if i % 30 == 0 or i == len(qpos) - 1:
                print(f"Rendered {i + 1}/{len(qpos)} frames", flush=True)
    finally:
        writer.close()
    print("Done.")
else:
    print("Opening interactive viewer...", flush=True)
    import mujoco.viewer

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            for i in range(len(qpos)):
                env.set_physics(qpos=qpos[i], qvel=qvel[i])
                viewer.sync()
                time.sleep(dt)
                if i % 30 == 0 or i == len(qpos) - 1:
                    print(f"Played {i + 1}/{len(qpos)} frames", flush=True)
                if not viewer.is_running():
                    break