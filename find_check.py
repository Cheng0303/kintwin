from pathlib import Path
import re
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

TB_DIR = Path("kintwin/models_recommended_v6_7ll/tb")
CKPT_DIR = Path("kintwin/models_recommended_v6_7ll")

TAGS = {
    "balance": "reward_terms/r_balance",
    "foot_over": "reward_terms/r_foot_over",
    "foot_penalty": "reward_terms/r_foot_penalty",
    "foot_slip_cost": "reward_terms/r_foot_slip_cost",
    "low_pose_cost": "reward_terms/r_low_pose_cost",
    "low_pose_err": "reward_terms/r_low_pose_err",
    "qpos": "reward_terms/r_qpos",
    "qvel": "reward_terms/r_qvel",
    "racket": "reward_terms/r_racket",
    "racket_orient": "reward_terms/r_racket_orient",
    "racket_tip": "reward_terms/r_racket_tip",
    "racket_tip_err": "reward_terms/r_racket_tip_err",
    "root": "reward_terms/r_root",
    "track": "reward_terms/r_track",
    "upright": "reward_terms/r_upright",
}

def load_scalars(tb_dir):
    event_files = sorted(tb_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {tb_dir}")

    raw = {k: [] for k in TAGS}

    for f in event_files:
        print("reading", f)
        ea = EventAccumulator(str(f))
        ea.Reload()
        available = set(ea.Tags().get("scalars", []))

        for name, tag in TAGS.items():
            if tag not in available:
                print(f"missing {tag} in {f}")
                continue
            raw[name].extend((x.step, x.value) for x in ea.Scalars(tag))

    data = {}
    for name, vals in raw.items():
        # 若兩個 event file 有重複 step，保留後讀到的值
        step_to_value = {}
        for step, value in vals:
            step_to_value[int(step)] = float(value)
        data[name] = sorted(step_to_value.items(), key=lambda x: x[0])

    return data

def nearest_value(series, step):
    arr = np.array(series, dtype=float)
    idx = np.argmin(np.abs(arr[:, 0] - step))
    return float(arr[idx, 1])

def parse_step(path):
    # 例如 rl_model_55000000_steps.zip
    nums = re.findall(r"\d+", path.name)
    if not nums:
        return None
    return int(nums[-1])

def score_checkpoint(vals):
    # hard filter：先排掉明顯退化的 checkpoint
    if vals.get("qpos", 0) < 0.100:
        return None
    if vals.get("balance", 0) < 1.90:
        return None
    if vals.get("low_pose_err", 999) > 0.220:
        return None
    if vals.get("racket_tip_err", 999) > 1.62:
        return None
    if vals.get("racket_tip", 0) < 0.250:
        return None

    # 分數：重視 body/racket，懲罰 low pose / foot / tip error
    score = (
        3.0 * vals.get("qpos", 0)
        + 2.5 * vals.get("root", 0)
        + 1.8 * vals.get("racket", 0)
        + 1.8 * vals.get("racket_tip", 0)
        + 0.4 * vals.get("balance", 0)
        + 0.8 * vals.get("racket_orient", 0)
        - 1.2 * vals.get("foot_penalty", 0)
        - 1.5 * vals.get("low_pose_err", 0)
        - 0.25 * vals.get("racket_tip_err", 0)
        - 0.6 * vals.get("foot_slip_cost", 0)
    )
    return score

def main():
    data = load_scalars(TB_DIR)

    ckpts = []
    for p in CKPT_DIR.rglob("*.zip"):
        step = parse_step(p)
        if step is not None:
            ckpts.append((step, p))
    ckpts.sort()

    print(f"\nfound {len(ckpts)} checkpoint zips")

    results = []
    for step, path in ckpts:
        vals = {}
        for name, series in data.items():
            if series:
                vals[name] = nearest_value(series, step)

        score = score_checkpoint(vals)
        if score is None:
            continue

        results.append((score, step, path, vals))

    results.sort(reverse=True, key=lambda x: x[0])

    print("\nTop checkpoints:")
    for score, step, path, vals in results[:10]:
        print("=" * 80)
        print(f"score={score:.4f} step={step}")
        print(path)
        for k in [
            "balance",
            "qpos",
            "root",
            "racket",
            "racket_tip",
            "racket_tip_err",
            "low_pose_err",
            "foot_penalty",
            "foot_slip_cost",
        ]:
            if k in vals:
                print(f"  {k:16s}: {vals[k]:.5f}")

if __name__ == "__main__":
    main()