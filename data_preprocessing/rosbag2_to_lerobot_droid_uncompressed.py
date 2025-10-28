#!/usr/bin/env python3
"""
Convert ROS 2 (rosbag2) logs (with uncompressed Image topics) into a LeRobot (HF) dataset
compatible with OpenPI's LeRobotDROIDDataConfig + DroidInputs.

- Two cameras:
    --camera-topic-base  /camera1/camera1/color/image_raw
    --camera-topic-wrist /camera2/camera2/color/image_raw
  (type: sensor_msgs/msg/Image)

- States:
    joint_position:   absolute 7-D (arm only)
    gripper_position: absolute 1-D

- Actions:
    velocities (dq/dt for 7 joints + dg/dt for gripper)  -> shape (8,)

Written as a simple HuggingFace LeRobot dataset via `LeRobotDataset`:
  - dataset.add_frame({...}) per frame
  - dataset.save_episode() per episode
  - stored at  $HF_LEROBOT_HOME/<repo_id>

Usage:
  uv run rosbag2_to_lerobot_droid_uncompressed.py \
    --input-root /path/to/bags \
    --repo-id /output/dataset/path/name \                
    --camera-topic-base  /camera1/camera1/color/image_raw \
    --camera-topic-wrist /camera2/camera2/color/image_raw \
    --joint-topic /joint_states \
    --gripper-name robotiq_85_left_knuckle_joint \
    --fps 20 \
    --episode-sec 12 \
    --overlap-sec 2 \
    --prompt "sort trash by category"

"""

import os
import cv2
import glob
import argparse
import numpy as np
from typing import List, Tuple, Dict, Any

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from PIL import Image

# LeRobot helpers (same API as your reference script)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME


# ---------------------------
# Utilities
# ---------------------------

def is_bag_dir(path: str) -> bool:
    """Return True iff path looks like a rosbag2 directory (has metadata.yaml)."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "metadata.yaml"))


def find_bag_dirs(root: str) -> List[str]:
    """Return a list of rosbag2 directories under root (or root itself if it’s a bag)."""
    if is_bag_dir(root):
        return [root]
    out = []
    for p in glob.glob(os.path.join(root, "*")):
            if is_bag_dir(p):
                out.append(p)
    return sorted(out)


def to_rgb_from_image_msg(msg) -> np.ndarray:
    """
    Convert sensor_msgs/msg/Image to RGB uint8 HxWx3.
    Handles common encodings: rgb8, bgr8, rgba8, bgra8, mono8.
    """
    h, w = msg.height, msg.width
    enc = (msg.encoding or "").lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == "rgb8":
        img = buf.reshape(h, w, 3)
        return img
    elif enc == "bgr8":
        img = buf.reshape(h, w, 3)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif enc == "rgba8":
        img = buf.reshape(h, w, 4)[:, :, :3]
        return img
    elif enc == "bgra8":
        img = buf.reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif enc == "mono8":
        img = buf.reshape(h, w)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        raise RuntimeError(f"Unsupported Image encoding '{msg.encoding}' (h={h}, w={w}).")


def resize_image_uint8_rgb(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize HxWx3 uint8 RGB to (W,H)=size using PIL bicubic.
    `size` is (width, height).
    """
    im = Image.fromarray(img)
    return np.asarray(im.resize(size, resample=Image.BICUBIC))


def finite_diff(vals: np.ndarray, dt: float) -> np.ndarray:
    """Compute forward finite differences (velocities) from absolute values; pad last row."""
    if len(vals) < 2:
        return np.zeros_like(vals)
    dv = (vals[1:] - vals[:-1]) / dt
    dv = np.vstack([dv, dv[-1:]])  # keep same length as input
    return dv


def slice_into_episodes(n_steps: int, hz: float, episode_sec: float, overlap_sec: float) -> List[slice]:
    """Create sliding-window episode slices covering n_steps with given length/overlap in seconds."""
    win = int(round(episode_sec * hz))
    ovl = int(round(overlap_sec * hz))
    if win <= 2:
        return [slice(0, n_steps)]
    out = []
    start = 0
    step = max(1, win - ovl)
    while start + win <= n_steps:
        out.append(slice(start, start + win))
        start += step
    return out


# ---------------------------
# Bag reading & alignment
# ---------------------------

def load_ros2_bag_uncompressed(
    bag_dir: str,
    topic_img_base: str,
    topic_img_wrist: str,
    topic_joint: str,
) -> Dict[str, Any]:
    """
    Read one rosbag2 directory and extract:
      - base_images:  list of (t_ns, rgb uint8 HxWx3)
      - wrist_images: list of (t_ns, rgb)
      - joints:       list of (t_ns, names[], pos[])   (absolute joint positions)
    Does not reorder joints; preserves message order as-is (we validate later).
    """
    reader = rosbag2_py.SequentialReader()
    storage_in = rosbag2_py.StorageOptions(uri=bag_dir, storage_id='sqlite3')
    reader.open(storage_in, rosbag2_py.ConverterOptions('', ''))

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    def resolve_topic(name: str) -> str:
        """Resolve topic name regardless of leading slash / namespace tweaks."""
        if name in topic_types:
            return name
        with_slash = "/" + name if not name.startswith("/") else name
        if with_slash in topic_types:
            return with_slash
        stripped = name.strip("/")
        for k in topic_types:
            if k.strip("/") == stripped:
                return k
        return name  # may 404 later

    topic_img_base = resolve_topic(topic_img_base)
    topic_img_wrist = resolve_topic(topic_img_wrist)
    topic_joint = resolve_topic(topic_joint)

    # Message classes
    ImageMsg = get_message("sensor_msgs/msg/Image") if topic_types.get(topic_img_base) else None
    JointState = get_message(topic_types.get(topic_joint)) if topic_types.get(topic_joint) else None

    base_images, wrist_images, joints = [], [], []
    warned_img = False

    while reader.has_next():
        topic, data, t = reader.read_next()  # t is int nanoseconds
        if topic == topic_img_base or topic == topic_img_wrist:
            if ImageMsg is None:
                if not warned_img:
                    print(f"⚠️  Image type not found for image topics in {bag_dir}. Skipping images.")
                    warned_img = True
                continue
            msg = deserialize_message(data, ImageMsg)
            rgb = to_rgb_from_image_msg(msg)
            if topic == topic_img_base:
                base_images.append((t, rgb))
            else:
                wrist_images.append((t, rgb))
        elif topic == topic_joint and JointState is not None:
            js = deserialize_message(data, JointState)
            names = list(js.name)                               # keep order as recorded
            pos = np.array(js.position, dtype=np.float32)       # absolute positions
            joints.append((t, names, pos))

    return dict(base_images=base_images, wrist_images=wrist_images, joints=joints)


def split_and_validate_joint_arrays(
    joints_msgs,
    gripper_name: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    From list of (t, names[], pos[]) create:
      - t_ns: (N,)          timestamps
      - q7:   (N, 7)        arm joint absolute positions in the bag's native order (excluding gripper)
      - g1:   (N,)          gripper absolute position
      - arm_names: list     the 7 arm joint names (preserved order)
    We DO NOT reorder. We require all messages to share the exact same names list as the first one.
    Rows not matching are dropped.
    """
    if not joints_msgs:
        return np.empty((0,), np.int64), np.empty((0, 7), np.float32), np.empty((0,), np.float32), []

    _, first_names, _ = joints_msgs[0]
    if gripper_name not in first_names:
        raise RuntimeError(f"Gripper '{gripper_name}' not found in first JointState names: {first_names}")

    grip_idx = first_names.index(gripper_name)
    arm_indices = [i for i, n in enumerate(first_names) if i != grip_idx]
    if len(arm_indices) < 7:
        raise RuntimeError(f"Expected at least 7 arm joints (excluding gripper). Found {len(arm_indices)}.")
    arm_indices = arm_indices[:7]     # take first 7 if more are present
    arm_names = [first_names[i] for i in arm_indices]

    t_list, q_list, g_list = [], [], []
    dropped = 0
    for (t, names, pos) in joints_msgs:
        if names != first_names:
            dropped += 1
            continue
        q = pos[arm_indices]
        g = pos[grip_idx]
        if not (np.isfinite(q).all() and np.isfinite(g)):
            dropped += 1
            continue
        t_list.append(t)
        q_list.append(q.astype(np.float32))
        g_list.append(np.float32(g))

    if dropped:
        print(f"⚠️  Dropped {dropped} JointState rows due to name-order mismatch or NaNs.")

    t_ns = np.array(t_list, dtype=np.int64)
    q7 = np.stack(q_list, axis=0) if q_list else np.empty((0, 7), np.float32)
    g1 = np.array(g_list, dtype=np.float32) if g_list else np.empty((0,), np.float32)
    return t_ns, q7, g1, arm_names


# ---------------------------
# Conversion: bag -> LeRobot
# ---------------------------

def convert_bag_to_lerobot_episodes(
    bag_dir: str,
    topic_img_base: str,
    topic_img_wrist: str,
    joint_topic: str,
    gripper_name: str,
    fps: float,
    episode_sec: float,
    overlap_sec: float,
    image_size_wh: Tuple[int, int],
    prompt: str | None,
    dataset: LeRobotDataset,
):
    """
    Convert a single bag to LeRobot episodes and write them into `dataset`:
      - resample to `fps`
      - write frames with `dataset.add_frame`
      - close episodes with `dataset.save_episode()`
    """
    raw = load_ros2_bag_uncompressed(bag_dir, topic_img_base, topic_img_wrist, joint_topic)
    if len(raw["joints"]) < 5:
        print(f"⚠️  No/too few joint messages in {bag_dir}. Skipping.")
        return

    t_j, q7, g1, _ = split_and_validate_joint_arrays(raw["joints"], gripper_name)
    if len(t_j) < 5:
        print(f"⚠️  Too few valid JointState rows after validation in {bag_dir}.")
        return

    # Build uniform grid at fps
    t0_ns, t1_ns = int(t_j.min()), int(t_j.max())
    if t1_ns <= t0_ns:
        return
    dt = 1.0 / fps
    t0 = t0_ns / 1e9
    t1 = t1_ns / 1e9
    grid = np.arange(t0, t1, dt, dtype=np.float64)
    if len(grid) < 5:
        return

    # Prepare images
    base_list = raw["base_images"]
    wrist_list = raw["wrist_images"]

    def times_and_frames(img_list):
        if not img_list:
            return np.empty((0,), dtype=np.float64), []
        t = np.array([t/1e9 for (t, _) in img_list], dtype=np.float64)
        frames = [img for (_, img) in img_list]
        return t, frames

    tb, frames_b = times_and_frames(base_list)
    tw, frames_w = times_and_frames(wrist_list)

    # Interpolate absolute joint/gripper to grid
    def interp_vec(ts_src, vals_src, ts_tgt):
        out = np.zeros((len(ts_tgt), vals_src.shape[1]), dtype=np.float32)
        for d in range(vals_src.shape[1]):
            out[:, d] = np.interp(ts_tgt, ts_src, vals_src[:, d])
        return out

    q7_on = interp_vec(t_j/1e9, q7, grid)
    g1_on = np.interp(grid, t_j/1e9, g1).astype(np.float32)

    # Nearest images for each grid tick (fill with zeros if missing)
    def nearest_images(t_src, frames_src, t_tgt, default_shape=(224, 224, 3)):
        if len(t_src) == 0:
            blank = np.zeros(default_shape, dtype=np.uint8)
            return [blank.copy() for _ in range(len(t_tgt))]
        idx = np.searchsorted(t_src, t_tgt, side="left")
        idx = np.clip(idx, 0, len(t_src)-1)
        left = np.maximum(idx-1, 0)
        choose_left = (idx == len(t_src)) | (
            (idx > 0) & ((t_tgt - t_src[left]) <= (t_src[idx] - t_tgt))
        )
        pick = np.where(choose_left, left, idx)
        return [frames_src[i] for i in pick]

    # Pre-resize shape
    W, H = image_size_wh  # (width, height)
    default_shape = (H, W, 3)

    base_on_raw  = nearest_images(tb, frames_b, grid, default_shape=default_shape)
    wrist_on_raw = nearest_images(tw, frames_w, grid, default_shape=default_shape)

    # Resize all frames to a fixed size (LeRobot feature needs fixed shape)
    base_on  = [resize_image_uint8_rgb(img, (W, H)) for img in base_on_raw]
    wrist_on = [resize_image_uint8_rgb(img, (W, H)) for img in wrist_on_raw]

    # Actions = velocities from absolute series
    dq = finite_diff(q7_on, dt)                               # (T,7)
    dg = finite_diff(g1_on.reshape(-1, 1), dt)[:, 0]          # (T,)
    actions = np.concatenate([dq, dg[:, None]], axis=1).astype(np.float32)  # (T,8)

    # Slice into episodes and write frames
    sls = slice_into_episodes(len(grid), fps, episode_sec, overlap_sec)
    for sl in sls:
        # one episode
        for t in range(sl.start, sl.stop):
            dataset.add_frame({
                # Images (note: plain DROID-style keys; repacked internally by OpenPI)
                "exterior_image_1_left": base_on[t],
                "wrist_image_left":      wrist_on[t],

                # States (absolute)
                "joint_position":   q7_on[t].astype(np.float32),
                "gripper_position": np.array([g1_on[t]], dtype=np.float32),

                # Actions (velocities)
                "actions": actions[t],

                # Optional language instruction
                "task": (prompt or "do something"),
            })
        dataset.save_episode()


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, help="Folder containing rosbag2 dirs, or a single bag dir.")
    ap.add_argument("--repo-id", required=True, help="LeRobot repo id, e.g. your_hf_username/my_droid_dataset")
    ap.add_argument("--camera-topic-base", required=True, help="e.g. /camera1/camera1/color/image_raw")
    ap.add_argument("--camera-topic-wrist", required=True, help="e.g. /camera2/camera2/color/image_raw")
    ap.add_argument("--joint-topic", default="/joint_states")
    ap.add_argument("--gripper-name", default="robotiq_85_left_knuckle_joint",
                    help="Name of the gripper joint to extract as the 8th action channel.")
    ap.add_argument("--fps", type=float, default=20.0, help="Target timeline FPS for resampling & dataset fps.")
    ap.add_argument("--episode-sec", type=float, default=12.0)
    ap.add_argument("--overlap-sec", type=float, default=2.0)
    ap.add_argument("--img-width", type=int, default=224)
    ap.add_argument("--img-height", type=int, default=224)
    ap.add_argument("--robot-type", default="kinova_gen3", help="Meta tag; free-form, e.g. 'kinova_gen3'.")
    ap.add_argument("--prompt", default=None, help="Optional language instruction per episode.")
    args = ap.parse_args()

    bags = find_bag_dirs(args.input_root)
    if not bags:
        raise SystemExit(f"No rosbag2 directories found under {args.input_root}")

    # Prepare output dataset at $HF_LEROBOT_HOME/<repo_id>
    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        # Clean any previous dataset with same repo id
        import shutil
        shutil.rmtree(output_path)

    # Define features (plain DROID-style names; OpenPI repacks internally)
    # We set fixed image size (default 224x224). You can change to (320,180) if you prefer DROID's size.
    features = {
        "exterior_image_1_left": {
            "dtype": "image",
            "shape": (args.img_height, args.img_width, 3),
            "names": ["height", "width", "channel"],
        },
        "wrist_image_left": {
            "dtype": "image",
            "shape": (args.img_height, args.img_width, 3),
            "names": ["height", "width", "channel"],
        },
        "joint_position": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["joint_position"],
        },
        "gripper_position": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["gripper_position"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (8,),
            "names": ["actions"],
        },

    }
    
    
    #Print the paths
    print("Cache root: ", HF_LEROBOT_HOME)
    print("Dataset path: ", output_path, flush=True)

    # Create LeRobotDataset (exact API your reference uses)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=int(round(args.fps)),
        features=features,
        image_writer_threads=8,
        image_writer_processes=4,
    )

    # Convert each bag into episodes inside this dataset
    for i, bag_dir in enumerate(bags, 1):
        print(f"[{i}/{len(bags)}] Converting {os.path.basename(bag_dir)} ...")
        convert_bag_to_lerobot_episodes(
            bag_dir=bag_dir,
            topic_img_base=args.camera_topic_base,
            topic_img_wrist=args.camera_topic_wrist,
            joint_topic=args.joint_topic,
            gripper_name=args.gripper_name,
            fps=args.fps,
            episode_sec=args.episode_sec,
            overlap_sec=args.overlap_sec,
            image_size_wh=(args.img_width, args.img_height),
            prompt=args.prompt,
            dataset=dataset,
        )

    print(f"✅ Saved LeRobot dataset to: {output_path}")
    print("   Now run norm stats + fine-tuning with OpenPI (LeRobotDROIDDataConfig).")


if __name__ == "__main__":
    main()
