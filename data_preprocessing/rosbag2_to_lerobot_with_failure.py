#!/usr/bin/env python3
"""
Convert ROS 2 (rosbag2) logs into a LeRobot (HF) dataset compatible with OpenPI's
LeRobotDROIDDataConfig + DroidInputs.

Supports TWO camera topics (exterior + wrist) and BOTH image encodings:
- Uncompressed: sensor_msgs/msg/Image (e.g., /camera1/.../image_raw)
- Compressed:   sensor_msgs/msg/CompressedImage (e.g., /camera1/.../image_raw/compressed)
  (enable with --compressed)

States:
  joint_position:   absolute 7-D (arm only)
  gripper_position: absolute 1-D

Actions (for Pi0-style training):
  - joints:   either velocities (dq/dt) or deltas (q[t+1] - q[t])  -> 7 dims
  - gripper:  absolute gripper position                           -> 1 dim
  => actions shape: (8,)

Optional episode truncation:
  - If --cut-gripper-threshold is set (e.g. 0.6), each bag/episode is cut
    at the first frame where gripper_position >= threshold.
  - Then we append --cut-extra-frames copies of that final frame with
    zero joint actions and constant gripper position (robot "frozen").
    This is useful to train "fail-on-purpose" behaviors where the robot
    stops closing the gripper just before fully closed.

Writes a simple LeRobot HF dataset via `LeRobotDataset`:
  - dataset.add_frame({...}) per frame
  - dataset.save_episode() once PER BAG (one episode per bag)
  - stored at $HF_LEROBOT_HOME/<repo_id>

Examples:
  # Uncompressed topics
  python3 rosbag2_to_lerobot_with_failure.py \
    --input-root data_collection_blue_cup_blue_bin_reordered \
    --repo-id lerobot_gen3_blue_cup_blue_bin_failure \
    --camera-topic-base  /camera1/camera1/color/image_raw \
    --camera-topic-wrist /camera2/camera2/color/image_raw \
    --joint-topic /joint_states \
    --gripper-name robotiq_85_left_knuckle_joint \
    --fps 20 \
    --joint-action-type velocity \
    --cut-gripper-threshold 0.5 \
    --cut-extra-frames 20 \
    --prompt "pick but stop before grasping"

  # Compressed topics
  python3 rosbag2_to_lerobot_with_failure.py \
    --input-root data_collection_blue_cup_blue_bin_reordered \
    --repo-id lerobot_gen3_blue_cup_blue_bin_failure \
    --camera-topic-base  /camera1/camera1/color/image_raw/compressed \
    --camera-topic-wrist /camera2/camera2/color/image_raw/compressed \
    --joint-topic /joint_states \
    --gripper-name robotiq_85_left_knuckle_joint \
    --fps 20 --compressed \
    --joint-action-type velocity \
    --cut-gripper-threshold 0.5 \
    --cut-extra-frames 20 \
    --prompt "pick but stop before grasping"
"""

import io
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

# LeRobot helpers (you’re using the "common" layout; keep it)
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, HF_LEROBOT_HOME


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
        return buf.reshape(h, w, 3)
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


def to_rgb_from_compressed_msg(msg) -> np.ndarray:
    """
    Convert sensor_msgs/msg/CompressedImage to RGB uint8 HxWx3.

    Uses PIL to decode JPEG bytes (robust to weird ROS2 data types for msg.data),
    then converts to a numpy RGB array.
    """
    # Normalize msg.data into plain bytes
    data = msg.data
    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)

    # Decode with PIL
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")  # ensure 3-channel RGB
        arr = np.array(im, dtype=np.uint8)  # H x W x 3

    # Optional sanity check
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise RuntimeError(
            f"Decoded compressed image has unexpected shape {arr.shape}, "
            f"format={getattr(msg, 'format', 'unknown')}"
        )

    return arr


def resize_image_uint8_rgb(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize HxWx3 uint8 RGB to (W,H)=size using PIL bicubic. `size` is (width, height).
    """
    im = Image.fromarray(img)
    return np.asarray(im.resize(size, resample=Image.BICUBIC))


def finite_diff(vals: np.ndarray, dt: float, as_velocity: bool = True) -> np.ndarray:
    """
    Compute forward finite differences from absolute values.

    If as_velocity=True:  (vals[t+1] - vals[t]) / dt
    If as_velocity=False:  vals[t+1] - vals[t]      (plain delta)

    Last row is padded with the previous diff to keep length.
    """
    if len(vals) < 2:
        return np.zeros_like(vals)
    dv = vals[1:] - vals[:-1]
    if as_velocity:
        dv = dv / dt
    dv = np.vstack([dv, dv[-1:]])  # keep same length as input
    return dv


# ---------------------------
# Bag reading & alignment
# ---------------------------

def load_ros2_bag(
    bag_dir: str,
    topic_img_base: str,
    topic_img_wrist: str,
    topic_joint: str,
    compressed: bool,
) -> Dict[str, Any]:
    """
    Read one rosbag2 directory and extract:
      - base_images:  list of (t_ns, rgb uint8 HxWx3)
      - wrist_images: list of (t_ns, rgb)
      - joints:       list of (t_ns, names[], pos[])   (absolute joint positions)
    Does not reorder joints; preserves message order as-is (validated later).
    Chooses between raw Image vs CompressedImage depending on `compressed`.
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
    if compressed:
        ImgMsg = get_message("sensor_msgs/msg/CompressedImage") if topic_types.get(topic_img_base) else None
    else:
        ImgMsg = get_message("sensor_msgs/msg/Image") if topic_types.get(topic_img_base) else None
    JointState = get_message(topic_types.get(topic_joint)) if topic_types.get(topic_joint) else None

    base_images, wrist_images, joints = [], [], []
    warned_img = False

    while reader.has_next():
        topic, data, t = reader.read_next()  # t is int nanoseconds
        if topic == topic_img_base or topic == topic_img_wrist:
            if ImgMsg is None:
                if not warned_img:
                    print(f"⚠️  Image type not found for image topics in {bag_dir}. Skipping images.")
                    warned_img = True
                continue
            msg = deserialize_message(data, ImgMsg)
            if compressed:
                rgb = to_rgb_from_compressed_msg(msg)
            else:
                rgb = to_rgb_from_image_msg(msg)
            if topic == topic_img_base:
                base_images.append((t, rgb))
            else:
                wrist_images.append((t, rgb))
        elif topic == topic_joint and JointState is not None:
            js = deserialize_message(data, JointState)
            names = list(js.name)                         # keep order as recorded
            pos = np.array(js.position, dtype=np.float32) # absolute positions
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
    We DO NOT reorder. Require all messages to share the exact same names list as the first one.
    Rows not matching are dropped.
    """
    if not joints_msgs:
        return np.empty((0,), np.int64), np.empty((0, 7), np.float32), np.empty((0,), np.float32), []

    _, first_names, _ = joints_msgs[0]
    if gripper_name not in first_names:
        raise RuntimeError(f"Gripper '{gripper_name}' not found in first JointState names: {first_names}")

    grip_idx = first_names.index(gripper_name)
    arm_indices = [i for i, _ in enumerate(first_names) if i != grip_idx]
    if len(arm_indices) < 7:
        raise RuntimeError(f"Expected at least 7 arm joints (excluding gripper). Found {len(arm_indices)}.")
    arm_indices = arm_indices[:7]
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
# Conversion: bag -> LeRobot (ONE episode per bag)
# ---------------------------

def convert_bag_to_lerobot_episode(
    bag_dir: str,
    topic_img_base: str,
    topic_img_wrist: str,
    joint_topic: str,
    gripper_name: str,
    fps: float,
    image_size_wh: Tuple[int, int],
    prompt: str | None,
    dataset: LeRobotDataset,
    compressed: bool,
    joint_action_type: str = "velocity",   # "velocity" or "delta" for joints
    cut_gripper_threshold: float | None = None,  # if set, cut episode when gripper >= this
    cut_extra_frames: int = 20,                  # number of frozen frames to append after cutoff
):
    """
    Convert a single bag into EXACTLY ONE LeRobot episode:
      - resample entire bag to `fps`
      - optionally truncate when gripper_position exceeds a threshold and append frozen frames
      - write every frame with `dataset.add_frame(...)`
      - close the episode once with `dataset.save_episode()`
      - choose raw vs compressed decoding based on `compressed`
    """
    raw = load_ros2_bag(bag_dir, topic_img_base, topic_img_wrist, joint_topic, compressed=compressed)
    if len(raw["joints"]) < 3:
        print(f"⚠️  No/too few joint messages in {bag_dir}. Skipping.", flush=True)
        return

    # Absolute joint/gripper series (preserve bag’s native order; enforce consistent names)
    t_j, q7, g1, _ = split_and_validate_joint_arrays(raw["joints"], gripper_name)
    if len(t_j) < 3:
        print(f"⚠️  Too few valid JointState rows after validation in {bag_dir}.", flush=True)
        return

    # Build uniform grid for the entire bag at fps
    t0_ns, t1_ns = int(t_j.min()), int(t_j.max())
    if t1_ns <= t0_ns:
        return
    dt = 1.0 / fps
    t0 = t0_ns / 1e9
    t1 = t1_ns / 1e9
    grid = np.arange(t0, t1, dt, dtype=np.float64)
    if len(grid) < 3:
        return

    # Prepare images (two streams)
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

    q7_on = interp_vec(t_j/1e9, q7, grid)  # (T,7) absolute
    g1_on = np.interp(grid, t_j/1e9, g1).astype(np.float32)  # (T,) absolute

    # Nearest images per grid tick (fill zeros if missing)
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

    W, H = image_size_wh
    default_shape = (H, W, 3)

    base_on_raw  = nearest_images(tb, frames_b, grid, default_shape=default_shape)
    wrist_on_raw = nearest_images(tw, frames_w, grid, default_shape=default_shape)

    # Resize to fixed size that matches your declared features
    base_on  = [resize_image_uint8_rgb(img, (W, H)) for img in base_on_raw]
    wrist_on = [resize_image_uint8_rgb(img, (W, H)) for img in wrist_on_raw]

    # ----------------------------------------
    # Optional: cut episode when gripper exceeds threshold and pad with frozen frames.
    # ----------------------------------------
    if cut_gripper_threshold is not None:
        idx_over = np.nonzero(g1_on >= cut_gripper_threshold)[0]
        if len(idx_over) > 0:
            cut_idx = int(idx_over[0])  # first time gripper >= threshold
            # Snapshot the final state at cutoff
            last_q = q7_on[cut_idx].copy()
            last_g = float(g1_on[cut_idx])
            last_base = base_on[cut_idx]
            last_wrist = wrist_on[cut_idx]

            # Trim sequences up to and including cut_idx
            q7_on = q7_on[:cut_idx + 1]
            g1_on = g1_on[:cut_idx + 1]
            base_on = base_on[:cut_idx + 1]
            wrist_on = wrist_on[:cut_idx + 1]

            # Append extra frozen frames: same obs/state, no motion
            for _ in range(cut_extra_frames):
                q7_on = np.vstack([q7_on, last_q[None, :]])
                g1_on = np.concatenate([g1_on, np.array([last_g], dtype=np.float32)])
                base_on.append(last_base)
                wrist_on.append(last_wrist)

    # ----------------------------------------
    # Actions:
    # - joints: velocity (dq/dt) or delta (q[t+1]-q[t]) based on joint_action_type
    # - gripper: ALWAYS absolute position (Pi0-style)
    # ----------------------------------------
    dq = finite_diff(
        q7_on,
        dt,
        as_velocity=(joint_action_type == "velocity"),
    )  # (T,7)

    g_abs = g1_on.astype(np.float32)  # (T,)

    actions = np.concatenate(
        [dq, g_abs[:, None]],
        axis=1,
    ).astype(np.float32)  # (T,8)

    # ---- ONE EPISODE FOR THE ENTIRE BAG ----
    T = len(g_abs)
    assert len(base_on) == T and len(wrist_on) == T and q7_on.shape[0] == T and actions.shape[0] == T

    for t in range(T):
        dataset.add_frame({
            # two cameras (DROID-style keys; OpenPI repacks internally)
            "exterior_image_1_left": base_on[t],
            "wrist_image_left":      wrist_on[t],

            # absolute state
            "joint_position":   q7_on[t].astype(np.float32),
            "gripper_position": np.array([g_abs[t]], dtype=np.float32),

            # actions: joints = vel/delta, gripper = absolute
            "actions": actions[t],

            # language task (string)
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
    ap.add_argument("--camera-topic-base", required=True, help="e.g., /camera1/camera1/color/image_raw OR .../compressed")
    ap.add_argument("--camera-topic-wrist", required=True, help="e.g., /camera2/camera2/color/image_raw OR .../compressed")
    ap.add_argument("--joint-topic", default="/joint_states")
    ap.add_argument("--gripper-name", default="robotiq_85_left_knuckle_joint",
                    help="Name of the gripper joint to extract as the 8th action channel.")
    ap.add_argument("--fps", type=float, default=20.0, help="Target timeline FPS for resampling & dataset fps.")
    ap.add_argument("--img-width", type=int, default=224)
    ap.add_argument("--img-height", type=int, default=224)
    ap.add_argument("--robot-type", default="kinova_gen3", help="Meta tag; free-form, e.g. 'kinova_gen3'.")
    ap.add_argument("--prompt", default=None, help="Optional language instruction per episode.")
    ap.add_argument("--compressed", action="store_true",
                    help="If set, read sensor_msgs/msg/CompressedImage and decode JPEG bytes.")

    # NEW: joint action representation
    ap.add_argument(
        "--joint-action-type",
        choices=["velocity", "delta"],
        default="velocity",
        help="How to derive joint actions from absolute positions: "
             "'velocity' = dq/dt, 'delta' = q[t+1]-q[t]. Gripper action is always absolute.",
    )

    # NEW: episode cutoff based on gripper position
    ap.add_argument(
        "--cut-gripper-threshold",
        type=float,
        default=None,
        help="If set, truncate each episode at the first frame where absolute gripper_position "
             ">= this value, then append --cut-extra-frames frozen frames. "
             "If omitted, no truncation is applied.",
    )
    ap.add_argument(
        "--cut-extra-frames",
        type=int,
        default=20,
        help="Number of additional frozen frames to append after gripper threshold cutoff.",
    )

    args = ap.parse_args()

    bags = find_bag_dirs(args.input_root)
    if not bags:
        raise SystemExit(f"No rosbag2 directories found under {args.input_root}")

    # Prepare output dataset at $HF_LEROBOT_HOME/<repo_id>
    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)

    # Define features (plain DROID-style names; OpenPI repacks internally)
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

    print("Cache root:", HF_LEROBOT_HOME)
    print("Dataset path:", output_path, flush=True)

    # Create LeRobotDataset (one dataset for all episodes/bags)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        fps=int(round(args.fps)),
        features=features,
        image_writer_threads=8,
        image_writer_processes=4,
    )

    # Convert each bag into ONE episode
    for i, bag_dir in enumerate(bags, 1):
        print(f"[{i}/{len(bags)}] Converting {os.path.basename(bag_dir)} ...")
        convert_bag_to_lerobot_episode(
            bag_dir=bag_dir,
            topic_img_base=args.camera_topic_base,
            topic_img_wrist=args.camera_topic_wrist,
            joint_topic=args.joint_topic,
            gripper_name=args.gripper_name,
            fps=args.fps,
            image_size_wh=(args.img_width, args.img_height),
            prompt=args.prompt,
            dataset=dataset,
            compressed=args.compressed,
            joint_action_type=args.joint_action_type,
            cut_gripper_threshold=args.cut_gripper_threshold,
            cut_extra_frames=args.cut_extra_frames,
        )

    print(f"✅ Saved LeRobot dataset to: {output_path}")
    print("   Now run norm stats + fine-tuning with OpenPI (LeRobotDROIDDataConfig).")


if __name__ == "__main__":
    main()
