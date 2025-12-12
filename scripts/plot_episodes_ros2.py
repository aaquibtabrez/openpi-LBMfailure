#!/usr/bin/env python3
import os
import glob
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState


# -------- CONFIG --------
BAG_DIR = "~/data_collection/data_collection_green_container_all_reordered"
JOINT_TOPIC = "/joint_states"
MAX_EPISODES = 150
SAVE_DIR = "~/data_collection/data_collection_green_container_all_reordered/plots_joint"
# ------------------------


# -------------------------------------------------------------
# UTIL: Expand "~"
# -------------------------------------------------------------
def expand(path):
    return os.path.abspath(os.path.expanduser(path))


# -------------------------------------------------------------
# UTIL: Detect ROS2 bags (folder or .db3 file)
# -------------------------------------------------------------
def find_ros2_bag_paths(bag_root):
    """
    Returns a list of ROS2 bag paths.
    Supports both:
        - folder-style bags (containing metadata.yaml)
        - single-file data.db3 bags
    """
    bag_root = expand(bag_root)

    if not os.path.exists(bag_root):
        raise FileNotFoundError(f"Bag directory does not exist: {bag_root}")

    bag_paths = []

    # 1) Folder-based bags (preferred)
    for name in sorted(os.listdir(bag_root)):
        full = os.path.join(bag_root, name)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "metadata.yaml")):
            bag_paths.append(full)

    # 2) Fallback: single-file bags
    single_file_bags = sorted(glob.glob(os.path.join(bag_root, "*.db3")))
    bag_paths.extend(single_file_bags)

    return bag_paths


# -------------------------------------------------------------
# Load a single ROS2 bag episode
# -------------------------------------------------------------
def load_ros2_episode(bag_path, joint_topic):
    """
    Loads joint positions from a ROS2 bag (folder or .db3 file).
    Returns array shape (T, num_joints)
    """
    positions = []

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id="sqlite3"
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader = rosbag2_py.SequentialReader()

    try:
        reader.open(storage_options, converter_options)
    except Exception as e:
        print(f"  ❌ ERROR: Failed to open bag {bag_path}: {e}")
        return None

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    if joint_topic not in topic_types:
        print(f"  ⚠️ Topic '{joint_topic}' not found in bag.")
        return None

    while reader.has_next():
        try:
            topic, data, t = reader.read_next()
        except Exception as e:
            print(f"  ⚠️ Skipping corrupted message in {bag_path}: {e}")
            continue

        if topic != joint_topic:
            continue

        try:
            msg = deserialize_message(data, JointState)
        except Exception:
            print("  ⚠️ Could not deserialize JointState, skipping.")
            continue

        if msg.position:
            positions.append(np.array(msg.position, dtype=np.float32))

    if not positions:
        return None

    return np.stack(positions, axis=0)


# -------------------------------------------------------------
# Load all bag episodes
# -------------------------------------------------------------
def load_ros2_episodes(bag_dir, joint_topic):
    bag_paths = find_ros2_bag_paths(bag_dir)

    if not bag_paths:
        raise FileNotFoundError(f"No ROS2 bags (folder or .db3) found in: {bag_dir}")

    print(f"🔍 Found {len(bag_paths)} bag episodes.")

    episodes = OrderedDict()

    for ep_idx, bag_path in enumerate(bag_paths):
        print(f"\n📁 Loading episode {ep_idx}: {bag_path}")

        traj = load_ros2_episode(bag_path, joint_topic)
        if traj is None:
            print(f"  ⚠️ WARNING: No valid JointState data in this bag.")
            continue

        print(f"  ✔ Episode {ep_idx}: trajectory shape {traj.shape}")
        episodes[ep_idx] = traj

    if not episodes:
        raise RuntimeError("No valid episodes were loaded.")

    return episodes


# -------------------------------------------------------------
# Plotting
# -------------------------------------------------------------
def plot_joint_trajectories(episode_trajs, max_eps=MAX_EPISODES, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n📈 Saving plots to: {save_dir}")

    # Number of joints from first episode
    first_ep = next(iter(episode_trajs.values()))
    num_joints = first_ep.shape[1]

    for joint_idx in range(num_joints):
        plt.figure(figsize=(12, 6))

        for i, (ep, traj) in enumerate(episode_trajs.items()):
            if i >= max_eps:
                break

            y = traj[:, joint_idx]
            x = np.arange(len(y))

            plt.plot(x, y, linewidth=1.2, alpha=0.6)
            plt.annotate(str(ep), xy=(len(y)-1, y[-1]), fontsize=7, alpha=0.6)

        plt.title(f"Joint {joint_idx} — {min(max_eps, len(episode_trajs))} Episodes")
        plt.xlabel("Frame Index")
        plt.ylabel("Joint Position")
        plt.grid(True, alpha=0.3)

        outpath = os.path.join(save_dir, f"joint_{joint_idx}.png")
        plt.savefig(outpath, dpi=200)
        print(f"  ✔ Saved {outpath}")

        plt.close()


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------
def main():
    bag_dir = expand(BAG_DIR)
    episodes = load_ros2_episodes(bag_dir, JOINT_TOPIC)
    plot_joint_trajectories(episodes)


if __name__ == "__main__":
    main()
