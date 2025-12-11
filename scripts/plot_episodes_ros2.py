import os
import glob
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import JointState


# -------- CONFIG --------
BAG_DIR = "bags"             # directory of .db3 ROS2 bags
JOINT_TOPIC = "/joint_states"
MAX_EPISODES = 50
SAVE_DIR = "plots_ros2"
# ------------------------


def load_ros2_episode(bag_path, joint_topic):
    """
    Loads joint positions from a ROS2 bag (SQLite .db3 file).
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
    reader.open(storage_options, converter_options)

    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}

    if joint_topic not in type_map:
        print(f"  WARNING: topic {joint_topic} not in bag")
        return None

    while reader.has_next():
        topic, data, t = reader.read_next()

        if topic != joint_topic:
            continue

        msg = deserialize_message(data, JointState)
        if len(msg.position) > 0:
            positions.append(np.array(msg.position, dtype=np.float32))

    if not positions:
        return None

    return np.stack(positions, axis=0)  # shape (T, num_joints)


def load_ros2_episodes(bag_dir, joint_topic):
    """
    Loads all ROS2 bags in a directory.
    One bag = one episode.
    """
    episode_trajs = OrderedDict()

    bag_paths = sorted(glob.glob(os.path.join(bag_dir, "*.db3")))
    if not bag_paths:
        raise FileNotFoundError(f"No .db3 ROS2 bags found in {bag_dir}")

    print(f"Found {len(bag_paths)} ROS2 bag episodes")

    for ep_idx, bag_path in enumerate(bag_paths):
        print(f"Loading episode {ep_idx}: {bag_path}")

        traj = load_ros2_episode(bag_path, joint_topic)
        if traj is None:
            print(f"  WARNING: No joint_states in {bag_path}")
            continue

        print(f"  Episode {ep_idx}: trajectory shape {traj.shape}")
        episode_trajs[ep_idx] = traj

    return episode_trajs


def plot_joint_trajectories(episode_trajs, max_eps=MAX_EPISODES, save_dir=SAVE_DIR):
    os.makedirs(save_dir, exist_ok=True)

    first_ep = next(iter(episode_trajs.values()))
    num_joints = first_ep.shape[1]

    for joint_idx in range(num_joints):
        plt.figure(figsize=(12, 6))

        for i, (ep, traj) in enumerate(episode_trajs.items()):
            if i >= max_eps:
                break

            y = traj[:, joint_idx]
            x = np.arange(len(y))

            plt.plot(x, y, alpha=0.55, linewidth=1.3)

            # annotate the episode number at the end of its trajectory
            plt.annotate(str(ep), xy=(len(y) - 1, y[-1]), fontsize=7, alpha=0.75)

        plt.title(f"Joint {joint_idx} — {max_eps} ROS2 bag episodes")
        plt.xlabel("Frame index")
        plt.ylabel("Joint position")
        plt.grid(True)

        outpath = os.path.join(save_dir, f"joint_{joint_idx}.png")
        plt.savefig(outpath, dpi=200)
        print(f"Saved {outpath}")

        plt.close()


def main():
    episodes = load_ros2_episodes(BAG_DIR, JOINT_TOPIC)
    plot_joint_trajectories(episodes)


if __name__ == "__main__":
    main()
