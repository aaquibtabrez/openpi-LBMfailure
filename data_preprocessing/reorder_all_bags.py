#!/usr/bin/env python3
import os
import sys
import math
import argparse
import shutil
from typing import List

import rosbag2_py
from rclpy.serialization import deserialize_message, serialize_message
from rosidl_runtime_py.utilities import get_message


def is_bag_dir(path: str) -> bool:
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "metadata.yaml"))


def find_bag_dirs(root: str) -> List[str]:
    bags = []
    if is_bag_dir(root):
        bags.append(root)
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if is_bag_dir(p):
            bags.append(p)
    return bags


def norm_topic(t: str) -> str:
    # normalize "/foo/" -> "/foo"
    return t[:-1] if t.endswith("/") and t != "/" else t


def make_index_map(current_names, desired_order):
    idx = {n: i for i, n in enumerate(current_names)}
    return [idx.get(n, -1) for n in desired_order]


def reorder_joint_state_msg(msg, desired_order):
    idx_map = make_index_map(msg.name, desired_order)

    def pick(arr, i):
        if i < 0 or i >= len(arr):
            return math.nan
        return arr[i]

    msg.name     = [desired_order[j] for j in range(len(idx_map))]
    msg.position = [pick(msg.position, di) for di in idx_map]
    msg.velocity = [pick(msg.velocity, di) for di in idx_map]
    msg.effort   = [pick(msg.effort,   di) for di in idx_map]
    return msg


def process_single_bag(in_bag_dir: str, out_bag_dir: str, js_topic: str, desired_order: List[str], overwrite: bool) -> dict:
    """
    Read one bag dir, reorder JointState on js_topic, write new bag dir.
    Returns stats dict.
    """
    stats = {
        "bag": in_bag_dir,
        "messages_total": 0,
        "messages_reordered": 0,
        "topic_present": False,
    }

    # Prepare reader (input)
    reader = rosbag2_py.SequentialReader()
    storage_in = rosbag2_py.StorageOptions(uri=in_bag_dir, storage_id='sqlite3')
    reader.open(storage_in, rosbag2_py.ConverterOptions('', ''))

    all_topics = list(reader.get_all_topics_and_types())
    topic_types = {t.name: t.type for t in all_topics}

    # Normalize names for matching
    js_topic_norm = norm_topic(js_topic)
    topic_names_norm = {norm_topic(name): name for name in topic_types.keys()}
    matched_topic_name = topic_names_norm.get(js_topic_norm, None)

    js_type_str = topic_types.get(matched_topic_name) if matched_topic_name else None
    JointState = get_message(js_type_str) if js_type_str else None
    stats["topic_present"] = js_type_str is not None

    # Prepare writer (output)

    parent = os.path.dirname(os.path.abspath(out_bag_dir))
    os.makedirs(parent or ".", exist_ok=True)

    # If the output bag directory already exists:
    if os.path.isdir(out_bag_dir):
        # If it's empty (maybe left by a previous failed attempt), remove it.
        if not os.listdir(out_bag_dir):
            import shutil
            shutil.rmtree(out_bag_dir)
        else:
            if not overwrite:
                raise RuntimeError(
                    f"Output bag directory already exists and is not empty: {out_bag_dir} "
                    f"(use --overwrite to replace)"
                )
            else:
                import shutil
                shutil.rmtree(out_bag_dir)

    # Now open the writer; do NOT pre-create out_bag_dir yourself.
    writer = rosbag2_py.SequentialWriter()
    storage_out = rosbag2_py.StorageOptions(uri=out_bag_dir, storage_id='sqlite3')
    writer.open(storage_out, rosbag2_py.ConverterOptions('', ''))


    # Recreate topics (pass-through)
    for t in all_topics:
        md = rosbag2_py.TopicMetadata(
            name=t.name,
            type=t.type,
            serialization_format='cdr'
        )
        writer.create_topic(md)

    # Iterate messages
    while reader.has_next():
        topic, data, t = reader.read_next()
        stats["messages_total"] += 1

        if matched_topic_name and norm_topic(topic) == js_topic_norm and JointState is not None:
            msg = deserialize_message(data, JointState)
            msg = reorder_joint_state_msg(msg, desired_order)
            out_bytes = serialize_message(msg)
            writer.write(topic, out_bytes, t)
            stats["messages_reordered"] += 1
        else:
            writer.write(topic, data, t)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Batch-reorder sensor_msgs/JointState arrays in many rosbag2 directories."
    )
    parser.add_argument("input_root", help="Folder containing rosbag2 directories (and/or itself a bag dir).")
    parser.add_argument("output_root", help="Folder to place reordered bags; each bag becomes <name>_reordered.")
    parser.add_argument("--topic", default="/joint_states",
                        help="JointState topic to reorder (default: /joint_states). Trailing slash will be ignored.")
    parser.add_argument("--order", required=True,
                        help="Comma-separated desired joint order, e.g. 'joint_1,joint_2,...,robotiq_85_left_knuckle_joint'")
    parser.add_argument("--overwrite", action="store_true",
                        help="If set, delete existing output bag directories before writing.")
    args = parser.parse_args()

    desired_order = [s.strip() for s in args.order.split(",") if s.strip()]
    if not desired_order:
        print("ERROR: --order must contain at least one joint name.", file=sys.stderr)
        sys.exit(2)

    in_root = os.path.abspath(args.input_root)
    out_root = os.path.abspath(args.output_root)
    os.makedirs(out_root, exist_ok=True)

    bag_dirs = find_bag_dirs(in_root)
    if not bag_dirs:
        print(f"No rosbag2 directories (with metadata.yaml) found under: {in_root}")
        sys.exit(1)

    print(f"Found {len(bag_dirs)} bag(s). Starting processing...\n")

    grand_total = 0
    grand_reordered = 0

    for i, bag_dir in enumerate(bag_dirs, 1):
        base = os.path.basename(os.path.normpath(bag_dir))
        out_dir = os.path.join(out_root, f"{base}_reordered")
        print(f"[{i}/{len(bag_dirs)}] Processing '{base}'...")

        try:
            stats = process_single_bag(bag_dir, out_dir, args.topic, desired_order, args.overwrite)
        except Exception as e:
            print(f"   ❌ Error processing '{base}': {e}")
            print(f"   Skipping. (You can re-run with --overwrite if it already exists.)\n")
            continue

        grand_total += stats["messages_total"]
        grand_reordered += stats["messages_reordered"]

        if not stats["topic_present"]:
            print(f"   ⚠️  Topic '{norm_topic(args.topic)}' not found. Copied bag unchanged.")
        else:
            print(f"   ✅ Finished bag '{base}'. Reordered {stats['messages_reordered']} of {stats['messages_total']} messages.")
        print(f"   Output written to: {out_dir}\n")

    print("────────────────────────────────────────────")
    print(f"✔ All done. Processed {len(bag_dirs)} bag(s).")
    print(f"Total messages: {grand_total}")
    print(f"Reordered on topic '{norm_topic(args.topic)}': {grand_reordered}")
    print(f"Output root directory: {out_root}")
    print("────────────────────────────────────────────")


if __name__ == "__main__":
    main()
