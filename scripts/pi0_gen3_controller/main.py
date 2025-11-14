#!/usr/bin/env python3

import argparse
import rclpy

from .pi0_client import PiZeroOpenPIClient


def main():
    ap = argparse.ArgumentParser("PiZero / OpenPI client with optional FollowJointTrajectory control")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default="do something")

    # ap.add_argument("--image-topic-external", default="/camera1/camera1/color/image_raw")
    # ap.add_argument("--image-topic-wrist",    default="/camera2/camera2/color/image_raw")
    ap.add_argument("--image-topic-external", default="/camera1/camera1/color/image_raw/compressed")
    ap.add_argument("--image-topic-wrist",    default="/camera2/camera2/color/image_raw/compressed")

    ap.add_argument("--joint-topic",          default="/joint_states")
    ap.add_argument("--gripper-name",         default="robotiq_85_left_knuckle_joint")
    ap.add_argument("--compressed", action="store_true", help="Use sensor_msgs/CompressedImage for cameras")

    ap.add_argument("--control-hz", type=float, default=10.0)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--prefetch-remaining", type=int, default=5, help="Start prefetch when <= this many remain")
    ap.add_argument("--min-exec-steps", type=int, default=4, help="Must execute at least this many steps before swapping")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-vel-rad", type=float, default=1.0, help="Clamp on returned joint velocities (rad/s)")

    ap.add_argument("--cmd-mode", choices=["log","velocity","trajectory"], default="log",
                    help="How to consume actions each tick")
    ap.add_argument("--gripper-action", choices=["velocity","position"], default="velocity",
                    help="What the policy's 8th channel means")

    ap.add_argument("--traj-action-ns", default="/joint_trajectory_controller/follow_joint_trajectory")
    ap.add_argument("--grip-action-ns", default="/robotiq_gripper_controller/gripper_cmd")
    ap.add_argument("--traj-goal-time-tol", type=float, default=0.25)
    ap.add_argument("--traj-min-segment-time", type=float, default=0.10, help="Minimum time per trajectory segment (s)")

    ap.add_argument("--vel-scale", type=float, default=0.1, help="Scale factor for joint/gripper velocities (safety)")

    args = ap.parse_args()

    rclpy.init()
    node = PiZeroOpenPIClient(
        host=args.host,
        port=args.port,
        prompt=args.prompt,
        img_topic_ext=args.image_topic_external,
        img_topic_wrist=args.image_topic_wrist,
        joint_topic=args.joint_topic,
        gripper_name=args.gripper_name,
        compressed=args.compressed,
        control_hz=args.control_hz,
        chunk_size=args.chunk_size,
        prefetch_remaining=args.prefetch_remaining,
        min_exec_steps=args.min_exec_steps,
        img_size=args.img_size,
        max_vel_rad=args.max_vel_rad,
        cmd_mode=args.cmd_mode,
        gripper_action=args.gripper_action,
        traj_action_ns=args.traj_action_ns,
        grip_action_ns=args.grip_action_ns,
        traj_goal_time_tol=args.traj_goal_time_tol,
        traj_min_segment_time=args.traj_min_segment_time,
        vel_scale=args.vel_scale,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
