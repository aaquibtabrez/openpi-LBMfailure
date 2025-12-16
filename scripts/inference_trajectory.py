#!/usr/bin/env python3
"""
Smooth trajectory ROS 2 client for OpenPI inference.

Usage example:

python3 inference_trajectory.py \
    --host 128.253.224.8 \
    --port 8000 \
    --cmd-mode trajectory \
    --compressed \
    --control-hz 10 \
    --prompt "pick up the blue cup and place it in red bin" \
    --speed-scale 1.3 \
    --gripper-action position \
    --steps-per-infer 10
"""

import argparse
import time
import math
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import CompressedImage, Image as RosImage, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration as RosDuration

# ----- Helpers ----- #

def dur_from_seconds(s: float) -> RosDuration:
    s = max(0.0, float(s))
    sec = int(s)
    nsec = int((s - sec) * 1e9)
    return RosDuration(sec=sec, nanosec=nsec)

def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi

def resize_with_pad_uint8(rgb: np.ndarray, size: int = 224) -> np.ndarray:
    import cv2
    h, w, _ = rgb.shape
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(rgb, (nw, nh))
    pad_h = size - nh
    pad_w = size - nw
    padded = cv2.copyMakeBorder(resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    return padded

# ----- Main Node ----- #

class SmoothPiZeroClient(Node):
    def __init__(
        self,
        host: str,
        port: int,
        prompt: str,
        joint_topic: str,
        gripper_name: str,
        compressed: bool,
        img_topic_ext: str,
        img_topic_wrist: str,
        control_hz: float,
        cmd_mode: str,
        gripper_action: str,
        vel_scale: float,
        speed_scale: float,
        chunk_size: int,
        steps_per_infer: int,
        traj_min_segment_time: float,
        traj_goal_time_tol: float,
    ):
        super().__init__("smooth_pi_zero_client")
        self.prompt = prompt
        self.joint_topic = joint_topic
        self.gripper_name = gripper_name
        self.use_compressed = compressed
        self.control_hz = control_hz
        self.cmd_mode = cmd_mode
        self.gripper_action = gripper_action
        self.vel_scale = vel_scale
        self.speed_scale = speed_scale
        self.chunk_size = chunk_size
        self.steps_per_infer = steps_per_infer
        self.traj_min_segment_time = traj_min_segment_time
        self.traj_goal_time_tol = traj_goal_time_tol
        self.dt = 1.0 / control_hz

        # State
        self.latest_q: Optional[np.ndarray] = None
        self.latest_g: Optional[float] = None
        self.latest_joint_names: List[str] = []

        # Dummy OpenPI client stub
        self.client = None  # Replace with your websocket client if needed

        # Subscribers
        SENSOR_QOS = QoSProfile(reliability=QoSReliabilityPolicy.RELIABLE,
                                history=QoSHistoryPolicy.KEEP_LAST, depth=5)
        if compressed:
            self.create_subscription(CompressedImage, img_topic_ext, self._cb_ext_compressed, SENSOR_QOS)
            self.create_subscription(CompressedImage, img_topic_wrist, self._cb_wrist_compressed, SENSOR_QOS)
        else:
            self.create_subscription(RosImage, img_topic_ext, self._cb_ext_raw, SENSOR_QOS)
            self.create_subscription(RosImage, img_topic_wrist, self._cb_wrist_raw, SENSOR_QOS)

        self.create_subscription(JointState, joint_topic, self._cb_joint, SENSOR_QOS)

        # Trajectory clients
        self._traj_client = ActionClient(self, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory")
        self._grip_client = ActionClient(self, GripperCommand, "/robotiq_gripper_controller/gripper_cmd")

        # Timer
        self.timer = self.create_timer(self.dt, self._tick)

        # Current chunk
        self._curr_chunk = None
        self._curr_idx = 0
        self._traj_active = False
        self._traj_active_until = None

        self.get_logger().info("SmoothPiZeroClient initialized")

    # --- Image callbacks --- #
    def _cb_ext_raw(self, msg): pass
    def _cb_wrist_raw(self, msg): pass
    def _cb_ext_compressed(self, msg): pass
    def _cb_wrist_compressed(self, msg): pass

    # --- Joint callback --- #
    def _cb_joint(self, msg: JointState):
        try:
            pos = np.array(msg.position, dtype=np.float32)
            if self.gripper_name in msg.name:
                gi = msg.name.index(self.gripper_name)
                q7 = np.delete(pos, gi)
                g1 = float(pos[gi])
            else:
                q7 = pos[:7]
                g1 = 0.0
            self.latest_q = q7
            self.latest_g = g1
            self.latest_joint_names = [n for n in msg.name if n != self.gripper_name]
        except Exception as e:
            self.get_logger().warn(f"Joint parse error: {e}")

    # --- Tick: execute trajectory --- #
    def _tick(self):
        if self.latest_q is None or self.latest_g is None:
            return

        if self.cmd_mode != "trajectory":
            return

        # Only infer when no trajectory running
        if self._traj_active:
            return

        # Get new chunk (stub: replace with actual OpenPI inference)
        if self._curr_chunk is None:
            self._curr_chunk = self._dummy_infer_chunk()
            self._curr_idx = 0
            if self._curr_chunk is None:
                return

        # Execute chunk
        self._send_trajectory_for_window(self._curr_chunk, start_idx=0, n_steps=len(self._curr_chunk))
        self._curr_chunk = None

    # --- Dummy chunk generator --- #
    def _dummy_infer_chunk(self):
        # Return (steps, 8) random small velocities
        return (np.random.rand(self.steps_per_infer, 8) - 0.5) * 0.1

    # --- Smooth trajectory sender --- #
    def _send_trajectory_for_window(self, chunk: np.ndarray, start_idx: int, n_steps: int):
        if self.latest_q is None:
            return

        # integrate velocities into positions smoothly
        q_curr = self.latest_q.tolist()
        g_curr = self.latest_g
        q_points: List[List[float]] = []
        g_points: List[float] = []
        t_accum = 0.0
        tfs_list: List[float] = []

        for a in chunk:
            dq = np.clip(a[:7], -1.0, 1.0) * self.vel_scale * self.speed_scale
            if self.gripper_action == "velocity":
                dg = float(a[7]) * self.vel_scale * self.speed_scale
                g_next = g_curr + dg * self.dt
            else:
                g_next = float(a[7])

            q_next = (np.array(q_curr) + dq * self.dt).tolist()
            q_points.append(q_next)
            g_points.append(g_next)
            q_curr = q_next
            g_curr = g_next

            t_accum += max(self.traj_min_segment_time, self.dt / self.speed_scale)
            tfs_list.append(t_accum)

        # Build arm trajectory
        traj = JointTrajectory()
        traj.joint_names = [f"joint_{i+1}" for i in range(7)]
        for q, tfs in zip(q_points, tfs_list):
            pt = JointTrajectoryPoint()
            pt.positions = q
            pt.time_from_start = dur_from_seconds(tfs)
            traj.points.append(pt)

        # Send FollowJointTrajectory goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        goal.goal_time_tolerance = dur_from_seconds(self.traj_goal_time_tol)
        if not self._traj_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("Trajectory server not available")
            return

        self._traj_active = True
        send_future = self._traj_client.send_goal_async(goal)
        send_future.add_done_callback(lambda fut: self._traj_done_callback(fut, g_points[-1]))

    def _traj_done_callback(self, fut, g_target):
        try:
            gh = fut.result()
            if not gh or not gh.accepted:
                self.get_logger().warn("Trajectory goal rejected")
                self._traj_active = False
                return
            result_future = gh.get_result_async()
            result_future.add_done_callback(lambda rf: self._traj_result_callback(rf))
        except Exception as e:
            self.get_logger().warn(f"Trajectory send error: {e}")
            self._traj_active = False

        # Send gripper goal
        try:
            if self._grip_client.wait_for_server(timeout_sec=0.5):
                g_goal = GripperCommand.Goal()
                g_goal.command.position = float(np.clip(g_target, 0.0, 1.0))
                g_goal.command.max_effort = 25.0
                self._grip_client.send_goal_async(g_goal)
        except Exception as e:
            self.get_logger().warn(f"Gripper send error: {e}")

    def _traj_result_callback(self, rf):
        try:
            res = rf.result()
            self.get_logger().info(f"Trajectory finished with status {res.status}")
        except Exception as e:
            self.get_logger().warn(f"Trajectory result exception: {e}")
        self._traj_active = False
        self._traj_active_until = None

# ----- Entry Point ----- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default="pick and place")
    ap.add_argument("--joint-topic", default="/joint_states")
    ap.add_argument("--gripper-name", default="robotiq_85_left_knuckle_joint")
    ap.add_argument("--compressed", action="store_true")
    ap.add_argument("--image-topic-external", default="/camera1/camera1/color/image_raw/compressed")
    ap.add_argument("--image-topic-wrist", default="/camera2/camera2/color/image_raw/compressed")
    ap.add_argument("--control-hz", type=float, default=10.0)
    ap.add_argument("--cmd-mode", choices=["log","velocity","trajectory"], default="trajectory")
    ap.add_argument("--gripper-action", choices=["velocity","position"], default="position")
    ap.add_argument("--vel-scale", type=float, default=0.1)
    ap.add_argument("--speed-scale", type=float, default=1.0)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--steps-per-infer", type=int, default=10)
    ap.add_argument("--traj-min-segment-time", type=float, default=0.05)
    ap.add_argument("--traj-goal-time-tol", type=float, default=0.25)

    args = ap.parse_args()
    rclpy.init()
    node = SmoothPiZeroClient(
        host=args.host,
        port=args.port,
        prompt=args.prompt,
        joint_topic=args.joint_topic,
        gripper_name=args.gripper_name,
        compressed=args.compressed,
        img_topic_ext=args.image_topic_external,
        img_topic_wrist=args.image_topic_wrist,
        control_hz=args.control_hz,
        cmd_mode=args.cmd_mode,
        gripper_action=args.gripper_action,
        vel_scale=args.vel_scale,
        speed_scale=args.speed_scale,
        chunk_size=args.chunk_size,
        steps_per_infer=args.steps_per_infer,
        traj_min_segment_time=args.traj_min_segment_time,
        traj_goal_time_tol=args.traj_goal_time_tol,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
