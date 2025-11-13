#!/usr/bin/env python3
"""
PiZero / OpenPI ROS 2 client with double-buffer action chunks and optional trajectory execution.

Modes:
  --cmd-mode log         : print actions (safe default)
  --cmd-mode velocity    : (stub) stream dq (rad/s) to your velocity controller (wire later)
  --cmd-mode trajectory  : convert each predicted velocity chunk to a short FollowJointTrajectory goal
                           and sync Robotiq gripper over GripperCommand action.

Trajectory control references your example:
  - /joint_trajectory_controller/follow_joint_trajectory  (control_msgs/FollowJointTrajectory)
  - /robotiq_gripper_controller/gripper_cmd               (control_msgs/GripperCommand)

Assumptions:
  - Observation gripper is absolute position.
  - ACTION channel 8 is *gripper velocity* by default for your model (set --gripper-action=velocity).
    If your model outputs gripper *position* like DROID, set --gripper-action=position.

Swap strategy:
  - Prefetch next chunk when remaining <= --prefetch-remaining (default 5).
  - Swap as soon as a prefetched chunk is ready AND we've executed at least --min-exec-steps from the current chunk.
"""

import argparse
import concurrent.futures
import json
import math
import threading
import time
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import JointState

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration as RosDuration

# OpenPI client + utils
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# ---------- Helpers ----------

# SENSOR_QOS = QoSProfile(
#     reliability=QoSReliabilityPolicy.BEST_EFFORT,
#     history=QoSHistoryPolicy.KEEP_LAST,
#     depth=5,
# )

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
)


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi

def to_rgb_from_raw_msg(msg: RosImage) -> np.ndarray:
    import numpy as _np, cv2 as _cv2
    h, w = msg.height, msg.width
    enc = (msg.encoding or "").lower()
    buf = _np.frombuffer(msg.data, dtype=_np.uint8)
    if enc == "rgb8":
        return buf.reshape(h, w, 3)
    elif enc == "bgr8":
        img = buf.reshape(h, w, 3)
        return _cv2.cvtColor(img, _cv2.COLOR_BGR2RGB)
    elif enc == "rgba8":
        img = buf.reshape(h, w, 4)[:, :, :3]
        return img
    elif enc == "bgra8":
        img = buf.reshape(h, w, 4)
        return _cv2.cvtColor(img, _cv2.COLOR_BGRA2RGB)
    elif enc == "mono8":
        img = buf.reshape(h, w)
        return _cv2.cvtColor(img, _cv2.COLOR_GRAY2RGB)
    else:
        raise RuntimeError(f"Unsupported Image encoding '{msg.encoding}' ({h}x{w}).")

def to_rgb_from_compressed_msg(msg: CompressedImage) -> np.ndarray:
    import numpy as _np, cv2 as _cv2
    np_arr = _np.frombuffer(msg.data, _np.uint8)
    bgr = _cv2.imdecode(np_arr, _cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to decode CompressedImage (format={getattr(msg, 'format', 'unknown')})")
    return _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)

def resize_with_pad_uint8(rgb: np.ndarray, size: int = 224) -> np.ndarray:
    arr = image_tools.resize_with_pad(rgb, size, size)
    return image_tools.convert_to_uint8(arr)

def dur_from_seconds(s: float) -> RosDuration:
    s = max(0.0, float(s))
    sec = int(s)
    nsec = int((s - sec) * 1e9)
    return RosDuration(sec=sec, nanosec=nsec)

# ---------- Main Node ----------

class PiZeroOpenPIClient(Node):
    def __init__(
        self,
        *,
        host: str,
        port: int,
        prompt: str,
        img_topic_ext: str,
        img_topic_wrist: str,
        joint_topic: str,
        gripper_name: str,
        compressed: bool,
        control_hz: float,
        chunk_size: int,
        prefetch_remaining: int,
        min_exec_steps: int,
        img_size: int,
        max_vel_rad: float,
        cmd_mode: str,                # "log" | "velocity" | "trajectory"
        gripper_action: str,          # "velocity" | "position"  (what the policy outputs)
        traj_action_ns: str,          # /joint_trajectory_controller/follow_joint_trajectory
        grip_action_ns: str,          # /robotiq_gripper_controller/gripper_cmd
        traj_goal_time_tol: float,
        traj_min_segment_time: float, # seconds per point (lower bound)
        vel_scale: float,  
    ):
        super().__init__("pizero_openpi_traj_client")

        # OpenPI client
        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

        # Config
        self.prompt = prompt
        self.img_topic_ext = img_topic_ext
        self.img_topic_wrist = img_topic_wrist
        self.joint_topic = joint_topic
        self.gripper_name = gripper_name
        self.use_compressed = compressed
        self.dt = 1.0 / float(control_hz)
        self.chunk_size = int(chunk_size)
        self.prefetch_remaining = int(prefetch_remaining)
        self.min_exec_steps = int(min_exec_steps)
        self.img_size = int(img_size)
        self.max_vel = float(max_vel_rad)
        self.vel_scale = float(vel_scale)
        self.cmd_mode = cmd_mode
        self.gripper_action = gripper_action
        self.traj_goal_time_tol = float(traj_goal_time_tol)
        self.traj_min_segment_time = float(traj_min_segment_time)

        # Data caches
        self.latest_ext: Optional[np.ndarray] = None
        self.latest_wrist: Optional[np.ndarray] = None
        self.latest_q: Optional[np.ndarray] = None     # (7,)
        self.latest_g: Optional[float] = None
        self.latest_joint_names: List[str] = []

        self.latest_chunk : Optional[np.ndarray] = None  # (N,8)
        self.desired_order = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7", self.gripper_name]


        # Subscriptions
        if self.use_compressed:
            self.create_subscription(CompressedImage, self.img_topic_ext, self._cb_ext_compressed, SENSOR_QOS)
            self.create_subscription(CompressedImage, self.img_topic_wrist, self._cb_wrist_compressed, SENSOR_QOS)
        else:
            self.create_subscription(RosImage, self.img_topic_ext, self._cb_ext_raw, SENSOR_QOS)
            self.create_subscription(RosImage, self.img_topic_wrist, self._cb_wrist_raw, SENSOR_QOS)
        self.create_subscription(JointState, self.joint_topic, self._cb_joint, SENSOR_QOS)

        # Double-buffer state for action chunks
        self._curr_chunk: Optional[np.ndarray] = None  # (N,8)
        self._curr_idx: int = 0
        self._prefetch_future: Optional[concurrent.futures.Future] = None
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Trajectory action clients (used only in trajectory mode)
        self._traj_client = ActionClient(self, FollowJointTrajectory, traj_action_ns)
        self._grip_client = ActionClient(self, GripperCommand, grip_action_ns)

        # Timer loop
        self.timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info(
            f"OpenPI client @{host}:{port} | {1.0/self.dt:.1f} Hz | chunk={self.chunk_size} | "
            f"prefetch_when_remaining<={self.prefetch_remaining} | min_exec_steps={self.min_exec_steps} | "
            f"cmd_mode={self.cmd_mode} | gripper_action={self.gripper_action} | compressed={self.use_compressed}"
        )

    # --- ROS callbacks ---
    def _cb_ext_raw(self, msg: RosImage):
        try: self.latest_ext = resize_with_pad_uint8(to_rgb_from_raw_msg(msg), self.img_size)
        except Exception as e: self.get_logger().warn(f"ext raw decode error: {e}")

    def _cb_wrist_raw(self, msg: RosImage):
        try: self.latest_wrist = resize_with_pad_uint8(to_rgb_from_raw_msg(msg), self.img_size)
        except Exception as e: self.get_logger().warn(f"wrist raw decode error: {e}")

    def _cb_ext_compressed(self, msg: CompressedImage):
        try: self.latest_ext = resize_with_pad_uint8(to_rgb_from_compressed_msg(msg), self.img_size)
        except Exception as e: self.get_logger().warn(f"ext compressed decode error: {e}")



    def _cb_wrist_compressed(self, msg: CompressedImage):
        try: self.latest_wrist = resize_with_pad_uint8(to_rgb_from_compressed_msg(msg), self.img_size)
        except Exception as e: self.get_logger().warn(f"wrist compressed decode error: {e}")

    def make_index_map(self, current_names, desired_order):
        idx = {n: i for i, n in enumerate(current_names)}
        return [idx.get(n, -1) for n in desired_order]


    def reorder_joint_state_msg(self, msg, desired_order):
        idx_map = self.make_index_map(msg.name, desired_order)

        def pick(arr, i):
            if i < 0 or i >= len(arr):
                return math.nan
            return arr[i]

        msg.name     = [desired_order[j] for j in range(len(idx_map))]
        msg.position = [pick(msg.position, di) for di in idx_map]
        msg.velocity = [pick(msg.velocity, di) for di in idx_map]
        msg.effort   = [pick(msg.effort,   di) for di in idx_map]
        return msg

    def _cb_joint(self, msg: JointState):
        # self.get_logger().info(f"✅ joint callback fired with {len(msg.name)} joints")
        try:
            js = self.reorder_joint_state_msg(msg, self.desired_order)
            names = list(js.name)
            #print(names, file=sys.stderr)
            pos = np.asarray(js.position, dtype=np.float32)
            pos = np.nan_to_num(pos, nan=0.0)
            if self.gripper_name in names:
                gi = names.index(self.gripper_name)
                arm_mask = np.ones(len(names), dtype=bool); arm_mask[gi] = False
                q7 = pos[arm_mask][:7]
                g1 = float(np.nan_to_num(pos[gi], nan=0.0))
            else:
                q7 = pos[:7]; g1 = 0.0
            #self.latest_q = wrap_to_pi(q7.astype(np.float32))
            self.latest_q = q7.astype(np.float32)
            self.latest_g = g1
            # key based joint names (excluding gripper)
            self.latest_joint_names = [n for n in names if n != self.gripper_name]
     

        except Exception as e:
            self.get_logger().warn(f"joint parse error: {e}")

    # --- Control loop with swap-when-ready (min_exec_steps) ---
    def _tick(self):
        if self.latest_q is None or self.latest_g is None or self.latest_ext is None or self.latest_wrist is None:
            if self.latest_ext is None:
                self.get_logger().info("Waiting for external image…")
            if self.latest_wrist is None:
                self.get_logger().info("Waiting for wrist image…")
            if self.latest_q is None:
                self.get_logger().info("Waiting for joint state…")
            if self.latest_g is None:
                self.get_logger().info("Waiting for gripper state…")

            return

        # Ensure current chunk exists
        if self._curr_chunk is None:
            self._curr_chunk = self._blocking_infer_chunk()
            self._curr_idx = 0
            if self._curr_chunk is None: return

            # If trajectory mode, send trajectory now for (at least) min_exec_steps
            if self.cmd_mode == "trajectory":
                self._send_trajectory_for_window(self._curr_chunk, start_idx=0, n_steps=max(self.min_exec_steps, 1))

        # Execute one step or just time the loop if trajectory is running
        remaining = len(self._curr_chunk) - self._curr_idx

        # Prefetch when small remaining
        if remaining <= self.prefetch_remaining and self._prefetch_future is None:
            self._prefetch_future = self._pool.submit(self._infer_chunk_safe)

        # In log/velocity mode, consume per tick
        if self.cmd_mode in ("log", "velocity"):
            a = self._curr_chunk[self._curr_idx]
            dq = np.clip(a[:7], -self.max_vel, self.max_vel)
            dg = float(np.clip(a[7], -self.max_vel, self.max_vel))

            # --- Safety scaling ---
            dq *= getattr(self, "vel_scale", 0.1)
            dg *= getattr(self, "vel_scale", 0.1)
            # ----------------------
            if self.cmd_mode == "log":
                self.get_logger().info(f"[{self._curr_idx+1}/{len(self._curr_chunk)}] dq(rad/s)={dq.round(3).tolist()} dg={round(dg,3)}")
            elif self.cmd_mode == "velocity":
                # Example: publish to joint_group_velocity_controller
            
                
                # Lazy-create publisher once
                if not hasattr(self, "publisher"):
                    self.publisher = self.create_publisher(
                        JointTrajectory, "/joint_group_velocity_controller/joint_trajectory", 10
                    )

                # Key-based joint order
                if hasattr(self, "latest_joint_names") and self.latest_joint_names:
                    joint_names = self.latest_joint_names
                    dq_map = {f"joint_{i+1}": dq[i] for i in range(min(len(dq), 7))}
                    dq_final = [dq_map.get(jn, 0.0) for jn in joint_names]
                else:
                    joint_names = [f"joint_{i+1}" for i in range(7)]
                    dq_final = dq.tolist()

                msg = JointTrajectory()
                msg.joint_names = joint_names
                pt = JointTrajectoryPoint()
                # pt.velocities = dq_final
                pt.velocities = [float(v) for v in dq_final]
                pt.time_from_start = dur_from_seconds(self.dt)
                msg.points = [pt]
                self.publisher.publish(msg)
                self._curr_idx += 1

        # In trajectory mode, we’re letting the controller run; keep _curr_idx advancing to reflect elapsed time
        elif self.cmd_mode == "trajectory":
            self._curr_idx += 1  

        # Swap when ready AND we’ve executed at least min_exec_steps
        if self._prefetch_future is not None and self._prefetch_future.done() and self._curr_idx >= self.min_exec_steps:
            next_chunk = self._prefetch_future.result()
            self._prefetch_future = None
            if next_chunk is not None and len(next_chunk) > 0:
                self._curr_chunk = next_chunk
                self._curr_idx = 0
                if self.cmd_mode == "trajectory":
                    self._send_trajectory_for_window(self._curr_chunk, start_idx=0, n_steps=max(self.min_exec_steps, 1))
                return

        # If we run out, block to get a fresh one
        if self._curr_idx >= len(self._curr_chunk):
            next_chunk = self._blocking_infer_chunk()
            if next_chunk is not None and len(next_chunk) > 0:
                self.latest_chunk = next_chunk
                self._curr_chunk = next_chunk
                self._curr_idx = 0
                if self.cmd_mode == "trajectory":
                    self._send_trajectory_for_window(self._curr_chunk, start_idx=0, n_steps=max(self.min_exec_steps, 1))

    # --- OpenPI infer helpers ---
    def _build_observation(self) -> dict:
        return {
            "observation/exterior_image_1_left": self.latest_ext,            # uint8 HxWx3
            "observation/wrist_image_left":      self.latest_wrist,          # uint8 HxWx3
            "observation/joint_position":        self.latest_q.tolist(),     # 7 abs rad
            "observation/gripper_position":      [float(self.latest_g)],     # 1 abs
            "prompt": self.prompt,
            "chunk_size": self.chunk_size,
        }

    def _blocking_infer_chunk(self) -> Optional[np.ndarray]:
        try:
            t0 = time.perf_counter()
            result = self.client.infer(self._build_observation())
            #print(result['actions'])
            dt_ms = (time.perf_counter() - t0) * 1000.0
            acts = np.asarray(result.get("actions", []), dtype=np.float32)
            if acts.ndim != 2 or acts.shape[1] != 8:
                self.get_logger().warn(f"Bad action shape {acts.shape}; expected (N,8)")
                return None
            self.get_logger().info(f"infer OK {acts.shape} in {dt_ms:.1f} ms")
            return acts
        except Exception as e:
            self.get_logger().warn(f"infer error: {e}")
            return None

    def _infer_chunk_safe(self) -> Optional[np.ndarray]:
        try:
            result = self.client.infer(self._build_observation())
            acts = np.asarray(result.get("actions", []), dtype=np.float32)
            if acts.ndim == 2 and acts.shape[1] == 8:
                return acts
        except Exception as e:
            self.get_logger().warn(f"prefetch infer error: {e}")
        return None

    # --- Trajectory building & sending ---

    def _sanitize_segment(self, q_curr: List[float], q_next: List[float]) -> List[float]:
        """Shortest angular path joint-by-joint (like your sanitize_next_joint_angles)."""
        out = []
        for c, t in zip(q_curr, q_next):
            delta = t - c
            while delta > math.pi:  delta -= 2 * math.pi
            while delta < -math.pi: delta += 2 * math.pi
            out.append(c + delta)
        return out

    def _send_trajectory_for_window(self, chunk: np.ndarray, start_idx: int, n_steps: int):
        """Convert [start_idx : start_idx+n_steps] velocities into a short trajectory and send it."""
        if self.latest_q is None: return
        # Clamp window
        end_idx = min(start_idx + n_steps, len(chunk))
        window = chunk[start_idx:end_idx]  # shape (K, 8)
        if window.size == 0: return

        # print(f"[TRAJ] dq window mean={window.mean():.4f}, std={window.std():.4f}")
        # print(f"[TRAJ] first row={np.round(window[0], 4)}")


        print(f"Sending trajectory for steps {start_idx} to {end_idx} (total {len(chunk)})")

        # Current arm pose
        q0 = self.latest_q.astype(np.float64).tolist()   # 7
        g0 = float(self.latest_g)

        # Integrate velocities -> absolute positions over K steps
        q_points: List[List[float]] = []
        g_points: List[float] = []

        q_curr = q0[:]
        g_curr = g0
        tfs_list: List[float] = []
        t_accum = 0.0

        for i, a in enumerate(window):
            dq = np.clip(a[:7], -self.max_vel, self.max_vel)
            if self.gripper_action == "velocity":
                dg = float(np.clip(a[7], -self.max_vel, self.max_vel))
                g_next = g_curr + dg * self.dt
            else:  # position
                g_next = float(a[7])

            # integrate joints
            q_next = (np.array(q_curr) + dq * self.dt).astype(np.float64)


            #SANITIZATION CHANGE
            #q_next = wrap_to_pi(q_next)
            
            # sanitize shortest path from q_curr to q_next (belt & suspenders)
            q_next = np.array(self._sanitize_segment(q_curr, q_next.tolist()), dtype=np.float64)

            q_points.append(q_next.tolist())
            g_points.append(g_next)
            q_curr = q_next.tolist()
            g_curr = g_next

            t_accum += max(self.traj_min_segment_time, self.dt*30)  # guarantee a minimum segment time
            tfs_list.append(t_accum)

        # Build JointTrajectory (ARM ONLY; gripper via separate action)
        # Joint order: use the *current* joint state order (excluding gripper)
        # We try to read names from the last /joint_states (minus the gripper joint)
        # Fallback: fixed names if needed.
        arm_joint_names = []
        # We don't keep names cache, so we fetch from /joint_states by reading last message
        # In ROS2 Python, JointState doesn't persist names here; we infer from last message in _cb_joint
        # We'll construct names as j0..j6 if unknown; controllers usually require exact names.
        # You likely want to pass --arm-joint-names explicitly in practice. For now, we try best effort.
        # (To keep the script self-contained, we ask the controller to accept order as configured.)
        # If you have exact names, add a CLI flag and set them here.

        # Try to sniff names from /joint_states on the fly (not stored). We'll just not set names here.
        # Better: ask user to set --arm-joint-names. To keep simple, we leave names empty to use controller defaults.
        # However, FollowJointTrajectory REQUIRES joint_names; so we must set something.
        # We'll warn if we can't infer; user should edit names to match controller.
        # Common Kinova Gen3 order (example; replace with your exact mapping):
        default_names = [
            "joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7"
        ]
        arm_joint_names = default_names


        traj = JointTrajectory()
        traj.joint_names = arm_joint_names
        for q, tfs in zip(q_points, tfs_list):
            pt = JointTrajectoryPoint()
            #pt.positions = wrap_to_pi(np.array(q)).tolist()
            pt.positions = q
            print(pt.positions)
            
            #print(self.latest_chunk)
            #print("hello")
            pt.time_from_start = dur_from_seconds(tfs)
            traj.points.append(pt)

        # Send trajectory goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        goal.goal_time_tolerance = dur_from_seconds(self.traj_goal_time_tol*30)

        if not self._traj_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("FollowJointTrajectory action server not available.")
            return

        send_future = self._traj_client.send_goal_async(goal)
        # Fire & forget; we don't block the loop. If you want, attach callbacks:
        def _on_goal(fut):
            gh = fut.result()
            if gh and gh.accepted:
                self.get_logger().info(f"Trajectory goal accepted ({len(traj.points)} pts, {tfs_list[-1]:.2f}s).")
            else:
                self.get_logger().warn("Trajectory goal rejected.")
        send_future.add_done_callback(_on_goal)

        # Sync gripper (simple: send last g target)
        # If you need per-point gripper sync, you can schedule multiple gripper actions along the way.
        try:
            if not self._grip_client.wait_for_server(timeout_sec=0.5):
                self.get_logger().warn("Gripper action server not available.")
                return
            g_target = float(np.clip(g_points[-1], 0.0, 1.0))
            g_goal = GripperCommand.Goal()
            g_goal.command.position = g_target
            g_goal.command.max_effort = 25.0
            self._grip_client.send_goal_async(g_goal)
        except Exception as e:
            self.get_logger().warn(f"Gripper send error: {e}")


# ---------- Entrypoint ----------

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
