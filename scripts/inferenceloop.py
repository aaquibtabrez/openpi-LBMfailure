#!/usr/bin/env python3
"""
PiZero / OpenPI ROS 2 client with double-buffer action chunks and optional trajectory execution.

Modes:
 --cmd-mode log : print actions (safe default)
 --cmd-mode velocity : (stub) stream dq (rad/s) to your velocity controller (wire later)
 --cmd-mode trajectory : convert each predicted velocity chunk to a short FollowJointTrajectory goal
 and sync Robotiq gripper over GripperCommand action.

Trajectory control references your example:
 - /joint_trajectory_controller/follow_joint_trajectory (control_msgs/FollowJointTrajectory)
 - /robotiq_gripper_controller/gripper_cmd (control_msgs/GripperCommand)

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
# reliability=QoSReliabilityPolicy.BEST_EFFORT,
# history=QoSHistoryPolicy.KEEP_LAST,
# depth=5,
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
        cmd_mode: str,  # "log" | "velocity" | "trajectory"
        gripper_action: str,  # "velocity" | "position" (what the policy outputs)
        traj_action_ns: str,  # /joint_trajectory_controller/follow_joint_trajectory
        grip_action_ns: str,  # /robotiq_gripper_controller/gripper_cmd
        traj_goal_time_tol: float,
        traj_min_segment_time: float,  # seconds per point (lower bound)
        vel_scale: float, 
        steps_per_infer: int,
        speed_scale: float,
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
        
        self.chunk_size = int(chunk_size)
        self.prefetch_remaining = int(prefetch_remaining)
        self.min_exec_steps = int(min_exec_steps)
        self.img_size = int(img_size)
        self.max_vel = float(max_vel_rad)
        self.vel_scale = float(vel_scale)
        self.steps_per_infer = max(1, int(steps_per_infer))
        self.speed_scale = max(1e-6, float(speed_scale))  # avoid divide-by-zero
        self.dt = 1.0 / float(control_hz) 
        self.cmd_mode = cmd_mode
        self.gripper_action = gripper_action
        self.traj_goal_time_tol = float(traj_goal_time_tol)
        self.traj_min_segment_time = float(traj_min_segment_time)

        # Data caches
        self.latest_ext: Optional[np.ndarray] = None
        self.latest_wrist: Optional[np.ndarray] = None
        self.latest_q: Optional[np.ndarray] = None  # (7,)
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
        # self._prefetch_future: Optional[concurrent.futures.Future] = None
        # self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

        # Trajectory action clients (used only in trajectory mode)
        self._traj_client = ActionClient(self, FollowJointTrajectory, traj_action_ns)
        self._grip_client = ActionClient(self, GripperCommand, grip_action_ns)

        # Trajectory state for gating new inferences
        self._traj_active = False          # True while a FollowJointTrajectory goal is running
        self._traj_active_until: Optional[float] = None  # Fallback timeout (wall-clock time)

        # Timer loop
        self.timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info(
            f"OpenPI client @{host}:{port} | {1.0/self.dt:.1f} Hz | chunk={self.chunk_size} | "
            f"prefetch_when_remaining<={self.prefetch_remaining} | min_exec_steps={self.min_exec_steps} | "
            f"cmd_mode={self.cmd_mode} | gripper_action={self.gripper_action} | compressed={self.use_compressed}"
        )

    # --- ROS callbacks ---
    def _cb_ext_raw(self, msg: RosImage):
        try:
            self.latest_ext = resize_with_pad_uint8(to_rgb_from_raw_msg(msg), self.img_size)
        except Exception as e:
            self.get_logger().warn(f"ext raw decode error: {e}")

    def _cb_wrist_raw(self, msg: RosImage):
        try:
            self.latest_wrist = resize_with_pad_uint8(to_rgb_from_raw_msg(msg), self.img_size)
        except Exception as e:
            self.get_logger().warn(f"wrist raw decode error: {e}")

    def _cb_ext_compressed(self, msg: CompressedImage):
        try:
            self.latest_ext = resize_with_pad_uint8(to_rgb_from_compressed_msg(msg), self.img_size)
        except Exception as e:
            self.get_logger().warn(f"ext compressed decode error: {e}")

    def _cb_wrist_compressed(self, msg: CompressedImage):
        try:
            self.latest_wrist = resize_with_pad_uint8(to_rgb_from_compressed_msg(msg), self.img_size)
        except Exception as e:
            self.get_logger().warn(f"wrist compressed decode error: {e}")

    def make_index_map(self, current_names, desired_order):
        idx = {n: i for i, n in enumerate(current_names)}
        return [idx.get(n, -1) for n in desired_order]

    def reorder_joint_state_msg(self, msg, desired_order):
        idx_map = self.make_index_map(msg.name, desired_order)

        def pick(arr, i):
            if i < 0 or i >= len(arr):
                return math.nan
            return arr[i]

        msg.name = [desired_order[j] for j in range(len(idx_map))]
        msg.position = [pick(msg.position, di) for di in idx_map]
        msg.velocity = [pick(msg.velocity, di) for di in idx_map]
        msg.effort = [pick(msg.effort, di) for di in idx_map]
        return msg

    def _cb_joint(self, msg: JointState):
        # self.get_logger().info(f"✅ joint callback fired with {len(msg.name)} joints")
        try:
            js = self.reorder_joint_state_msg(msg, self.desired_order)
            names = list(js.name)
            # print(names, file=sys.stderr)
            pos = np.asarray(js.position, dtype=np.float32)
            pos = np.nan_to_num(pos, nan=0.0)
            if self.gripper_name in names:
                gi = names.index(self.gripper_name)
                arm_mask = np.ones(len(names), dtype=bool); arm_mask[gi] = False
                q7 = pos[arm_mask][:7]
                g1 = float(np.nan_to_num(pos[gi], nan=0.0))
            else:
                q7 = pos[:7]; g1 = 0.0
            # self.latest_q = wrap_to_pi(q7.astype(np.float32))
            self.latest_q = q7.astype(np.float32)
            self.latest_g = g1
            # key based joint names (excluding gripper)
            self.latest_joint_names = [n for n in names if n != self.gripper_name]
        
        except Exception as e:
            self.get_logger().warn(f"joint parse error: {e}")

    # --- Control loop with swap-when-ready (min_exec_steps) ---
    def _tick(self):
        # Wait until we have all sensors
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

        # --- Trajectory mode: don't infer while a trajectory is still running ---
        if self.cmd_mode == "trajectory":
            now = time.monotonic()

            # If action reports "still active", just wait.
            if self._traj_active:
                return

            # Fallback: if we have an expected end time, also gate until then.
            if self._traj_active_until is not None:
                if now < self._traj_active_until:
                    return
                else:
                    # Safety: timeout elapsed, clear it so we can infer again
                    self._traj_active_until = None
        # ------------------------------------------------------------------------

        # If no current chunk, do a blocking inference now.
        if self._curr_chunk is None:
            self._curr_chunk = self._blocking_infer_chunk()
            self._curr_idx = 0
            if self._curr_chunk is None:
                return

            if self.cmd_mode == "trajectory":
                # Only execute the first window as a single short trajectory.
                window_n = max(1, min(self.steps_per_infer, len(self._curr_chunk)))
                total_traj_time = self._send_trajectory_for_window(
                    self._curr_chunk, start_idx=0, n_steps=window_n
                )

                # Fallback timeout based on planned trajectory duration.
                if total_traj_time is not None and total_traj_time > 0.0:
                    self._traj_active_until = time.monotonic() + total_traj_time + self.traj_goal_time_tol

                # We’re done with this chunk; next inference will only happen
                # once the trajectory completes (see gating at top of _tick).
                self._curr_chunk = None
                self._curr_idx = 0
                return

        # Safety check
        if self._curr_chunk is None or len(self._curr_chunk) == 0:
            self._curr_chunk = None
            return

        # Consume exactly one step per tick in log/velocity mode
        if self.cmd_mode in ("log", "velocity"):
            if self._curr_idx >= len(self._curr_chunk):
                # finished the window: force the next inference
                self._curr_chunk = None
                return

            a = self._curr_chunk[self._curr_idx]
            dq = np.clip(a[:7], -self.max_vel, self.max_vel)
            dg = float(np.clip(a[7], -self.max_vel, self.max_vel))

            # --- Speed control ---
            # Combine existing vel_scale with speed_scale.
            # speed_scale < 1.0 => slower (smaller commands)
            # speed_scale > 1.0 => faster (larger commands)
            eff_scale = float(self.vel_scale) * float(self.speed_scale)
            dq *= eff_scale
            dg *= eff_scale
            # ---------------------

            if self.cmd_mode == "log":
                self.get_logger().info(
                    f"[{self._curr_idx+1}/{len(self._curr_chunk)}] dq(rad/s)={dq.round(3).tolist()} dg={round(dg,3)}"
                )
            else:  # velocity
                # Lazy-create publisher once
                if not hasattr(self, "publisher"):
                    self.publisher = self.create_publisher(
                        JointTrajectory, "/joint_group_velocity_controller/joint_trajectory", 10
                    )

                # Key-based joint order
                if hasattr(self, "latest_joint_names") and self.latest_joint_names:
                    joint_names = self.latest_joint_names
                    dq_map = {f"joint_{i+1}": dq[i] for i in range(min(len(dq), 7))}
                    dq_final = [float(dq_map.get(jn, 0.0)) for jn in joint_names]
                else:
                    joint_names = [f"joint_{i+1}" for i in range(7)]
                    dq_final = [float(v) for v in dq.tolist()]

                msg = JointTrajectory()
                msg.joint_names = joint_names
                pt = JointTrajectoryPoint()
                pt.velocities = dq_final
                # Keep ROS timer cadence; durations here are per-point expectations.
                # If you want the controller itself to move slower/faster, prefer speed_scale above.
                pt.time_from_start = dur_from_seconds(self.dt)
                msg.points = [pt]
                self.publisher.publish(msg)

            self._curr_idx += 1

            # If we’ve executed steps_per_infer steps (or finished the chunk), force next infer
            if self._curr_idx >= min(self.steps_per_infer, len(self._curr_chunk)):
                self._curr_chunk = None  # next tick will infer again
                # (optional) reset idx for clarity
                self._curr_idx = 0

    # --- OpenPI infer helpers ---
    def _build_observation(self) -> dict:
        return {
            "observation/exterior_image_1_left": self.latest_ext,  # uint8 HxWx3
            "observation/wrist_image_left": self.latest_wrist,  # uint8 HxWx3
            "observation/joint_position": self.latest_q.tolist(),  # 7 abs rad
            "observation/gripper_position": [float(self.latest_g)],  # 1 abs
            "prompt": self.prompt,
            "chunk_size": self.chunk_size,
        }

    def _blocking_infer_chunk(self) -> Optional[np.ndarray]:
        try:
            t0 = time.perf_counter()
            result = self.client.infer(self._build_observation())
            # print(result['actions'])
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

    # --- Trajectory building & sending ---

    def _sanitize_segment(self, q_curr: List[float], q_next: List[float]) -> List[float]:
        """Shortest angular path joint-by-joint (like your sanitize_next_joint_angles)."""
        out = []
        for c, t in zip(q_curr, q_next):
            delta = t - c
            while delta > math.pi:
                delta -= 2 * math.pi
            while delta < -math.pi:
                delta += 2 * math.pi
            out.append(c + delta)
        return out

    def _send_trajectory_for_window(self, chunk: np.ndarray, start_idx: int, n_steps: int) -> Optional[float]:
        """Convert [start_idx : start_idx+n_steps] velocities into a short trajectory and send it."""
        if self.latest_q is None:
            return None
        # Clamp window
        end_idx = min(start_idx + n_steps, len(chunk))
        window = chunk[start_idx:end_idx]  # shape (K, 8)
        if window.size == 0:
            return None

        print(f"Sending trajectory for steps {start_idx} to {end_idx} (total {len(chunk)})")

        # Current arm pose
        q0 = self.latest_q.astype(np.float64).tolist()  # 7
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
            #q_next = wrap_to_pi(q_next)
            # sanitize shortest path from q_curr to q_next (belt & suspenders)
            #q_next = np.array(self._sanitize_segment(q_curr, q_next.tolist()), dtype=np.float64)

            q_points.append(q_next.tolist())
            g_points.append(g_next)
            q_curr = q_next.tolist()
            g_curr = g_next

            # NEW (seconds, scaled):
            base_seg = max(self.traj_min_segment_time, self.dt)
            # If speed_scale < 1.0 (slower), we make segments longer by 1/speed_scale.
            seg_time = base_seg / self.speed_scale
            t_accum += seg_time
            tfs_list.append(t_accum)

        if not tfs_list:
            return None
        total_traj_time = float(tfs_list[-1])

        # Build JointTrajectory (ARM ONLY; gripper via separate action)
        default_names = [
            "joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7"
        ]
        arm_joint_names = default_names

        traj = JointTrajectory()
        traj.joint_names = arm_joint_names
        for q, tfs in zip(q_points, tfs_list):
            pt = JointTrajectoryPoint()
            pt.positions = q
            #print(pt.positions)
            pt.time_from_start = dur_from_seconds(tfs)
            traj.points.append(pt)

        # Send trajectory goal
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        goal.goal_time_tolerance = dur_from_seconds(self.traj_goal_time_tol)

        if not self._traj_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("FollowJointTrajectory action server not available.")
            return None

        # Mark trajectory as active *before* sending, so _tick will gate.
        self._traj_active = True

        send_future = self._traj_client.send_goal_async(goal)

        def _on_goal(fut):
            try:
                gh = fut.result()
            except Exception as e:
                self.get_logger().warn(f"Trajectory send exception: {e}")
                self._traj_active = False
                return

            if not gh or not gh.accepted:
                self.get_logger().warn("Trajectory goal rejected.")
                self._traj_active = False
                return

            self.get_logger().info(
                f"Trajectory goal accepted ({len(traj.points)} pts, {tfs_list[-1]:.2f}s)."
            )

            # When the controller finishes executing the trajectory, this result callback fires.
            result_future = gh.get_result_async()

            def _on_result(rf):
                try:
                    res = rf.result()
                    self.get_logger().info(f"Trajectory finished with status {res.status}.")
                except Exception as e:
                    self.get_logger().warn(f"Trajectory result exception: {e}")
                # Clear active flags so _tick can infer again.
                self._traj_active = False
                self._traj_active_until = None

            result_future.add_done_callback(_on_result)

        send_future.add_done_callback(_on_goal)

        # Sync gripper (simple: send last g target)
        # If you need per-point gripper sync, you can schedule multiple gripper actions along the way.
        try:
            if not self._grip_client.wait_for_server(timeout_sec=0.5):
                self.get_logger().warn("Gripper action server not available.")
                return total_traj_time
            g_target = float(np.clip(g_points[-1], 0.0, 1.0))
            g_goal = GripperCommand.Goal()
            g_goal.command.position = g_target
            g_goal.command.max_effort = 25.0
            self._grip_client.send_goal_async(g_goal)
        except Exception as e:
            self.get_logger().warn(f"Gripper send error: {e}")

        return total_traj_time


# ---------- Entrypoint ----------

def main():
    ap = argparse.ArgumentParser("PiZero / OpenPI client with optional FollowJointTrajectory control")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--joint-topic", default="/joint_states")
    ap.add_argument("--gripper-name", default="robotiq_85_left_knuckle_joint")
    ap.add_argument("--compressed", action="store_true", help="Use sensor_msgs/CompressedImage for cameras")

    ap.add_argument("--control-hz", type=float, default=10.0)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--prefetch-remaining", type=int, default=5, help="Start prefetch when <= this many remain")
    ap.add_argument("--min-exec-steps", type=int, default=4, help="Must execute at least this many steps before swapping")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-vel-rad", type=float, default=1.0, help="Clamp on returned joint velocities (rad/s)")

    ap.add_argument("--prompt", default="do something")

    # ap.add_argument("--image-topic-external", default="/camera1/camera1/color/image_raw")
    # ap.add_argument("--image-topic-wrist", default="/camera2/camera2/color/image_raw")
    ap.add_argument("--image-topic-external", default="/camera1/camera1/color/image_raw/compressed")
    ap.add_argument("--image-topic-wrist", default="/camera2/camera2/color/image_raw/compressed")

    ap.add_argument("--cmd-mode", choices=["log","velocity","trajectory"], default="log",
                    help="How to consume actions each tick")
    ap.add_argument("--gripper-action", choices=["velocity","position"], default="velocity",
                    help="What the policy's 8th channel means")

    ap.add_argument("--traj-action-ns", default="/joint_trajectory_controller/follow_joint_trajectory")
    ap.add_argument("--grip-action-ns", default="/robotiq_gripper_controller/gripper_cmd")
    ap.add_argument("--traj-goal-time-tol", type=float, default=0.25)
    ap.add_argument("--traj-min-segment-time", type=float, default=0.10, help="Minimum time per trajectory segment (s)")

    ap.add_argument("--vel-scale", type=float, default=0.1, help="Scale factor for joint/gripper velocities (safety)")
    ap.add_argument("--steps-per-infer", type=int, default=4,
                    help="How many actions to execute per inference cycle")
    ap.add_argument("--speed-scale", type=float, default=1.0,
                    help="Overall speed multiplier. <1.0 = slower, >1.0 = faster")

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
        steps_per_infer=args.steps_per_infer,
        speed_scale=args.speed_scale,
    )
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
