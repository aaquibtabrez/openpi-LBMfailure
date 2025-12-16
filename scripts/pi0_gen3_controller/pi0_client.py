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

from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import JointState

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory, GripperCommand
from rclpy.action import ActionClient

# OpenPI client + utils (unchanged)
from openpi_client import image_tools
from openpi_client import websocket_client_policy

from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Base_pb2, BaseCyclic_pb2

import time

# bring in the exact helper names/constants your original file used
# from helpers import (
#     SENSOR_QOS,
#     wrap_to_pi,
#     to_rgb_from_raw_msg,
#     to_rgb_from_compressed_msg,
#     resize_with_pad_uint8,
#     dur_from_seconds,
# )

# class PiZeroOpenPIClient(Node):
#     def __init__(
#         self,
#         *,
#         host: str,
#         port: int,
#         prompt: str,
#         img_topic_ext: str,
#         img_topic_wrist: str,
#         joint_topic: str,
#         gripper_name: str,
#         compressed: bool,
#         control_hz: float,
#         chunk_size: int,
#         prefetch_remaining: int,
#         min_exec_steps: int,
#         img_size: int,
#         max_vel_rad: float,
#         cmd_mode: str,  # "log" | "velocity" | "trajectory"
#         gripper_action: str,  # "velocity" | "position" (what the policy outputs)
#         traj_action_ns: str,  # /joint_trajectory_controller/follow_joint_trajectory
#         grip_action_ns: str,  # /robotiq_gripper_controller/gripper_cmd
#         traj_goal_time_tol: float,
#         traj_min_segment_time: float,  # seconds per point (lower bound)
#         vel_scale: float, 
#         steps_per_infer: int,
#         speed_scale: float,
#     ):
#         super().__init__("pizero_openpi_traj_client")

#         # OpenPI client
#         self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

#         # Config
#         self.prompt = prompt
#         self.img_topic_ext = img_topic_ext
#         self.img_topic_wrist = img_topic_wrist
#         self.joint_topic = joint_topic
#         self.gripper_name = gripper_name
#         self.use_compressed = compressed
        
#         self.chunk_size = int(chunk_size)
#         self.prefetch_remaining = int(prefetch_remaining)
#         self.min_exec_steps = int(min_exec_steps)
#         self.img_size = int(img_size)
#         self.max_vel = float(max_vel_rad)
#         self.vel_scale = float(vel_scale)
#         self.steps_per_infer = max(1, int(steps_per_infer))
#         self.speed_scale = max(1e-6, float(speed_scale))  # avoid divide-by-zero
#         self.dt = 1.0 / float(control_hz) 
#         self.cmd_mode = cmd_mode
#         self.gripper_action = gripper_action
#         self.traj_goal_time_tol = float(traj_goal_time_tol)
#         self.traj_min_segment_time = float(traj_min_segment_time)

#         # Data caches
#         self.latest_ext: Optional[np.ndarray] = None
#         self.latest_wrist: Optional[np.ndarray] = None
#         self.latest_q: Optional[np.ndarray] = None  # (7,)
#         self.latest_g: Optional[float] = None
#         self.latest_joint_names: List[str] = []

#         self.latest_chunk : Optional[np.ndarray] = None  # (N,8)
#         self.desired_order = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7", self.gripper_name]

#         # Subscriptions
#         if self.use_compressed:
#             self.create_subscription(CompressedImage, self.img_topic_ext, self._cb_ext_compressed, SENSOR_QOS)
#             self.create_subscription(CompressedImage, self.img_topic_wrist, self._cb_wrist_compressed, SENSOR_QOS)
#         else:
#             self.create_subscription(RosImage, self.img_topic_ext, self._cb_ext_raw, SENSOR_QOS)
#             self.create_subscription(RosImage, self.img_topic_wrist, self._cb_wrist_raw, SENSOR_QOS)
#         self.create_subscription(JointState, self.joint_topic, self._cb_joint, SENSOR_QOS)

#         # Double-buffer state for action chunks
#         self._curr_chunk: Optional[np.ndarray] = None  # (N,8)
#         self._curr_idx: int = 0
#         # self._prefetch_future: Optional[concurrent.futures.Future] = None
#         # self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

#         # Trajectory action clients (used only in trajectory mode)
#         self._traj_client = ActionClient(self, FollowJointTrajectory, traj_action_ns)
#         self._grip_client = ActionClient(self, GripperCommand, grip_action_ns)

#         # Trajectory state for gating new inferences
#         self._traj_active = False          # True while a FollowJointTrajectory goal is running
#         self._traj_active_until: Optional[float] = None  # Fallback timeout (wall-clock time)

#         # Timer loop
#         self.timer = self.create_timer(self.dt, self._tick)

#         self.get_logger().info(
#             f"OpenPI client @{host}:{port} | {1.0/self.dt:.1f} Hz | chunk={self.chunk_size} | "
#             f"prefetch_when_remaining<={self.prefetch_remaining} | min_exec_steps={self.min_exec_steps} | "
#             f"cmd_mode={self.cmd_mode} | gripper_action={self.gripper_action} | compressed={self.use_compressed}"
#         )

#     # --- ROS callbacks ---
#     def _cb_ext_raw(self, msg: RosImage):
#         try:
#             self.latest_ext = resize_with_pad_uint8(to_rgb_from_raw_msg(msg), self.img_size)
#         except Exception as e:
#             self.get_logger().warn(f"ext raw decode error: {e}")

#     def _cb_wrist_raw(self, msg: RosImage):
#         try:
#             self.latest_wrist = resize_with_pad_uint8(to_rgb_from_raw_msg(msg), self.img_size)
#         except Exception as e:
#             self.get_logger().warn(f"wrist raw decode error: {e}")

#     def _cb_ext_compressed(self, msg: CompressedImage):
#         try:
#             self.latest_ext = resize_with_pad_uint8(to_rgb_from_compressed_msg(msg), self.img_size)
#         except Exception as e:
#             self.get_logger().warn(f"ext compressed decode error: {e}")

#     def _cb_wrist_compressed(self, msg: CompressedImage):
#         try:
#             self.latest_wrist = resize_with_pad_uint8(to_rgb_from_compressed_msg(msg), self.img_size)
#         except Exception as e:
#             self.get_logger().warn(f"wrist compressed decode error: {e}")

#     def make_index_map(self, current_names, desired_order):
#         idx = {n: i for i, n in enumerate(current_names)}
#         return [idx.get(n, -1) for n in desired_order]

#     def reorder_joint_state_msg(self, msg, desired_order):
#         idx_map = self.make_index_map(msg.name, desired_order)

#         def pick(arr, i):
#             if i < 0 or i >= len(arr):
#                 return math.nan
#             return arr[i]

#         msg.name = [desired_order[j] for j in range(len(idx_map))]
#         msg.position = [pick(msg.position, di) for di in idx_map]
#         msg.velocity = [pick(msg.velocity, di) for di in idx_map]
#         msg.effort = [pick(msg.effort, di) for di in idx_map]
#         return msg

#     def _cb_joint(self, msg: JointState):
#         # self.get_logger().info(f"✅ joint callback fired with {len(msg.name)} joints")
#         try:
#             js = self.reorder_joint_state_msg(msg, self.desired_order)
#             names = list(js.name)
#             # print(names, file=sys.stderr)
#             pos = np.asarray(js.position, dtype=np.float32)
#             pos = np.nan_to_num(pos, nan=0.0)
#             if self.gripper_name in names:
#                 gi = names.index(self.gripper_name)
#                 arm_mask = np.ones(len(names), dtype=bool); arm_mask[gi] = False
#                 q7 = pos[arm_mask][:7]
#                 g1 = float(np.nan_to_num(pos[gi], nan=0.0))
#             else:
#                 q7 = pos[:7]; g1 = 0.0
#             # self.latest_q = wrap_to_pi(q7.astype(np.float32))
#             self.latest_q = q7.astype(np.float32)
#             self.latest_g = g1
#             # key based joint names (excluding gripper)
#             self.latest_joint_names = [n for n in names if n != self.gripper_name]
        
#         except Exception as e:
#             self.get_logger().warn(f"joint parse error: {e}")

#     # --- Control loop with swap-when-ready (min_exec_steps) ---
#     def _tick(self):
#         # Wait until we have all sensors
#         if self.latest_q is None or self.latest_g is None or self.latest_ext is None or self.latest_wrist is None:
#             if self.latest_ext is None:
#                 self.get_logger().info("Waiting for external image…")
#             if self.latest_wrist is None:
#                 self.get_logger().info("Waiting for wrist image…")
#             if self.latest_q is None:
#                 self.get_logger().info("Waiting for joint state…")
#             if self.latest_g is None:
#                 self.get_logger().info("Waiting for gripper state…")
#             return

#         # --- Trajectory mode: don't infer while a trajectory is still running ---
#         if self.cmd_mode == "trajectory":
#             now = time.monotonic()

#             # If action reports "still active", just wait.
#             if self._traj_active:
#                 return

#             # Fallback: if we have an expected end time, also gate until then.
#             if self._traj_active_until is not None:
#                 if now < self._traj_active_until:
#                     return
#                 else:
#                     # Safety: timeout elapsed, clear it so we can infer again
#                     self._traj_active_until = None
#         # ------------------------------------------------------------------------

#         # If no current chunk, do a blocking inference now.
#         if self._curr_chunk is None:
#             self._curr_chunk = self._blocking_infer_chunk()
#             self._curr_idx = 0
#             if self._curr_chunk is None:
#                 return

#             if self.cmd_mode == "trajectory":
#                 # Only execute the first window as a single short trajectory.
#                 window_n = max(1, min(self.steps_per_infer, len(self._curr_chunk)))
#                 total_traj_time = self._send_trajectory_for_window(
#                     self._curr_chunk, start_idx=0, n_steps=window_n
#                 )

#                 # Fallback timeout based on planned trajectory duration.
#                 if total_traj_time is not None and total_traj_time > 0.0:
#                     self._traj_active_until = time.monotonic() + total_traj_time + self.traj_goal_time_tol

#                 # We’re done with this chunk; next inference will only happen
#                 # once the trajectory completes (see gating at top of _tick).
#                 self._curr_chunk = None
#                 self._curr_idx = 0
#                 return

#         # Safety check
#         if self._curr_chunk is None or len(self._curr_chunk) == 0:
#             self._curr_chunk = None
#             return

#         # Consume exactly one step per tick in log/velocity mode
#         if self.cmd_mode in ("log", "velocity"):
#             if self._curr_idx >= len(self._curr_chunk):
#                 # finished the window: force the next inference
#                 self._curr_chunk = None
#                 return

#             a = self._curr_chunk[self._curr_idx]
#             dq = np.clip(a[:7], -self.max_vel, self.max_vel)
#             dg = float(np.clip(a[7], -self.max_vel, self.max_vel))

#             # --- Speed control ---
#             # Combine existing vel_scale with speed_scale.
#             # speed_scale < 1.0 => slower (smaller commands)
#             # speed_scale > 1.0 => faster (larger commands)
#             eff_scale = float(self.vel_scale) * float(self.speed_scale)
#             dq *= eff_scale
#             dg *= eff_scale
#             # ---------------------

#             if self.cmd_mode == "log":
#                 self.get_logger().info(
#                     f"[{self._curr_idx+1}/{len(self._curr_chunk)}] dq(rad/s)={dq.round(3).tolist()} dg={round(dg,3)}"
#                 )
#             else:  # velocity
#                 # Lazy-create publisher once
#                 if not hasattr(self, "publisher"):
#                     self.publisher = self.create_publisher(
#                         JointTrajectory, "/joint_group_velocity_controller/joint_trajectory", 10
#                     )

#                 # Key-based joint order
#                 if hasattr(self, "latest_joint_names") and self.latest_joint_names:
#                     joint_names = self.latest_joint_names
#                     dq_map = {f"joint_{i+1}": dq[i] for i in range(min(len(dq), 7))}
#                     dq_final = [float(dq_map.get(jn, 0.0)) for jn in joint_names]
#                 else:
#                     joint_names = [f"joint_{i+1}" for i in range(7)]
#                     dq_final = [float(v) for v in dq.tolist()]

#                 msg = JointTrajectory()
#                 msg.joint_names = joint_names
#                 pt = JointTrajectoryPoint()
#                 pt.velocities = dq_final
#                 # Keep ROS timer cadence; durations here are per-point expectations.
#                 # If you want the controller itself to move slower/faster, prefer speed_scale above.
#                 pt.time_from_start = dur_from_seconds(self.dt)
#                 msg.points = [pt]
#                 self.publisher.publish(msg)

#             self._curr_idx += 1

#             # If we’ve executed steps_per_infer steps (or finished the chunk), force next infer
#             if self._curr_idx >= min(self.steps_per_infer, len(self._curr_chunk)):
#                 self._curr_chunk = None  # next tick will infer again
#                 # (optional) reset idx for clarity
#                 self._curr_idx = 0

#     # --- OpenPI infer helpers ---
#     def _build_observation(self) -> dict:
#         return {
#             "observation/exterior_image_1_left": self.latest_ext,  # uint8 HxWx3
#             "observation/wrist_image_left": self.latest_wrist,  # uint8 HxWx3
#             "observation/joint_position": self.latest_q.tolist(),  # 7 abs rad
#             "observation/gripper_position": [float(self.latest_g)],  # 1 abs
#             "prompt": self.prompt,
#             "chunk_size": self.chunk_size,
#         }

#     def _blocking_infer_chunk(self) -> Optional[np.ndarray]:
#         try:
#             t0 = time.perf_counter()
#             result = self.client.infer(self._build_observation())
#             # print(result['actions'])
#             dt_ms = (time.perf_counter() - t0) * 1000.0
#             acts = np.asarray(result.get("actions", []), dtype=np.float32)
#             if acts.ndim != 2 or acts.shape[1] != 8:
#                 self.get_logger().warn(f"Bad action shape {acts.shape}; expected (N,8)")
#                 return None
#             self.get_logger().info(f"infer OK {acts.shape} in {dt_ms:.1f} ms")
#             return acts
#         except Exception as e:
#             self.get_logger().warn(f"infer error: {e}")
#             return None

#     # --- Trajectory building & sending ---

#     def _sanitize_segment(self, q_curr: List[float], q_next: List[float]) -> List[float]:
#         """Shortest angular path joint-by-joint (like your sanitize_next_joint_angles)."""
#         out = []
#         for c, t in zip(q_curr, q_next):
#             delta = t - c
#             while delta > math.pi:
#                 delta -= 2 * math.pi
#             while delta < -math.pi:
#                 delta += 2 * math.pi
#             out.append(c + delta)
#         return out

#     def _send_trajectory_for_window(self, chunk: np.ndarray, start_idx: int, n_steps: int) -> Optional[float]:
#         """Convert [start_idx : start_idx+n_steps] velocities into a short trajectory and send it."""
#         if self.latest_q is None:
#             return None
#         # Clamp window
#         end_idx = min(start_idx + n_steps, len(chunk))
#         window = chunk[start_idx:end_idx]  # shape (K, 8)
#         if window.size == 0:
#             return None

#         print(f"Sending trajectory for steps {start_idx} to {end_idx} (total {len(chunk)})")
#         #print(f"len(chunk)})")

#         # Current arm pose
#         q0 = self.latest_q.astype(np.float64).tolist()  # 7
#         g0 = float(self.latest_g)

#         # Integrate velocities -> absolute positions over K steps
#         q_points: List[List[float]] = []
#         g_points: List[float] = []

#         q_curr = q0[:]
#         g_curr = g0
#         tfs_list: List[float] = []
#         t_accum = 0.0

#         for i, a in enumerate(window):
#             dq = np.clip(a[:7], -self.max_vel, self.max_vel)
#             if self.gripper_action == "velocity":
#                 dg = float(np.clip(a[7], -self.max_vel, self.max_vel))
#                 g_next = g_curr + dg * self.dt
#             else:  # position
#                 g_next = float(a[7])

#             # integrate joints
#             q_next = (np.array(q_curr) + dq * self.dt).astype(np.float64)
#             #q_next = wrap_to_pi(q_next)
#             # sanitize shortest path from q_curr to q_next (belt & suspenders)
#             #q_next = np.array(self._sanitize_segment(q_curr, q_next.tolist()), dtype=np.float64)

#             q_points.append(q_next.tolist())
#             g_points.append(g_next)
#             q_curr = q_next.tolist()
#             g_curr = g_next

#             # NEW (seconds, scaled):
#             base_seg = max(self.traj_min_segment_time, self.dt)
#             # If speed_scale < 1.0 (slower), we make segments longer by 1/speed_scale.
#             seg_time = base_seg / self.speed_scale
#             t_accum += seg_time
#             tfs_list.append(t_accum)

#         if not tfs_list:
#             return None
#         total_traj_time = float(tfs_list[-1])

#         # Build JointTrajectory (ARM ONLY; gripper via separate action)
#         default_names = [
#             "joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7"
#         ]
#         arm_joint_names = default_names

#         traj = JointTrajectory()
#         traj.joint_names = arm_joint_names
#         for q, tfs in zip(q_points, tfs_list):
#             pt = JointTrajectoryPoint()
#             pt.positions = q
#             #print(pt.positions)
#             pt.time_from_start = dur_from_seconds(tfs)
#             traj.points.append(pt)

#         # Send trajectory goal
#         goal = FollowJointTrajectory.Goal()
#         goal.trajectory = traj
#         goal.goal_time_tolerance = dur_from_seconds(self.traj_goal_time_tol)

#         if not self._traj_client.wait_for_server(timeout_sec=1.0):
#             self.get_logger().warn("FollowJointTrajectory action server not available.")
#             return None

#         # Mark trajectory as active *before* sending, so _tick will gate.
#         self._traj_active = True

#         send_future = self._traj_client.send_goal_async(goal)

#         def _on_goal(fut):
#             try:
#                 gh = fut.result()
#             except Exception as e:
#                 self.get_logger().warn(f"Trajectory send exception: {e}")
#                 self._traj_active = False
#                 return

#             if not gh or not gh.accepted:
#                 self.get_logger().warn("Trajectory goal rejected.")
#                 self._traj_active = False
#                 return

#             self.get_logger().info(
#                 f"Trajectory goal accepted ({len(traj.points)} pts, {tfs_list[-1]:.2f}s)."
#             )

#             # When the controller finishes executing the trajectory, this result callback fires.
#             result_future = gh.get_result_async()

#             def _on_result(rf):
#                 try:
#                     res = rf.result()
#                     self.get_logger().info(f"Trajectory finished with status {res.status}.")
#                 except Exception as e:
#                     self.get_logger().warn(f"Trajectory result exception: {e}")
#                 # Clear active flags so _tick can infer again.
#                 self._traj_active = False
#                 self._traj_active_until = None

#             result_future.add_done_callback(_on_result)

#         send_future.add_done_callback(_on_goal)

#         # Sync gripper (simple: send last g target)
#         # If you need per-point gripper sync, you can schedule multiple gripper actions along the way.
#         try:
#             if not self._grip_client.wait_for_server(timeout_sec=0.5):
#                 self.get_logger().warn("Gripper action server not available.")
#                 return total_traj_time
#             g_target = float(np.clip(g_points[-1], 0.0, 1.0))
#             g_goal = GripperCommand.Goal()
#             g_goal.command.position = g_target
#             g_goal.command.max_effort = 25.0
#             self._grip_client.send_goal_async(g_goal)
#         except Exception as e:
#             self.get_logger().warn(f"Gripper send error: {e}")

#         return total_traj_time


# def send_joint_velocities(base):
#     command = Base_pb2.JointSpeeds()

#     # Example: 7 DOF Gen3 robot
#     for i in range(7):
#         js = command.joint_speeds.add()
#         js.joint_identifier = i
#         js.value = 0.5   # rad/s
#         js.duration = 0  # execute immediately

#     base.SendJointSpeedsCommand(command)



# class PiZeroOpenPIClient(Node):
#     def __init__(
#         self,
#         *,
#         host: str,
#         port: int,
#         prompt: str,
#         img_topic_ext: str,
#         img_topic_wrist: str,
#         joint_topic: str,
#         gripper_name: str,
#         compressed: bool,
#         control_hz: float,
#         chunk_size: int,
#         prefetch_remaining: int,
#         min_exec_steps: int,
#         img_size: int,
#         max_vel_rad: float,
#         cmd_mode: str,                # "log" | "velocity" | "trajectory"
#         gripper_action: str,          # "velocity" | "position"  (what the policy outputs)
#         traj_action_ns: str,          # /joint_trajectory_controller/follow_joint_trajectory
#         grip_action_ns: str,          # /robotiq_gripper_controller/gripper_cmd
#         traj_goal_time_tol: float,
#         traj_min_segment_time: float, # seconds per point (lower bound)
#         vel_scale: float,  
#         # base: BaseClient = None,              # <-- add
#         # base_cyclic: BaseCyclicClient = None, # <-- add
#     ):
#         super().__init__("pizero_openpi_traj_client")

#         # OpenPI client
#         self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

#         # Config
#         self.prompt = prompt
#         self.img_topic_ext = img_topic_ext
#         self.img_topic_wrist = img_topic_wrist
#         self.joint_topic = joint_topic
#         self.gripper_name = gripper_name
#         self.use_compressed = compressed
#         self.dt = 1.0 / float(control_hz)
#         self.chunk_size = int(chunk_size)
#         self.prefetch_remaining = int(prefetch_remaining)
#         self.min_exec_steps = int(min_exec_steps)
#         self.img_size = int(img_size)
#         self.max_vel = float(max_vel_rad)
#         self.vel_scale = float(vel_scale)
#         self.cmd_mode = cmd_mode

#         # Kortex clients
#         # self.base = base
#         # self.base_cyclic = base_cyclic

#         self.gripper_action = gripper_action
#         self.traj_goal_time_tol = float(traj_goal_time_tol)
#         self.traj_min_segment_time = float(traj_min_segment_time)

#         # Data caches
#         self.latest_ext: Optional[np.ndarray] = None
#         self.latest_wrist: Optional[np.ndarray] = None
#         self.latest_q: Optional[np.ndarray] = None     # (7,)
#         self.latest_g: Optional[float] = None
#         self.latest_joint_names: List[str] = []

#         self.latest_chunk : Optional[np.ndarray] = None  # (N,8)
#         self.desired_order = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7", self.gripper_name]

#         # Subscriptions
#         if self.use_compressed:
#             self.create_subscription(CompressedImage, self.img_topic_ext, self._cb_ext_compressed, SENSOR_QOS)
#             self.create_subscription(CompressedImage, self.img_topic_wrist, self._cb_wrist_compressed, SENSOR_QOS)
#         else:
#             self.create_subscription(RosImage, self.img_topic_ext, self._cb_ext_raw, SENSOR_QOS)
#             self.create_subscription(RosImage, self.img_topic_wrist, self._cb_wrist_raw, SENSOR_QOS)
#         self.create_subscription(JointState, self.joint_topic, self._cb_joint, SENSOR_QOS)

#         # Double-buffer state for action chunks
#         self._curr_chunk: Optional[np.ndarray] = None  # (N,8)
#         self._curr_idx: int = 0
#         self._prefetch_future: Optional[concurrent.futures.Future] = None
#         self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

#         # Trajectory action clients (used only in trajectory mode)
#         self._traj_client = ActionClient(self, FollowJointTrajectory, traj_action_ns)
#         self._grip_client = ActionClient(self, GripperCommand, grip_action_ns)

#         # Timer loop
#         self.timer = self.create_timer(self.dt, self._tick)

#         self.get_logger().info(
#             f"OpenPI client @{host}:{port} | {1.0/self.dt:.1f} Hz | chunk={self.chunk_size} | "
#             f"prefetch_when_remaining<={self.prefetch_remaining} | min_exec_steps={self.min_exec_steps} | "
#             f"cmd_mode={self.cmd_mode} | gripper_action={self.gripper_action} | compressed={self.use_compressed}"
#         )
    
#     # def _send_kortex_joint_velocities(self, dq: np.ndarray):
#     #     """
#     #     Send joint velocities (rad/s) to Kinova Gen3 via Kortex BaseClient.
#     #     Assumes dq has length 7.
#     #     """
#     #     if self.base is None:
#     #         self.get_logger().warn("Kortex BaseClient is not set; cannot send velocities.")
#     #         return

#     #     dq = np.asarray(dq, dtype=np.float64)
#     #     command = Base_pb2.JointSpeeds()

#     #     for i in range(7):
#     #         js = command.joint_speeds.add()
#     #         js.joint_identifier = i               # Gen3 joint index
#     #         js.value = float(dq[i])               # rad/s; use rad2deg if your setup wants deg/s
#     #         js.duration = 0                       # execute immediately

#     #     try:
#     #         self.base.SendJointSpeedsCommand(command)
#     #     except Exception as e:
#     #         self.get_logger().warn(f"Kortex SendJointSpeedsCommand error: {e}")

#     # --- ROS callbacks ---
#     def _cb_ext_raw(self, msg: RosImage):
#         try: self.latest_ext = resize_with_pad_uint8(to_rgb_from_raw_msg(msg), self.img_size)
#         except Exception as e: self.get_logger().warn(f"ext raw decode error: {e}")

#     def _cb_wrist_raw(self, msg: RosImage):
#         try: self.latest_wrist = resize_with_pad_uint8(to_rgb_from_raw_msg(msg), self.img_size)
#         except Exception as e: self.get_logger().warn(f"wrist raw decode error: {e}")

#     def _cb_ext_compressed(self, msg: CompressedImage):
#         try: self.latest_ext = resize_with_pad_uint8(to_rgb_from_compressed_msg(msg), self.img_size)
#         except Exception as e: self.get_logger().warn(f"ext compressed decode error: {e}")

#     def _cb_wrist_compressed(self, msg: CompressedImage):
#         try: self.latest_wrist = resize_with_pad_uint8(to_rgb_from_compressed_msg(msg), self.img_size)
#         except Exception as e: self.get_logger().warn(f"wrist compressed decode error: {e}")

#     def make_index_map(self, current_names, desired_order):
#         idx = {n: i for i, n in enumerate(current_names)}
#         return [idx.get(n, -1) for n in desired_order]

#     def reorder_joint_state_msg(self, msg, desired_order):
#         idx_map = self.make_index_map(msg.name, desired_order)

#         def pick(arr, i):
#             if i < 0 or i >= len(arr):
#                 return math.nan
#             return arr[i]

#         msg.name     = [desired_order[j] for j in range(len(idx_map))]
#         msg.position = [pick(msg.position, di) for di in idx_map]
#         msg.velocity = [pick(msg.velocity, di) for di in idx_map]
#         msg.effort   = [pick(msg.effort,   di) for di in idx_map]
#         return msg

#     def _cb_joint(self, msg: JointState):
#         try:
#             js = self.reorder_joint_state_msg(msg, self.desired_order)
#             names = list(js.name)
#             pos = np.asarray(js.position, dtype=np.float32)
#             pos = np.nan_to_num(pos, nan=0.0)
#             if self.gripper_name in names:
#                 gi = names.index(self.gripper_name)
#                 arm_mask = np.ones(len(names), dtype=bool); arm_mask[gi] = False
#                 q7 = pos[arm_mask][:7]
#                 g1 = float(np.nan_to_num(pos[gi], nan=0.0))
#             else:
#                 q7 = pos[:7]; g1 = 0.0
#             # self.latest_q = wrap_to_pi(q7.astype(np.float32))
#             self.latest_q = q7.astype(np.float32)
#             self.latest_g = g1
#             self.latest_joint_names = [n for n in names if n != self.gripper_name]
#         except Exception as e:
#             self.get_logger().warn(f"joint parse error: {e}")

#     # --- Control loop with swap-when-ready (min_exec_steps) ---
#     def _tick(self):
#         if self.latest_q is None or self.latest_g is None or self.latest_ext is None or self.latest_wrist is None:
#             if self.latest_ext is None:
#                 self.get_logger().info("Waiting for external image…")
#             if self.latest_wrist is None:
#                 self.get_logger().info("Waiting for wrist image…")
#             if self.latest_q is None:
#                 self.get_logger().info("Waiting for joint state…")
#             if self.latest_g is None:
#                 self.get_logger().info("Waiting for gripper state…")
#             return

#         # Ensure current chunk exists
#         if self._curr_chunk is None:
#             self._curr_chunk = self._blocking_infer_chunk()
#             self._curr_idx = 0
#             if self._curr_chunk is None: return

#             # If trajectory mode, send trajectory now for (at least) min_exec_steps
#             if self.cmd_mode == "trajectory":
#                 self._send_trajectory_for_window(self._curr_chunk, start_idx=0, n_steps=max(self.min_exec_steps, 1))

#         # Execute one step or just time the loop if trajectory is running
#         remaining = len(self._curr_chunk) - self._curr_idx

#         # Prefetch when small remaining
#         if remaining <= self.prefetch_remaining and self._prefetch_future is None:
#             self._prefetch_future = self._pool.submit(self._infer_chunk_safe)

#         # In log/velocity mode, consume per tick
#         if self.cmd_mode in ("log", "velocity"):
#             a = self._curr_chunk[self._curr_idx]
#             dq = np.clip(a[:7], -self.max_vel, self.max_vel)
#             dg = float(np.clip(a[7], -self.max_vel, self.max_vel))

#             # --- Safety scaling ---
#             dq *= getattr(self, "vel_scale", 0.1)
#             dg *= getattr(self, "vel_scale", 0.1)
#             # ----------------------

#             # --- Shortest angular path sanitization for velocity ---
#             if self.latest_q is not None:
#                 dq = self._shortest_path_velocity(self.latest_q, dq)
#             # -------------------------------------------------------

#             if self.cmd_mode == "log":
#                 self.get_logger().info(
#                     f"[{self._curr_idx+1}/{len(self._curr_chunk)}] dq(rad/s)={dq.round(3).tolist()} dg={round(dg,3)}"
#                 )
#                 self._curr_idx += 1

#             elif self.cmd_mode == "velocity":
#                 # OPTION A: send directly to Kortex (recommended for Gen3)
#                 # self._send_kortex_joint_velocities(dq)
#                 self._curr_idx += 1

#                 # OPTION B (legacy): if you still want to publish to ROS velocity controller,
#                 # uncomment this block and keep both.

#                 # if not hasattr(self, "publisher"):
#                 #     self.publisher = self.create_publisher(
#                 #         JointTrajectory, "/joint_group_velocity_controller/joint_trajectory", 10
#                 #     )
#                 #
#                 # if hasattr(self, "latest_joint_names") and self.latest_joint_names:
#                 #     joint_names = self.latest_joint_names
#                 #     dq_map = {f"joint_{i+1}": dq[i] for i in range(min(len(dq), 7))}
#                 #     dq_final = [dq_map.get(jn, 0.0) for jn in joint_names]
#                 # else:
#                 #     joint_names = [f"joint_{i+1}" for i in range(7)]
#                 #     dq_final = dq.tolist()
#                 #
#                 # msg = JointTrajectory()
#                 # msg.joint_names = joint_names
#                 # pt = JointTrajectoryPoint()
#                 # pt.velocities = [float(v) for v in dq_final]
#                 # pt.time_from_start = dur_from_seconds(self.dt)
#                 # msg.points = [pt]
#                 # self.publisher.publish(msg)

#         elif self.cmd_mode == "trajectory":
#             self._curr_idx += 1  

#         if self._prefetch_future is not None and self._prefetch_future.done() and self._curr_idx >= self.min_exec_steps:
#             next_chunk = self._prefetch_future.result()
#             self._prefetch_future = None
#             if next_chunk is not None and len(next_chunk) > 0:
#                 self._curr_chunk = next_chunk
#                 self._curr_idx = 0
#                 if self.cmd_mode == "trajectory":
#                     self._send_trajectory_for_window(self._curr_chunk, start_idx=0, n_steps=max(self.min_exec_steps, 1))
#                 return

#         if self._curr_idx >= len(self._curr_chunk):
#             next_chunk = self._blocking_infer_chunk()
#             if next_chunk is not None and len(next_chunk) > 0:
#                 self.latest_chunk = next_chunk
#                 self._curr_chunk = next_chunk
#                 self._curr_idx = 0
#                 if self.cmd_mode == "trajectory":
#                     self._send_trajectory_for_window(self._curr_chunk, start_idx=0, n_steps=max(self.min_exec_steps, 1))

#     # --- OpenPI infer helpers ---
#     def _build_observation(self) -> dict:
#         return {
#             "observation/exterior_image_1_left": self.latest_ext,            # uint8 HxWx3
#             "observation/wrist_image_left":      self.latest_wrist,          # uint8 HxWx3
#             "observation/joint_position":        self.latest_q.tolist(),     # 7 abs rad
#             "observation/gripper_position":      [float(self.latest_g)],     # 1 abs
#             "prompt": self.prompt,
#             "chunk_size": self.chunk_size,
#         }

#     def _blocking_infer_chunk(self) -> Optional[np.ndarray]:
#         try:
#             t0 = time.perf_counter()
#             result = self.client.infer(self._build_observation())
#             dt_ms = (time.perf_counter() - t0) * 1000.0
#             acts = np.asarray(result.get("actions", []), dtype=np.float32)
#             if acts.ndim != 2 or acts.shape[1] != 8:
#                 self.get_logger().warn(f"Bad action shape {acts.shape}; expected (N,8)")
#                 return None
#             self.get_logger().info(f"infer OK {acts.shape} in {dt_ms:.1f} ms")
#             return acts
#         except Exception as e:
#             self.get_logger().warn(f"infer error: {e}")
#             return None

#     def _infer_chunk_safe(self) -> Optional[np.ndarray]:
#         try:
#             result = self.client.infer(self._build_observation())
#             acts = np.asarray(result.get("actions", []), dtype=np.float32)
#             if acts.ndim == 2 and acts.shape[1] == 8:
#                 return acts
#         except Exception as e:
#             self.get_logger().warn(f"prefetch infer error: {e}")
#         return None

#     # --- Trajectory building & sending ---
#     def _sanitize_segment(self, q_curr: List[float], q_next: List[float]) -> List[float]:
#         out = []
#         for c, t in zip(q_curr, q_next):
#             delta = t - c
#             while delta > math.pi:  delta -= 2 * math.pi
#             while delta < -math.pi: delta += 2 * math.pi
#             out.append(c + delta)
#         return out
    
#     def _shortest_path_velocity(self, q_curr: List[float], dq_raw: np.ndarray) -> np.ndarray:
#         """
#         Take a raw velocity (rad/s), integrate one dt to get q_next_raw,
#         sanitize q_next via shortest angular path, then convert back to velocity.
#         """
#         q_curr = np.asarray(q_curr, dtype=np.float64)
#         dq_raw = np.asarray(dq_raw, dtype=np.float64)

#         # 1) naive next position from the velocity
#         q_next_raw = q_curr + dq_raw * self.dt

#         # 2) sanitize that position step
#         q_next_sanitized = np.array(
#             self._sanitize_segment(q_curr.tolist(), q_next_raw.tolist()),
#             dtype=np.float64,
#         )

#         # 3) convert back to velocity
#         dq_sane = (q_next_sanitized - q_curr) / self.dt
#         return dq_sane.astype(np.float32)

#     def _send_trajectory_for_window(self, chunk: np.ndarray, start_idx: int, n_steps: int):
#         if self.latest_q is None: return
#         end_idx = min(start_idx + n_steps, len(chunk))
#         window = chunk[start_idx:end_idx]  # shape (K, 8)
#         if window.size == 0: return

#         print(f"Sending trajectory for steps {start_idx} to {end_idx} (total {len(chunk)})")

#         q0 = self.latest_q.astype(np.float64).tolist()   # 7
#         g0 = float(self.latest_g)

#         q_points: List[List[float]] = []
#         g_points: List[float] = []

#         q_curr = q0[:]
#         g_curr = g0
#         tfs_list: List[float] = []
#         t_accum = 0.0

#         for i, a in enumerate(window):
#             dq = np.clip(a[:7], -self.max_vel, self.max_vel)
#             if self.gripper_action == "velocity":
#                 dg = float(np.clip(a[7], -self.max_vel, self.max_vel))
#                 g_next = g_curr + dg * self.dt
#             else:  # position
#                 g_next = float(a[7])

#             q_next = (np.array(q_curr) + dq * self.dt).astype(np.float64)

#             # q_next = wrap_to_pi(q_next)  # kept commented as in your code
#             q_next = np.array(self._sanitize_segment(q_curr, q_next.tolist()), dtype=np.float64)

#             q_points.append(q_next.tolist())
#             g_points.append(g_next)
#             q_curr = q_next.tolist()
#             g_curr = g_next

#             t_accum += max(self.traj_min_segment_time, self.dt*30)  # same as your code
#             tfs_list.append(t_accum)

#         default_names = [
#             "joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7"
#         ]
#         arm_joint_names = default_names

#         traj = JointTrajectory()
#         traj.joint_names = arm_joint_names
#         for q, tfs in zip(q_points, tfs_list):
#             pt = JointTrajectoryPoint()
#             # pt.positions = wrap_to_pi(np.array(q)).tolist()  # kept commented as in your code
#             pt.positions = q
#             print(pt.positions)
#             pt.time_from_start = dur_from_seconds(tfs)
#             traj.points.append(pt)

#         goal = FollowJointTrajectory.Goal()
#         goal.trajectory = traj
#         goal.goal_time_tolerance = dur_from_seconds(self.traj_goal_time_tol*30)

#         if not self._traj_client.wait_for_server(timeout_sec=1.0):
#             self.get_logger().warn("FollowJointTrajectory action server not available.")
#             return

#         send_future = self._traj_client.send_goal_async(goal)

#         def _on_goal(fut):
#             gh = fut.result()
#             if gh and gh.accepted:
#                 self.get_logger().info(f"Trajectory goal accepted ({len(traj.points)} pts, {tfs_list[-1]:.2f}s).")
#             else:
#                 self.get_logger().warn("Trajectory goal rejected.")
#         send_future.add_done_callback(_on_goal)

#         try:
#             if not self._grip_client.wait_for_server(timeout_sec=0.5):
#                 self.get_logger().warn("Gripper action server not available.")
#                 return
#             g_target = float(np.clip(g_points[-1], 0.0, 1.0))
#             g_goal = GripperCommand.Goal()
#             g_goal.command.position = g_target
#             g_goal.command.max_effort = 25.0
#             self._grip_client.send_goal_async(g_goal)
#         except Exception as e:
#             self.get_logger().warn(f"Gripper send error: {e}")