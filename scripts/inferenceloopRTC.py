#!/usr/bin/env python3
"""
RTC-style Real-Time Chunking runtime for PiZero/OpenPI with Kinova Gen3 using FollowJointTrajectory,
with THROTTLED goal updates (avoid jitter from sending goals every tick).

Core RTC scheduling:
  - Execute continuously while inference runs (async prefetch).
  - When next chunk arrives, freeze overlap K steps (guaranteed-to-execute prefix) from current committed plan,
    and splice in the new chunk's suffix for the future.

Trajectory execution:
  - Maintain an action queue (committed plan).
  - Consume 1 action per control tick (control_hz).
  - Send a short multi-point FollowJointTrajectory at a lower rate (goal_hz), so the controller can track smoothly.
  - Preempt (cancel) only when sending a new goal (NOT every tick).

Note:
  - This implements the RTC runtime scheduling + overlap freeze/splice. True RTC "inpainting" would require
    the policy server to regenerate the remaining actions conditioned on the frozen prefix.
"""

import argparse
import concurrent.futures
import math
import time
from collections import deque
from typing import Optional, List, Deque

import numpy as np

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

from openpi_client import image_tools
from openpi_client import websocket_client_policy


SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
)


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


class PiZeroRTCThrottled(Node):
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
        horizon_steps: int,
        img_size: int,
        max_vel_rad: float,
        gripper_action: str,  # "velocity" | "position"
        traj_action_ns: str,
        grip_action_ns: str,
        traj_goal_time_tol: float,
        traj_min_segment_time: float,
        speed_scale: float,
        # RTC knobs
        prefetch_lead_ms: float,
        infer_ema_alpha: float,
        min_freeze_steps: int,
        max_freeze_steps: int,
        # NEW: goal throttling knobs
        goal_hz: float,
        goal_refresh_margin_ms: float,
    ):
        super().__init__("pizero_rtc_throttled_traj_client")

        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)

        # Basic config
        self.prompt = prompt
        self.use_compressed = compressed
        self.img_topic_ext = img_topic_ext
        self.img_topic_wrist = img_topic_wrist
        self.joint_topic = joint_topic
        self.gripper_name = gripper_name
        self.gripper_action = gripper_action

        self.chunk_size = int(chunk_size)
        self.horizon_steps = max(1, int(horizon_steps))

        self.img_size = int(img_size)
        self.max_vel = float(max_vel_rad)

        self.dt = 1.0 / float(control_hz)
        self.speed_scale = max(1e-6, float(speed_scale))

        self.traj_goal_time_tol = float(traj_goal_time_tol)
        self.traj_min_segment_time = float(traj_min_segment_time)

        # RTC scheduling
        self.prefetch_lead_s = float(prefetch_lead_ms) / 1000.0
        self.infer_ema_alpha = float(infer_ema_alpha)
        self.min_freeze_steps = int(min_freeze_steps)
        self.max_freeze_steps = int(max_freeze_steps)

        self._infer_ema_s: Optional[float] = None

        # Goal throttling
        self.goal_hz = max(1e-6, float(goal_hz))
        self.goal_period_s = 1.0 / self.goal_hz
        self.goal_refresh_margin_s = max(0.0, float(goal_refresh_margin_ms) / 1000.0)

        # Next time we are allowed to send a new goal (rate limit)
        self._next_goal_send_time: float = 0.0
        # Latest time by which we SHOULD refresh (so controller doesn’t run out of plan)
        self._goal_refresh_deadline: Optional[float] = None

        # Sensors
        self.latest_ext: Optional[np.ndarray] = None
        self.latest_wrist: Optional[np.ndarray] = None
        self.latest_q: Optional[np.ndarray] = None
        self.latest_g: Optional[float] = None
        self.latest_joint_names: List[str] = []

        self.desired_order = [
            "joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7", self.gripper_name
        ]

        # Subscriptions
        if self.use_compressed:
            self.create_subscription(CompressedImage, self.img_topic_ext, self._cb_ext_compressed, SENSOR_QOS)
            self.create_subscription(CompressedImage, self.img_topic_wrist, self._cb_wrist_compressed, SENSOR_QOS)
        else:
            self.create_subscription(RosImage, self.img_topic_ext, self._cb_ext_raw, SENSOR_QOS)
            self.create_subscription(RosImage, self.img_topic_wrist, self._cb_wrist_raw, SENSOR_QOS)
        self.create_subscription(JointState, self.joint_topic, self._cb_joint, SENSOR_QOS)

        # Action clients
        self._traj_client = ActionClient(self, FollowJointTrajectory, traj_action_ns)
        self._grip_client = ActionClient(self, GripperCommand, grip_action_ns)
        self._active_traj_goal_handle = None

        # Committed plan (queue of 8D actions)
        self._queue: Deque[np.ndarray] = deque()

        # Prefetch
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._prefetch_future: Optional[concurrent.futures.Future] = None

        # Timer
        self.timer = self.create_timer(self.dt, self._tick)

        self.get_logger().info(
            f"RTC throttled | control_hz={1.0/self.dt:.1f} | goal_hz={self.goal_hz:.2f} | "
            f"horizon_steps={self.horizon_steps} | chunk_size={self.chunk_size} | "
            f"prefetch_lead_ms={prefetch_lead_ms} | refresh_margin_ms={goal_refresh_margin_ms} | "
            f"gripper_action={self.gripper_action}"
        )

    # ---------- ROS callbacks ----------

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

    def _make_index_map(self, current_names, desired_order):
        idx = {n: i for i, n in enumerate(current_names)}
        return [idx.get(n, -1) for n in desired_order]

    def _reorder_joint_state_msg(self, msg, desired_order):
        idx_map = self._make_index_map(msg.name, desired_order)

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
        try:
            js = self._reorder_joint_state_msg(msg, self.desired_order)
            names = list(js.name)
            pos = np.asarray(js.position, dtype=np.float32)
            pos = np.nan_to_num(pos, nan=0.0)

            if self.gripper_name in names:
                gi = names.index(self.gripper_name)
                arm_mask = np.ones(len(names), dtype=bool)
                arm_mask[gi] = False
                q7 = pos[arm_mask][:7]
                g1 = float(np.nan_to_num(pos[gi], nan=0.0))
            else:
                q7 = pos[:7]
                g1 = 0.0

            self.latest_q = q7.astype(np.float32)
            self.latest_g = g1
            self.latest_joint_names = [n for n in names if n != self.gripper_name]
        except Exception as e:
            self.get_logger().warn(f"joint parse error: {e}")

    # ---------- OpenPI inference ----------

    def _build_observation(self) -> dict:
        return {
            "observation/exterior_image_1_left": self.latest_ext,
            "observation/wrist_image_left": self.latest_wrist,
            "observation/joint_position": self.latest_q.tolist(),
            "observation/gripper_position": [float(self.latest_g)],
            "prompt": self.prompt,
            "chunk_size": self.chunk_size,
        }

    def _infer_chunk_blocking(self) -> Optional[np.ndarray]:
        try:
            t0 = time.perf_counter()
            result = self.client.infer(self._build_observation())
            dt_s = time.perf_counter() - t0

            acts = np.asarray(result.get("actions", []), dtype=np.float32)
            if acts.ndim != 2 or acts.shape[1] != 8 or len(acts) == 0:
                self.get_logger().warn(f"Bad action shape {acts.shape}; expected (N,8)")
                return None

            # Update EMA latency estimate
            if self._infer_ema_s is None:
                self._infer_ema_s = dt_s
            else:
                a = self.infer_ema_alpha
                self._infer_ema_s = (1.0 - a) * self._infer_ema_s + a * dt_s

            self.get_logger().info(
                f"infer OK {acts.shape} in {dt_s*1000.0:.1f} ms (ema={self._infer_ema_s*1000.0:.1f} ms)"
            )
            return acts
        except Exception as e:
            self.get_logger().warn(f"infer error: {e}")
            return None

    def _estimated_freeze_steps(self) -> int:
        if self._infer_ema_s is None:
            est = self.prefetch_lead_s
        else:
            est = max(self._infer_ema_s, self.prefetch_lead_s)

        k = int(math.ceil(est / self.dt))
        k = max(self.min_freeze_steps, k)
        k = min(self.max_freeze_steps, k)
        return k

    # ---------- RTC merge (freeze overlap + splice future) ----------

    def _rtc_merge_new_chunk(self, new_chunk: np.ndarray) -> None:
        if new_chunk is None or new_chunk.ndim != 2 or new_chunk.shape[1] != 8:
            return

        K = self._estimated_freeze_steps()
        committed = list(self._queue)

        keep = committed[:K]
        suffix_start = min(K, len(new_chunk))
        suffix = [new_chunk[i].copy() for i in range(suffix_start, len(new_chunk))]

        self._queue = deque(keep + suffix)

        self.get_logger().info(
            f"RTC merge: freeze K={K}, queue_len={len(self._queue)} (new_chunk_len={len(new_chunk)})"
        )

    # ---------- Trajectory sending (throttled) ----------

    def _segment_time(self) -> float:
        base_seg = max(self.traj_min_segment_time, self.dt)
        return base_seg / self.speed_scale

    def _plan_horizon_time(self, n_steps: int) -> float:
        return float(n_steps) * self._segment_time()

    def _should_send_goal(self, now: float) -> bool:
        # Must have at least something to send
        if len(self._queue) == 0:
            return False

        # If we've never sent a goal, send immediately
        if self._goal_refresh_deadline is None:
            return True

        # Rate limit: allow if period elapsed
        if now >= self._next_goal_send_time:
            return True

        # Safety: refresh if we're nearing the end of our last sent horizon
        if now >= self._goal_refresh_deadline:
            return True

        return False

    def _send_trajectory_goal(self, now: float) -> None:
        if self.latest_q is None or len(self._queue) == 0:
            return

        # Use up to horizon_steps of the committed queue
        H = min(self.horizon_steps, len(self._queue))
        window = list(self._queue)[:H]

        q0 = self.latest_q.astype(np.float64).tolist()
        g0 = float(self.latest_g)

        q_curr = q0[:]
        g_curr = g0

        q_points: List[List[float]] = []
        g_points: List[float] = []
        tfs_list: List[float] = []
        t_accum = 0.0

        seg_time = self._segment_time()

        for a in window:
            dq = np.clip(a[:7], -self.max_vel, self.max_vel).astype(np.float64)
            q_next = (np.array(q_curr, dtype=np.float64) + dq * self.dt).astype(np.float64)
            q_points.append(q_next.tolist())
            q_curr = q_next.tolist()

            if self.gripper_action == "velocity":
                dg = float(np.clip(a[7], -self.max_vel, self.max_vel))
                g_next = g_curr + dg * self.dt
            else:
                g_next = float(a[7])

            g_points.append(g_next)
            g_curr = g_next

            t_accum += seg_time
            tfs_list.append(t_accum)

        traj = JointTrajectory()
        traj.joint_names = ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7"]
        for q, tfs in zip(q_points, tfs_list):
            pt = JointTrajectoryPoint()
            pt.positions = q
            pt.time_from_start = dur_from_seconds(tfs)
            traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        goal.goal_time_tolerance = dur_from_seconds(self.traj_goal_time_tol)

        if not self._traj_client.wait_for_server(timeout_sec=0.05):
            self.get_logger().warn("FollowJointTrajectory server not available.")
            return

        # Preempt previous goal ONLY when sending a new one
        try:
            if self._active_traj_goal_handle is not None:
                self._active_traj_goal_handle.cancel_goal_async()
        except Exception:
            pass

        send_future = self._traj_client.send_goal_async(goal)

        def _on_goal(fut):
            try:
                gh = fut.result()
            except Exception as e:
                self.get_logger().warn(f"Trajectory send exception: {e}")
                return
            if not gh or not gh.accepted:
                self.get_logger().warn("Trajectory goal rejected.")
                return
            self._active_traj_goal_handle = gh

        send_future.add_done_callback(_on_goal)

        # Gripper: send last target (same behavior as your scripts)
        try:
            if self._grip_client.wait_for_server(timeout_sec=0.01):
                g_target = float(np.clip(g_points[-1], 0.0, 1.0))
                g_goal = GripperCommand.Goal()
                g_goal.command.position = g_target
                g_goal.command.max_effort = 25.0
                self._grip_client.send_goal_async(g_goal)
        except Exception:
            pass

        # Update send scheduling:
        self._next_goal_send_time = now + self.goal_period_s

        # Refresh deadline: try to update before we run out of the plan we just sent
        total_time = self._plan_horizon_time(H)
        margin = min(self.goal_refresh_margin_s, max(0.0, total_time - 0.01))
        self._goal_refresh_deadline = now + (total_time - margin)

    # ---------- Main tick ----------

    def _tick(self):
        # Need sensors
        if self.latest_q is None or self.latest_g is None or self.latest_ext is None or self.latest_wrist is None:
            return

        now = time.monotonic()

        # Seed if empty
        if len(self._queue) == 0 and self._prefetch_future is None:
            chunk = self._infer_chunk_blocking()
            if chunk is None:
                return
            self._queue = deque([chunk[i].copy() for i in range(len(chunk))])

        # Prefetch when getting close to empty
        lead_steps = max(1, int(math.ceil(self.prefetch_lead_s / self.dt)))
        if self._prefetch_future is None and len(self._queue) <= lead_steps:
            self._prefetch_future = self._pool.submit(self._infer_chunk_blocking)

        # If prefetch ready, RTC-merge it
        if self._prefetch_future is not None and self._prefetch_future.done():
            try:
                new_chunk = self._prefetch_future.result()
            except Exception as e:
                self.get_logger().warn(f"prefetch future exception: {e}")
                new_chunk = None
            self._prefetch_future = None
            if new_chunk is not None:
                self._rtc_merge_new_chunk(new_chunk)

        # Throttled goal sending (not every tick)
        if self._should_send_goal(now):
            self._send_trajectory_goal(now)

        # Advance real-time execution: consume 1 action per tick
        if len(self._queue) > 0:
            self._queue.popleft()

        # If we run dry (inference too slow), block to recover
        if len(self._queue) == 0 and self._prefetch_future is None:
            chunk = self._infer_chunk_blocking()
            if chunk is not None:
                self._queue = deque([chunk[i].copy() for i in range(len(chunk))])


def main():
    ap = argparse.ArgumentParser("PiZero RTC runtime (throttled FollowJointTrajectory updates)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)

    ap.add_argument("--joint-topic", default="/joint_states")
    ap.add_argument("--gripper-name", default="robotiq_85_left_knuckle_joint")
    ap.add_argument("--compressed", action="store_true")

    ap.add_argument("--control-hz", type=float, default=20.0)
    ap.add_argument("--chunk-size", type=int, default=10)
    ap.add_argument("--horizon-steps", type=int, default=10, help="How many future actions to encode in each trajectory goal")

    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-vel-rad", type=float, default=1.0)
    ap.add_argument("--prompt", default="do something")

    ap.add_argument("--image-topic-external", default="/camera1/camera1/color/image_raw/compressed")
    ap.add_argument("--image-topic-wrist", default="/camera2/camera2/color/image_raw/compressed")

    ap.add_argument("--gripper-action", choices=["velocity","position"], default="position")

    ap.add_argument("--traj-action-ns", default="/joint_trajectory_controller/follow_joint_trajectory")
    ap.add_argument("--grip-action-ns", default="/robotiq_gripper_controller/gripper_cmd")
    ap.add_argument("--traj-goal-time-tol", type=float, default=0.25)
    ap.add_argument("--traj-min-segment-time", type=float, default=0.10)
    ap.add_argument("--speed-scale", type=float, default=1.0)

    # RTC knobs
    ap.add_argument("--prefetch-lead-ms", type=float, default=100.0)
    ap.add_argument("--infer-ema-alpha", type=float, default=0.25)
    ap.add_argument("--min-freeze-steps", type=int, default=1)
    ap.add_argument("--max-freeze-steps", type=int, default=6)

    # NEW: throttled goal updates
    ap.add_argument("--goal-hz", type=float, default=3.0, help="How often to send (and preempt) trajectory goals")
    ap.add_argument("--goal-refresh-margin-ms", type=float, default=150.0,
                    help="Refresh this long before the previously sent horizon would end")

    args = ap.parse_args()

    rclpy.init()
    node = PiZeroRTCThrottled(
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
        horizon_steps=args.horizon_steps,
        img_size=args.img_size,
        max_vel_rad=args.max_vel_rad,
        gripper_action=args.gripper_action,
        traj_action_ns=args.traj_action_ns,
        grip_action_ns=args.grip_action_ns,
        traj_goal_time_tol=args.traj_goal_time_tol,
        traj_min_segment_time=args.traj_min_segment_time,
        speed_scale=args.speed_scale,
        prefetch_lead_ms=args.prefetch_lead_ms,
        infer_ema_alpha=args.infer_ema_alpha,
        min_freeze_steps=args.min_freeze_steps,
        max_freeze_steps=args.max_freeze_steps,
        goal_hz=args.goal_hz,
        goal_refresh_margin_ms=args.goal_refresh_margin_ms,
    )

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
