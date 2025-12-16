#!/usr/bin/env python3

import argparse
import rclpy

from pi0_client import PiZeroOpenPIClient

# --- Kortex imports ---
from kortex_api.RouterClient import RouterClient, RouterClientCallbacks
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.transport import TCPTransport
from kortex_api.autogen.messages import Session_pb2
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient


def create_kortex_clients(ip: str, username: str, password: str):
    """
    Connect to the Kinova Gen3 using Kortex API and return:
    base, base_cyclic, session_manager, router, transport
    """
    # Transport + router
    tcp_transport = TCPTransport()
    tcp_transport.connect(ip, port=10000)  # default Kortex TCP port

    router = RouterClient(tcp_transport, RouterClientCallbacks())

    # Session (login)
    session_manager = SessionManager(router)
    session_info = Session_pb2.CreateSessionInfo()
    session_info.username = username
    session_info.password = password
    session_info.session_inactivity_timeout = 60000   # ms
    session_info.connection_inactivity_timeout = 2000 # ms

    session_manager.CreateSession(session_info)

    # API clients
    base = BaseClient(router)
    base_cyclic = BaseCyclicClient(router)

    return base, base_cyclic, session_manager, router, tcp_transport


def close_kortex(session_manager, router, transport):
    """Cleanly close Kortex session and transport."""
    try:
        session_manager.CloseSession()
    except Exception:
        pass
    try:
        router.disconnect()
    except Exception:
        pass
    try:
        transport.disconnect()
    except Exception:
        pass


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
    ap.add_argument("--prefetch-remaining", type=int, default=5,
                    help="Start prefetch when <= this many remain")
    ap.add_argument("--min-exec-steps", type=int, default=4,
                    help="Must execute at least this many steps before swapping")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--max-vel-rad", type=float, default=1.0,
                    help="Clamp on returned joint velocities (rad/s)")

    ap.add_argument("--cmd-mode", choices=["log", "velocity", "trajectory"], default="log",
                    help="How to consume actions each tick")
    ap.add_argument("--gripper-action", choices=["velocity", "position"], default="velocity",
                    help="What the policy's 8th channel means")

    ap.add_argument("--traj-action-ns", default="/joint_trajectory_controller/follow_joint_trajectory")
    ap.add_argument("--grip-action-ns", default="/robotiq_gripper_controller/gripper_cmd")
    ap.add_argument("--traj-goal-time-tol", type=float, default=0.25)
    ap.add_argument("--traj-min-segment-time", type=float, default=0.10,
                    help="Minimum time per trajectory segment (s)")

    ap.add_argument("--vel-scale", type=float, default=0.1,
                    help="Scale factor for joint/gripper velocities (safety)")
    ap.add_argument("--steps-per-infer", type=int, default=4,
                    help="How many actions to execute per inference cycle")
    ap.add_argument("--speed-scale", type=float, default=1.0,
                    help="Overall speed multiplier. <1.0 = slower, >1.0 = faster")

    # Kortex connection args
    ap.add_argument("--robot-ip", default="192.168.1.10", help="Kinova Gen3 IP address")
    ap.add_argument("--robot-username", default="admin")
    ap.add_argument("--robot-password", default="admin")

    args = ap.parse_args()

    rclpy.init()

    # Connect to the robot via Kortex
    base, base_cyclic, session_manager, router, transport = create_kortex_clients(
        args.robot_ip, args.robot_username, args.robot_password
    )

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
        base=base,
        base_cyclic=base_cyclic,
    )

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        close_kortex(session_manager, router, transport)


if __name__ == "__main__":
    main()
