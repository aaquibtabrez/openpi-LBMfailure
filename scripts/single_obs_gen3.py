#!/usr/bin/env python3
# single_obs_openpi_infer.py
# Usage:
#   python3 single_obs_openpi_infer.py --host 127.0.0.1 --port 8000 \
#     --external /path/to/external_img_or_dir --wrist /path/to/wrist_img_or_dir \
#     --prompt "do something" --chunk-size 10
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import time


# Your provided state
DEFAULT_JOINT = [
    -0.1992577165365219,
    0.14662906527519226,
    2.69062876701355,
    -2.104942798614502,
    0.006928363349288702,
    -0.9787953495979309,
    0.8982262015342712,
]
DEFAULT_GRIPPER = 0.003568
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
def find_image(path_str: str) -> Path:
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.is_file():
        return p
    for q in sorted(p.iterdir()):
        if q.suffix.lower() in IMG_EXTS:
            return q
    raise FileNotFoundError(f"No image files found in: {p}")
def load_image_uint8_with_pad(path: Path, size: int = 224) -> np.ndarray:
    """Open -> RGB -> resize_with_pad -> uint8 array (H,W,3)."""
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im)
    arr = image_tools.resize_with_pad(arr, size, size)
    arr = image_tools.convert_to_uint8(arr)  # ensures dtype uint8, 0..255
    return arr
def main():
    ap = argparse.ArgumentParser("Single observation → OpenPI client.infer()")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--external", required=True, help="External image file or directory")
    ap.add_argument("--wrist", required=True, help="Wrist image file or directory")
    ap.add_argument("--prompt", default="do something")
    ap.add_argument("--chunk-size", type=int, default=10)
    # Optional overrides
    ap.add_argument("--joint", type=str, default=None, help='JSON list of 7 joint angles in radians')
    ap.add_argument("--gripper", type=float, default=None, help='Absolute gripper position')
    args = ap.parse_args()
    ext_path = find_image(args.external)
    wrist_path = find_image(args.wrist)
    ext_img = load_image_uint8_with_pad(ext_path, 224)
    wrist_img = load_image_uint8_with_pad(wrist_path, 224)
    joint = json.loads(args.joint) if args.joint else DEFAULT_JOINT
    if not (isinstance(joint, list) and len(joint) == 7):
        raise ValueError("--joint must be a JSON list of length 7")
    gripper = float(args.gripper) if args.gripper is not None else DEFAULT_GRIPPER
    # Build observation using the OpenPI/DROID-style keys we’ve been targeting
    observation = {
        "observation/exterior_image_1_left": ext_img,          # uint8 HxWx3
        "observation/wrist_image_left":      wrist_img,         # uint8 HxWx3
        "observation/joint_position":        joint,             # 7 abs rad
        "observation/gripper_position":      [gripper],         # 1 abs
        "prompt": args.prompt,
        "chunk_size": args.chunk_size,                          # hint to server
    }
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    start = time.time()
    result = client.infer(observation)
    end = time.time()
    print(f"\n Inference time: {(end - start)*1000:.2f} ms")
    actions = np.array(result.get("actions", []), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 8:
        raise RuntimeError(f"Unexpected action shape {actions.shape}; expected (N, 8)")
    np.set_printoptions(precision=4, suppress=True)
    print(f"\n=== Action chunk received: shape={actions.shape} ===")
    print(actions)
if __name__ == "__main__":
    main()