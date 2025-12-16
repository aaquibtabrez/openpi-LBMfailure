import numpy as np
import cv2

from builtin_interfaces.msg import Duration as RosDuration
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import CompressedImage

# OpenPI resize utilities (same as your original import location)
from openpi_client import image_tools


# ---------- QoS profile (same semantics as your file) ----------
# If you prefer BEST_EFFORT, you can switch back via code edit here without touching node.py
# SENSOR_QOS = QoSProfile(
#     reliability=QoSReliabilityPolicy.RELIABLE,
#     history=QoSHistoryPolicy.KEEP_LAST,
#     depth=5,
# )


# # ---------- helpers moved verbatim (names unchanged) ----------

# def wrap_to_pi(x: np.ndarray) -> np.ndarray:
#     return (x + np.pi) % (2 * np.pi) - np.pi


# def to_rgb_from_raw_msg(msg: RosImage) -> np.ndarray:
#     h, w = msg.height, msg.width
#     enc = (msg.encoding or "").lower()
#     buf = np.frombuffer(msg.data, dtype=np.uint8)
#     if enc == "rgb8":
#         return buf.reshape(h, w, 3)
#     elif enc == "bgr8":
#         img = buf.reshape(h, w, 3)
#         return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     elif enc == "rgba8":
#         img = buf.reshape(h, w, 4)[:, :, :3]
#         return img
#     elif enc == "bgra8":
#         img = buf.reshape(h, w, 4)
#         return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
#     elif enc == "mono8":
#         img = buf.reshape(h, w)
#         return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     else:
#         raise RuntimeError(f"Unsupported Image encoding '{msg.encoding}' ({h}x{w}).")


# def to_rgb_from_compressed_msg(msg: CompressedImage) -> np.ndarray:
#     np_arr = np.frombuffer(msg.data, np.uint8)
#     bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if bgr is None:
#         raise RuntimeError(f"Failed to decode CompressedImage (format={getattr(msg, 'format', 'unknown')})")
#     return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# def resize_with_pad_uint8(rgb: np.ndarray, size: int = 224) -> np.ndarray:
#     arr = image_tools.resize_with_pad(rgb, size, size)
#     return image_tools.convert_to_uint8(arr)


# def dur_from_seconds(s: float) -> RosDuration:
#     s = max(0.0, float(s))
#     sec = int(s)SENSOR_QOS = QoSProfile(
#     reliability=QoSReliabilityPolicy.RELIABLE,
#     history=QoSHistoryPolicy.KEEP_LAST,
#     depth=5,
# )


# # ---------- helpers moved verbatim (names unchanged) ----------

# def wrap_to_pi(x: np.ndarray) -> np.ndarray:
#     return (x + np.pi) % (2 * np.pi) - np.pi


# def to_rgb_from_raw_msg(msg: RosImage) -> np.ndarray:
#     h, w = msg.height, msg.width
#     enc = (msg.encoding or "").lower()
#     buf = np.frombuffer(msg.data, dtype=np.uint8)
#     if enc == "rgb8":
#         return buf.reshape(h, w, 3)
#     elif enc == "bgr8":
#         img = buf.reshape(h, w, 3)
#         return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     elif enc == "rgba8":
#         img = buf.reshape(h, w, 4)[:, :, :3]
#         return img
#     elif enc == "bgra8":
#         img = buf.reshape(h, w, 4)
#         return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
#     elif enc == "mono8":
#         img = buf.reshape(h, w)
#         return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#     else:
#         raise RuntimeError(f"Unsupported Image encoding '{msg.encoding}' ({h}x{w}).")


# def to_rgb_from_compressed_msg(msg: CompressedImage) -> np.ndarray:
#     np_arr = np.frombuffer(msg.data, np.uint8)
#     bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#     if bgr is None:
#         raise RuntimeError(f"Failed to decode CompressedImage (format={getattr(msg, 'format', 'unknown')})")
#     return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# def resize_with_pad_uint8(rgb: np.ndarray, size: int = 224) -> np.ndarray:
#     arr = image_tools.resize_with_pad(rgb, size, size)
#     return image_tools.convert_to_uint8(arr)


# def dur_from_seconds(s: float) -> RosDuration:
#     s = max(0.0, float(s))
#     sec = int(s)
#     nsec = int((s - sec) * 1e9)
#     return RosDuration(sec=sec, nanosec=nsec)

#     nsec = int((s - sec) * 1e9)
#     return RosDuration(sec=sec, nanosec=nsec)

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

