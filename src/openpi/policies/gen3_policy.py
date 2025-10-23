"""
Gen3 policy adapter for Pi0 / Pi0.5 models.

This maps raw RLDS observation dictionaries into the unified OpenPI
input format expected by the model, and converts model outputs into
robot-executable action arrays.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model



# def make_droid_example() -> dict:
#     """Creates a random input example for the Droid policy."""
#     return {
#         "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
#         "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
#         "observation/joint_position": np.random.rand(7),
#         "observation/gripper_position": np.random.rand(1),
#         "prompt": "do something",
#     }



def _parse_image(image: np.ndarray) -> np.ndarray:
    """Ensure image is in uint8 (H, W, C) format."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class Gen3Inputs(transforms.DataTransformFn):
    """Transforms RLDS observations into OpenPI model inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # --- State vector: joint positions + gripper ---
        gripper_pos = np.asarray(data.get("observation/gripper_position", []))
        if gripper_pos.ndim == 0:
            gripper_pos = gripper_pos[np.newaxis]
        joint_pos = np.asarray(data["observation/joint_position"])
        state = np.concatenate([joint_pos, gripper_pos])

        # --- Images ---
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        # --- Model-type-dependent naming ---
        if self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            images = (base_image, wrist_image, np.zeros_like(base_image))
            image_masks = (True, True, False)
        elif self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            images = (base_image, np.zeros_like(base_image), wrist_image)
            image_masks = (True, True, True)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # --- Optional fields ---
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class Gen3Outputs(transforms.DataTransformFn):
    """Extracts the first 8 dimensions (7 joints + 1 gripper) of the action array."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :8])}
