# Pi0 Training Data Pipeline (Kinova Gen3)

This readme contains the instructions for the data preparation pipeline used to train a Pi0 policy on a Kinova Gen3 (7-DoF) arm with a Robotiq gripper.

The pipeline converts recorded ROS 2 rosbag2 logs into a LeRobot (HuggingFace) dataset compatible with OpenPI’s `LeRobotDROIDDataConfig`.

## Overview

Each rosbag2 recording is expected to contain:

- Two camera topics:
  - Base/exterior camera
  - Wrist camera
- Joint state topic:
  - `/joint_states` containing 7 arm joints and 1 gripper joint

The pipeline is:

1. Reorder `/joint_states` in the rosbag2 so the joint arrays are consistently ordered as joints 1–7, then gripper (joints + gripper positions are out of order by default).
2. Convert reordered rosbag2 directories into a LeRobot dataset (one episode per bag), with support for multiple prompts.

A separate optional section at the end describes how to generate truncated “failure” episodes.

## Repository Scripts

- `reorder_all_bags.py`
  Batch-reorders the arrays inside `sensor_msgs/JointState` messages for a given topic (default `/joint_states`) across many rosbag2 directories.

- `rosbag2_to_lerobot_many_prompts_with_failure.py`
  Converts rosbag2 directories into a LeRobot dataset compatible with OpenPI’s DROID-style inputs (two camera streams + joint/gripper state + actions). This script supports:
  - Multiple prompts via a prompt file (one prompt per task subfolder under `input_root`)
  - Optional “failure” truncation (documented in a separate section)

Note: The converter decodes camera topics as `sensor_msgs/msg/CompressedImage` (JPEG). It is intended for `/.../compressed` camera topics. See the script usage example header.

## Step 1: Reorder JointState Arrays

ROS does not guarantee that `JointState.name`, `JointState.position`, `JointState.velocity`, and `JointState.effort` appear in a consistent order across recordings. Before conversion, reorder all bags so the joint arrays follow a fixed order:

- Arm joints in order 1–7
- Then the gripper joint

This ensures consistent tensor indexing and makes downstream dataset inspection simpler.

### Command
```
python3 reorder_all_bags.py \
  <input_root> \
  <output_root> \
  --topic /joint_states \
  --order joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,joint_7,robotiq_85_left_knuckle_joint \
  --overwrite
```
### Arguments

- `input_root`
  Folder containing rosbag2 directories (or a single rosbag2 directory).

- `output_root`
  Folder where reordered bags will be written.

- `--topic`
  JointState topic to reorder (default: `/joint_states`).

- `--order`
  Comma-separated desired joint name order. (default: `joint_1,joint_2,joint_3,joint_4,joint_5,joint_6,joint_7,robotiq_85_left_knuckle_joint`)

- `--overwrite`
  If set, deletes existing output bag directories before writing.

### Output

Each bag is written to:

`<output_root>/<original_bag_name>_reordered`

Only the specified JointState topic is modified. All other topics are copied through unchanged.

## Step 2: Convert Reordered Bags to LeRobot Dataset (Many Prompts)

This step uses `rosbag2_to_lerobot_many_prompts_with_failure.py` to create a single LeRobot dataset from many rosbag2 recordings.

Each rosbag2 directory becomes exactly one episode in the dataset.

### Expected Folder Layout

`--input-root` must contain subfolders. Each subfolder represents one “task group” and contains one or more rosbag2 directories.

Example:
```
<reordered_bag_root>/
  task_01/
    bag_a_reordered/
    bag_b_reordered/
  task_02/
    bag_c_reordered/
  task_03/
    bag_d_reordered/
    bag_e_reordered/
```
The prompt file must contain exactly one line per task subfolder. The script pairs prompts to subfolders in sorted order and asserts the counts match. :contentReference[oaicite:1]{index=1}

### Prompt File Format

Create a text file (for example `prompts.txt`) with one prompt per line, in the same order as the sorted task subfolders.

Example:
```
pick and place the object
sort items into the correct bin
move the object to the staging area
```
The prompt is written into the dataset `task` field for every frame in the episode. 

### Dataset Structure

For each timestep, the dataset stores:

Observations:
- `exterior_image_1_left` (base/exterior RGB image)
- `wrist_image_left` (wrist RGB image)
- `joint_position` (7,) absolute arm joint positions
- `gripper_position` (1,) absolute gripper position

Actions:
- `actions` (8,)
  - First 7 values: joint velocities computed as forward finite differences divided by dt
  - Last value: absolute gripper position

A language instruction string is stored in:
- `task`

### Recommended Usage (Typical)

This is the “typical” command most users should run. It includes required arguments and keeps defaults implicit (image size, fps defaults, etc. can be adjusted if needed).
```
python3 rosbag2_to_lerobot_many_prompts_with_failure.py \
  --input-root <reordered_bag_root> \
  --prompt-file prompts.txt \
  --repo-id <hf_namespace>/<dataset_name> \
  --camera-topic-base  /camera1/color/image_raw/compressed \
  --camera-topic-wrist /camera2/color/image_raw/compressed \
  --gripper-name robotiq_85_left_knuckle_joint \
  --fps 20
```
### Full Usage (All Options)
```
python3 rosbag2_to_lerobot_many_prompts_with_failure.py \
  --input-root <reordered_bag_root> \
  --prompt-file prompts.txt \
  --repo-id <hf_namespace>/<dataset_name> \
  --camera-topic-base  /camera1/color/image_raw/compressed \
  --camera-topic-wrist /camera2/color/image_raw/compressed \
  --joint-topic /joint_states \
  --gripper-name robotiq_85_left_knuckle_joint \
  --fps 20 \
  --img-width 224 \
  --img-height 224 \
  --compressed
```
Notes:
- The script uses `sensor_msgs/msg/CompressedImage` decoding for camera topics. 
- The script accepts `--compressed`, but decoding is already implemented as compressed-image decoding in the current version.

### Important Arguments

- `--input-root`
  Folder containing task subfolders, each containing one or more rosbag2 directories.

- `--prompt-file`
  Text file containing one prompt per task subfolder (one line per subfolder).

- `--repo-id`
  Target LeRobot dataset identifier, typically `<hf_namespace>/<dataset_name>`.

- `--camera-topic-base`
  Base/exterior camera compressed image topic.

- `--camera-topic-wrist`
  Wrist camera compressed image topic.

- `--joint-topic`
  JointState topic (default `/joint_states`).

- `--gripper-name`
  Name of the gripper joint inside `JointState.name`.

- `--fps`
  Uniform resampling frequency used to build the episode timeline.

- `--img-width`, `--img-height`
  Images are resized to this resolution before storage.

### Output Location

The dataset is written to:

`$HF_LEROBOT_HOME/<repo_id>`

`HF_LEROBOT_HOME` is the LeRobot cache root used by OpenPI. //NOTE: CLARIFY HOW TO FIND THIS PATH


### Notes and Assumptions

- Always run Step 1 (reordering) before conversion.
- Each rosbag2 directory becomes exactly one episode.
- Joint ordering must remain consistent across all data.
- The prompt file is matched to task folders by sorted order; keep folder naming stable and explicit (e.g., `task_01_*`, `task_02_*`).
- The action vector is always 8D:
  - 7D joint velocity (dq/dt)
  - 1D absolute gripper position

## Optional: Failure Episode Generation

This section is optional. Most users can ignore it.

The same converter script can generate truncated “failure” episodes on a per-task basis. Failure behavior is enabled using two mechanisms:

1. Prompt tagging:
   - If a prompt line starts with `failure-`, that task group is treated as “failure-enabled”.
   - The `failure-` prefix is stripped before writing the dataset `task` text. 

2. Failure cut flags:
   - `--cut-gripper-threshold` sets a cutoff value for `gripper_position`.
   - `--cut-extra-frames` appends additional frozen frames after the cutoff.

Failure truncation only occurs when:
- the prompt line is tagged with `failure-`, and
- `--cut-gripper-threshold` is provided.
### Prompt File Example (Mixed Success + Failure)

pick and place the object
failure-pick but stop before grasping
sort items into the correct bin

In this example, only the second task subfolder will use failure truncation.

### Failure Command (Typical)
```
python3 rosbag2_to_lerobot_many_prompts_with_failure.py \
  --input-root <reordered_bag_root> \
  --prompt-file prompts.txt \
  --repo-id <hf_namespace>/<dataset_name> \
  --camera-topic-base  /camera1/color/image_raw/compressed \
  --camera-topic-wrist /camera2/color/image_raw/compressed \
  --gripper-name robotiq_85_left_knuckle_joint \
  --fps 20 \
  --cut-gripper-threshold 0.3 \
  --cut-extra-frames 30
```
### Failure Parameters

- `--cut-gripper-threshold`
  If set, failure-tagged task groups will truncate an episode at the first frame where:
  `gripper_position >= threshold`

- `--cut-extra-frames`
  Number of frozen frames to append after truncation. These frames use:
  - the last images
  - the last gripper position
  - near-identical joint positions (small noise added)

This is useful for training behaviors that intentionally stop or “fail” at a specific gripper closure point.

Implementation reference: `rosbag2_to_lerobot_many_prompts_with_failure.py`. 