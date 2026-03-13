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

## Python Environment Setup (LeRobot Dataset Conversion)

The dataset conversion scripts require a specific Python environment with compatible versions of LeRobot and its dependencies.

It is strongly recommended to create a dedicated conda environment for running the dataset conversion pipeline.

### 1. Create the Environment

You can name the environment anything you like. In our lab we typically call it `lerobot`.

Example:
```
  conda create -n lerobot python=3.10 -y
  conda activate lerobot
```
Python 3.10 is known to work. Python 3.11 may also work, but it has not been fully validated.


### 2. Install Required Python Packages

Install the following versions to match the dataset pipeline scripts:
```
  pip install datasets==2.19.*
  pip install huggingface_hub==0.34.6
  pip install pyarrow==14.0.2
  pip install numpy==1.26.4
```
These versions are known to work with the dataset conversion scripts.


### 3. Install the Correct Version of LeRobot

The required version of LeRobot is not available through PyPI in the correct form, so it must be installed directly from the HuggingFace repository at a specific commit.

Use the following commit:
```
  https://github.com/huggingface/lerobot/tree/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
```
Clone the repository and install it:
```
  git clone https://github.com/huggingface/lerobot.git
  cd lerobot
  git checkout 0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
  pip install -e .
```
After installation, confirm the version:
```
  python -c "import lerobot; print('LeRobot installed successfully')"
```

### 4. Verifying the Environment

Before running any dataset conversion scripts, verify that the required packages import correctly:
```
  python -c "import datasets, huggingface_hub, pyarrow, numpy, lerobot; print('Environment ready')"
```
If this command runs without errors, the environment is correctly configured.


### 5. Activating the Environment

Each time you start a new terminal session to run dataset conversion, activate the environment:
```
  conda activate lerobot
```
Then run the dataset pipeline commands described later in this README.


### Notes

- Always activate the `lerobot` environment before running:
  - `reorder_all_bags.py`
  - `rosbag2_to_lerobot_many_prompts_with_failure.py`

- If you encounter errors related to `pyarrow`, `datasets`, or `huggingface_hub`, it is almost always due to version mismatches.

- The pinned versions above are known to work with the dataset pipeline used in this repository.

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

`HF_LEROBOT_HOME` is the LeRobot cache root used by OpenPI. 

### Finding the `$HF_LEROBOT_HOME` Directory

LeRobot stores datasets under the environment variable `HF_LEROBOT_HOME`.

If this variable is not set, it defaults to a cache directory inside your home folder.

To check the path being used, run:
```
  python -c "from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME; print(HF_LEROBOT_HOME)"
```
The printed path is where the converted dataset will be written.

You can also manually set the location before running the conversion:
```
  export HF_LEROBOT_HOME=~/lerobot_datasets
```


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

## Uploading the Dataset to Hugging Face

Once the dataset has been generated using the dataset conversion pipeline, it can be uploaded to Hugging Face so that it can be accessed by the Pi0 training pipeline and shared with collaborators.

The upload process has two parts:

1. Uploading the dataset
2. Creating a dataset version tag

### 1. Log in to Hugging Face

First log in using the Hugging Face CLI:
```
  huggingface-cli login
```
You will be prompted to paste your Hugging Face access token.

You can generate a token here:
```
  https://huggingface.co/settings/tokens
```
Make sure the token has permission to create datasets.


### 2. Upload the Dataset

The dataset conversion script already saves the dataset in LeRobot format under the directory:
```
  $HF_LEROBOT_HOME/<repo_id>
```
For example:
```
  ~/lerobot_datasets/hrc2/kinova_pick_dataset
```
The `<repo_id>` passed to the conversion script should already follow the Hugging Face format:
```
  <username>/<dataset_name>
```
Example:
```
  hrc2/kinova_pick_dataset
```
If the dataset was created with the correct repo_id, it can be pushed to Hugging Face using the LeRobot dataset utilities.

Example Python snippet:
```
  from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

  dataset = LeRobotDataset(repo_id="username/dataset_name")
  dataset.push_to_hub()
```
This uploads the dataset to the Hugging Face dataset repository.

Alternatively, the user can manually upload the files to huggingface, making sure to upload only the "meta" and "data" folders:

1. Go the HuggingFace website and login
2. Create a new dataset and open it
3. Click "Files and versions"
4. Click "Contribute"
5. Click "Upload Files"
6. Upload the "meta" and "data" folders from your locally saved dataset


### 3. Create a Dataset Version Tag

After the dataset is uploaded, a version tag should be created.

This ensures that training pipelines can reference a stable dataset version.

A helper script called `create_tag.py` is included in the same folder as this README.

Run the script once after the dataset is uploaded.

Example code inside `create_tag.py`:
```
  from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
  from huggingface_hub import HfApi

  hub_api = HfApi()
  hub_api.create_tag("name/dataset_name", tag="v2.1", repo_type="dataset")
```
Replace:
```
  name/dataset_name
```
with your actual Hugging Face dataset name.

Example:
```
  hrc2/kinova_pick_dataset
```
This script should only be run once per dataset version.


### 4. Example Workflow

Typical dataset publishing workflow:

1. Convert rosbag recordings into a LeRobot dataset
2. Verify the dataset locally
3. Push the dataset to Hugging Face
4. Create a version tag using `create_tag.py`

Example:
```
  python3 rosbag2_to_lerobot_many_prompts_with_failure.py ...

  python create_tag.py
```
After this step, the dataset will appear on Hugging Face and can be referenced by version.


### Notes

- Version tags allow training pipelines to use stable dataset snapshots.
- Do not overwrite existing tags unless you intentionally want to change the dataset version.
- Always bump the tag version (for example v2.0 → v2.1) when publishing an updated dataset.
- Only run the `create_tag.py` script once per dataset release.