

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi
import sys

api = HfApi()
repo_id = "Trontour/test2"

# List all tags (like v2.0, v2.1, v3.0)
tags = api.list_repo_refs(repo_id, repo_type="dataset").tags
print("Available tags:", [t.name for t in tags])



# # # repo_id = "Trontour/lerobot_gen3_v21"

# # Load from the Hugging Face Hub (will be cached locally)
dataset = LeRobotDataset(repo_id)
sample = dataset[100]
print("hello")
print(sample, file=sys.stderr)


# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from pathlib import Path
# local_path = "/home/aaquib/pi_ws/openpi-LBMfailure/data/lerobot_gen3_test_dataset_updated"
# dataset = LeRobotDataset(local_path)
# print(dataset[100])

# local_path = Path("/home/aaquib/pi_ws/openpi-LBMfailure/data/test2")
# print("Contents:", [p.name for p in local_path.glob('*')])

# dataset = LeRobotDataset(local_path)