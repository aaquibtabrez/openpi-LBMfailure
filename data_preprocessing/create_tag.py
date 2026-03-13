from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi

hub_api = HfApi()
hub_api.create_tag("username/dataset_name", tag="v2.1", repo_type="dataset")