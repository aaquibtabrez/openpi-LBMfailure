# plot_green_success_green_bin.py

from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import os

# ---------------------------------------------------------
# 1. LOAD DATASET (with HuggingFace login support)
# ---------------------------------------------------------

# if your dataset is private, login first:
# run in terminal:   hf login

DATASET_ID = "Trontour/lerobot_gen3_green_container_green_bin_success"   # TODO: replace this

print("Loading dataset...")
ds = load_dataset(DATASET_ID, split="train")

# convert columns to numpy arrays
ds = ds.with_format("numpy")

print("Loaded rows:", len(ds))

# ---------------------------------------------------------
# 2. GROUP DATA INTO EPISODES
# ---------------------------------------------------------
episodes = defaultdict(list)

for sample in ds:
    ep = int(sample["episode_index"])
    episodes[ep].append(sample)

print("Number of episodes:", len(episodes))


# ---------------------------------------------------------
# 3. SORT EPISODES BY FRAME INDEX
# ---------------------------------------------------------
for ep in episodes:
    episodes[ep] = sorted(episodes[ep], key=lambda x: int(x["frame_index"]))


# ---------------------------------------------------------
# 4. BUILD JOINT TRAJECTORY MATRICES (T × 7)
# ---------------------------------------------------------
episode_trajs = {}

for ep, samples in episodes.items():
    traj = np.stack([s["joint_position"] for s in samples], axis=0)
    episode_trajs[ep] = traj


# ---------------------------------------------------------
# 5. PLOT: ONE FIGURE PER JOINT, 50 EPISODES PER PLOT
# ---------------------------------------------------------

def plot_joint_trajectories_per_joint(episode_trajs, max_eps=50, save_dir="plots", show=False):
    import os
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)

    # number of joints
    first_ep = next(iter(episode_trajs.values()))
    num_joints = first_ep.shape[1]

    for joint_idx in range(num_joints):
        plt.figure(figsize=(12, 6))

        for i, (ep, traj) in enumerate(episode_trajs.items()):
            if i >= max_eps:
                break

            # plot trajectory for this joint
            plt.plot(traj[:, joint_idx], alpha=0.55, linewidth=1.3)

            # annotate episode ID at the END of the line
            plt.annotate(
                str(ep),
                xy=(len(traj) - 1, traj[-1, joint_idx]),
                fontsize=7,
                alpha=0.75
            )

        plt.title(f"Joint {joint_idx} — {max_eps} episodes (labeled by episode #)")
        plt.xlabel("Frame index")
        plt.ylabel("Joint position")
        plt.grid(True)

        # save figure
        out_path = os.path.join(save_dir, f"joint_{joint_idx}.png")
        plt.savefig(out_path, dpi=200)
        print(f"Saved: {out_path}")

        if show:
            plt.show()
        else:
            plt.close()



# ---------------------------------------------------------
# 6. CALL PLOTTER
# ---------------------------------------------------------
print("Generating plots...")
plot_joint_trajectories_per_joint(episode_trajs, max_eps=50, save_dir="plots", show=False)
print("Done.")

