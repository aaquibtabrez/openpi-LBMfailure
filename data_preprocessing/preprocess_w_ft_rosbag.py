import os
import glob
import time
import numpy as np
import pandas as pd
import cv2
import shutil

base_path = '/data/ros_datasets/peeling/subject1_carrot'


def extract_timestamps(image_files):
    """Extract timestamps from image filenames."""
    timestamps = []
    for f in sorted(image_files):
        try:
            ts = int(os.path.basename(f).split("_")[0])
            timestamps.append((ts, f))
        except Exception as e:
            print(f"Failed to parse timestamp from {f}: {e}")
    return timestamps

def find_closest(ts_target, source_ts_list):
    """Find index of closest timestamp in source_ts_list."""
    idx = np.abs(np.array(source_ts_list) - ts_target).argmin()
    return idx

def align_csv_with_timestamps(pd_data, ref_timestamps):
    """Align CSV rows with reference timestamps."""
    csv_times = pd_data["Time (sec)"].to_numpy()
    # Convert ref_timestamps if needed (ns → s)
    # if np.abs(ref_timestamps[0] - csv_times[0]) > 1e6:
    #     ref_timestamps = [ts / 1e9 for ts in ref_timestamps]
    aligned_rows = []
    for ts in ref_timestamps:
        idx = find_closest(ts, csv_times)
        aligned_rows.append(pd_data.iloc[idx])
    temp_df = pd.DataFrame(aligned_rows)
    temp_df['Time (sec)'] = temp_df['Time (sec)'].astype(int)
    return temp_df

def _is_valid(df):
    """Check if the current environment is valid for processing."""
    # Check for any NaNs
    if df.isnull().values.any():
        return False

    # Check that each column has more than one unique value
    for col in df.columns:
        if df[col].nunique() <= 1:
            print(col)
            return False

    return True

def _parse_example(episode_id):
    validity_check = {'flag': True, 'message': '', 'invalid_data': ['joint', 'gripper']}

    """Parse a single episode, aligning all data to c2 timestamps."""
    print(f"Now processing episode {episode_id}...")

    c1_folder = f"{base_path}_c1/{episode_id}"
    c2_folder = f"{base_path}_c2/{episode_id}"

    force_csv_path = f"{base_path}_force/{episode_id}_fullposes.csv"
    # wrist_csv = f"{base_path}_force/{episode_id}_wrenches_poses_wrist_joint_states.csv"

    if not os.path.exists(c2_folder):
        print(f"Skipping episode {episode_id}: c2 folder not found")
        validity_check['flag'] = False
        validity_check['message'] += f"c2 folder not found for episode {episode_id}. "
        return None

    image_files_c1 = sorted(glob.glob(os.path.join(c1_folder, '*.png')))
    image_files_c2 = sorted(glob.glob(os.path.join(c2_folder, '*.png')))

    if len(image_files_c2) == 0:
        print(f"Skipping episode {episode_id}: no images in c2")
        validity_check['flag'] = False
        validity_check['message'] += f"No images in c2 for episode {episode_id}. "
        return None

    # Extract timestamps
    ts_c1 = extract_timestamps(image_files_c1)
    ts_c2 = extract_timestamps(image_files_c2)

    ts_list_c1 = [ts for ts, _ in ts_c1]
    ts_list_c2 = [ts for ts, _ in ts_c2]

    # Align CSV data to c2
    if os.path.exists(force_csv_path):
        eepose_cols = ['Time (sec)', 'Robot Pose X (m)', 'Robot Pose Y (m)', 'Robot Pose Z (m)', 'Robot Pose Roll (rad)', 'Robot Pose Pitch (rad)', 'Robot Pose Yaw (rad)', 'Robot Pose w (rad)']
        pose_cols = ['Time (sec)', 'Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7', 'Gripper']
        force_cols = ['Time (sec)', 'Force X (N)', 'Force Y (N)', 'Force Z (N)',
                      'Torque X (Nm)', 'Torque Y (Nm)', 'Torque Z (Nm)']
        mocap_cols = ['Time (sec)', 'Mocap Pose X (m)', 'Mocap Pose Y (m)', 'Mocap Pose Z (m)',
                      'Mocap Pose Roll (rad)', 'Mocap Pose Pitch (rad)', 'Mocap Pose Yaw (rad)', 'Mocap Pose w (rad)']
        origin_df = pd.read_csv(force_csv_path)
        origin_force_df = origin_df[force_cols].dropna(subset=force_cols)
        origin_pose_df = origin_df[pose_cols].dropna(subset=pose_cols)
        origin_eepose_df = origin_df[eepose_cols].dropna(subset=eepose_cols)
        if 'mocap' not in validity_check['invalid_data']:
            origin_mocap_df = origin_df[mocap_cols].dropna(subset=mocap_cols)
        
        if 'force' not in validity_check['invalid_data']:
            origin_force_df = origin_df[force_cols].dropna(subset=force_cols)
            origin_force_df = align_csv_with_timestamps(origin_force_df, ts_list_c2)
            origin_force_df = origin_force_df[force_cols] #.to_numpy(dtype=np.float32)

            if not _is_valid(origin_force_df): 
                print(f"[Warning]: Invalid force data in episode {episode_id}, using zeros.")
                validity_check['message'] += f"Invalid force data in episode {episode_id}. "
                return

        if 'joint' not in validity_check['invalid_data']:
            origin_pose_df = origin_df[pose_cols].dropna(subset=pose_cols)
            origin_pose_df = align_csv_with_timestamps(origin_pose_df, ts_list_c2)
            origin_pose_df = origin_pose_df[pose_cols] #.to_numpy(dtype=np.float32)

            if not _is_valid(origin_pose_df):
                print(f"[Warning]: Invalid joint pose data in episode {episode_id}, using zeros.")
                validity_check['message'] += f"Invalid joint pose data in episode {episode_id}. "
                return


        if 'eepose' not in validity_check['invalid_data']:
            origin_eepose_df = origin_df[eepose_cols].dropna(subset=eepose_cols)
            origin_eepose_df = align_csv_with_timestamps(origin_eepose_df, ts_list_c2)
    
            if not _is_valid(origin_eepose_df):
                print(f"[Warning]: Invalid ee pose data in episode {episode_id}, using zeros.")
                validity_check['message'] += f"Invalid ee pose data in episode {episode_id}. "
                return

            origin_eepose_df = origin_eepose_df[eepose_cols]

            if 'joint' not in validity_check['invalid_data']:
                origin_eepose_df['Gripper'] = origin_pose_df['Gripper'].values
            else:
                origin_eepose_df['Gripper'] = 0

        if 'mocap' not in validity_check['invalid_data']:
            # if has mocap
            origin_mocap_df = origin_df[mocap_cols].dropna(subset=mocap_cols)
            origin_mocap_df = align_csv_with_timestamps(origin_mocap_df, ts_list_c2)
            if not _is_valid(origin_mocap_df):
                print(f"[Warning]: Invalid mocap pose data in episode {episode_id}, using zeros.")
                validity_check['message'] += f"Invalid mocap pose data in episode {episode_id}. "
                return
            origin_mocap_df = origin_mocap_df[mocap_cols]

            if 'joint' not in validity_check['invalid_data']:
                origin_mocap_df['Gripper'] = origin_pose_df['Gripper'].values
            else:
                origin_mocap_df['Gripper'] = 0

            clean_mocap_csv_path = force_csv_path.replace('ros_datasets', 'processed_ros_datasets').replace('fullposes', 'mocapposes')
            os.makedirs(os.path.dirname(clean_mocap_csv_path), exist_ok=True)
            origin_mocap_df.to_csv(clean_mocap_csv_path, index=False)
    else:
        print(f"[Warning]: Force CSV file not found for episode {episode_id}, using zeros.")
        validity_check['flag'] = False
        validity_check['message'] += f"Force CSV file not found for episode {episode_id}. "


    if not validity_check['flag']:
        print(f"[Error]: Invalid data for episode {episode_id}, skipping. {validity_check['message']}")
        return None
    
    # ----------- After all checks, proceed with saving data -----------
    if 'joint' not in validity_check['invalid_data']:
        pose_csv_path = force_csv_path.replace('ros_datasets', 'processed_ros_datasets').replace('fullposes', 'jointposes')
        os.makedirs(os.path.dirname(pose_csv_path), exist_ok=True)
        origin_pose_df.to_csv(pose_csv_path, index=False)

    if 'eepose' not in validity_check['invalid_data']:
        clean_eepose_csv_path = force_csv_path.replace('ros_datasets', 'processed_ros_datasets').replace('fullposes', 'eeposes')
        os.makedirs(os.path.dirname(clean_eepose_csv_path), exist_ok=True)
        origin_eepose_df.to_csv(clean_eepose_csv_path, index=False)

    if 'force' not in validity_check['invalid_data']:
        clean_force_csv_path = force_csv_path.replace('ros_datasets', 'processed_ros_datasets').replace('fullposes', 'forces')
        os.makedirs(os.path.dirname(clean_force_csv_path), exist_ok=True)
        origin_force_df.to_csv(clean_force_csv_path, index=False)

    # actions_csv_path = clean_force_csv_path.replace('forces', 'actions')
    # os.makedirs(os.path.dirname(actions_csv_path), exist_ok=True)
    
    # action_df = pd.DataFrame(actions, columns=['Joint 1', 'Joint 2', 'Joint 3', 'Joint 4', 'Joint 5', 'Joint 6', 'Joint 7', 'Gripper'])
    # action_df['Time (sec)'] = origin_pose_df['Time (sec)']
    # action_df.to_csv(actions_csv_path, index=False)

    for i, (ts_c2_val, path_c2) in enumerate(ts_c2):
        print(i)
        idx_c1 = find_closest(ts_c2_val, ts_list_c1) if ts_list_c1 else None
        c2_save_path = path_c2.replace('ros_datasets', 'processed_ros_datasets')
        os.makedirs(os.path.dirname(c2_save_path), exist_ok=True)
        shutil.copy2(path_c2, c2_save_path)

        if idx_c1 is not None:
            path_c1 = ts_c1[idx_c1][1]
            c1_save_path = path_c1.replace('ros_datasets', 'processed_ros_datasets').replace("rgb",str(i))
            print(c1_save_path)
            os.makedirs(os.path.dirname(c1_save_path), exist_ok=True)
            shutil.copy2(path_c1, c1_save_path)
            # print(f"Matched c1 timestamp {ts_list_c1[idx_c1]} to c2 timestamp {ts_c2_val}, time difference: {abs(ts_c2_val - ts_list_c1[idx_c1])}\nc1 path: {ts_c1[idx_c1][1]}")
        else:
            validity_check['flag'] = False
            validity_check['message'] += f"No matching c1 image for c2 timestamp {ts_c2_val}. "
            print(f"No matching c1 image for c2 timestamp {ts_c2_val}")
        img_c2 = cv2.imread(path_c2)
        img_c2 = cv2.cvtColor(img_c2, cv2.COLOR_BGR2RGB)
        img_c2 = cv2.resize(img_c2, (256, 256))
    print(f"\nSuccessfully processed episode {episode_id}...")

if __name__ == "__main__":
    episode_ids= [str(i) for i in range(1, 50)]
    for episode_id in episode_ids:
        _parse_example(episode_id)