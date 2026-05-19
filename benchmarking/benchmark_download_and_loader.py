# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
benchmark_download_and_loader.py: Download AMP datasets and benchmark loader performance with fixed config.
"""

from pathlib import Path
import tempfile
import time
import torch

from amp_rsl_rl.utils import AMPLoader, download_amp_dataset_from_hf

# =============================================
# CONFIGURATION (hard-coded)
# =============================================
repo_id = "ami-iit/amp-dataset"
robot_folder = "ergocub"
files = [
    "ergocub_stand_still.npy",
    "ergocub_walk_left0.npy",
    "ergocub_walk.npy",
    "ergocub_walk_right2.npy",
]
dataset_weights = [1.0] * len(files)
device_str = "cuda"  # or "cpu"
simulation_dt = 1.0 / 60.0
slow_down_factor = 1
num_samples = 20
batch_size = 32768

joint_names = [
    "l_ankle_pitch",
    "l_ankle_roll",
    "l_knee",
    "l_hip_yaw",
    "l_hip_roll",
    "l_hip_pitch",
    "r_ankle_pitch",
    "r_ankle_roll",
    "r_knee",
    "r_hip_yaw",
    "r_hip_roll",
    "r_hip_pitch",
    "torso_yaw",
    "torso_roll",
    "torso_pitch",
]


def main():
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # Download into a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        local_dir = Path(tmpdirname)
        print(f"Downloading {len(files)} files to {local_dir}...")
        dataset_names = download_amp_dataset_from_hf(
            destination_dir=local_dir,
            robot_folder=robot_folder,
            files=files,
            repo_id=repo_id,
        )
        print("Downloaded datasets:", dataset_names)

        # Build datasets dictionary using returned dataset_names (without .npy extension)
        datasets = {name: weight for name, weight in zip(dataset_names, dataset_weights)}

        # Initialize loader and measure time
        t0 = time.perf_counter()
        loader = AMPLoader(
            device=device,
            simulation_dt=simulation_dt,
            slow_down_factor=slow_down_factor,
            expected_joint_names=joint_names,
            sources=[{"dataset_path": str(local_dir), "motions": datasets}],
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"Loader initialization took {t1 - t0:.3f} seconds")

        # Sampling performance
        total_batches = max(1, num_samples // batch_size)
        print(f"Sampling {total_batches} batches of size {batch_size}...")
        sampler = loader.feed_forward_generator(total_batches, batch_size)
        # Warm-up (esp. for CUDA)
        for _ in range(2):
            try:
                next(sampler)
            except StopIteration:
                break
        sampler = loader.feed_forward_generator(total_batches, batch_size)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        frames = 0
        for obs, next_obs in sampler:
            frames += obs.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.perf_counter()
        fps = frames / (t3 - t2)
        print(
            f"Sampled {frames} frames in {(t3 - t2) * 1000:.3f} milliseconds → {fps:.1f} frames/s"
        )

        # Reset-state sampling performance
        print(f"Sampling {total_batches} reset-state batches of size {batch_size}...")
        if device.type == "cuda":
            torch.cuda.synchronize()
        t4 = time.perf_counter()
        states = 0
        for _ in range(total_batches):
            loader.get_state_for_reset(batch_size)
            states += batch_size
        if device.type == "cuda":
            torch.cuda.synchronize()
        t5 = time.perf_counter()
        sps = states / (t5 - t4)
        print(
            f"Sampled {states} states in {(t5 - t4) * 1000 :.3f} milliseconds → {sps:.1f} states/s"
        )


if __name__ == "__main__":
    main()
