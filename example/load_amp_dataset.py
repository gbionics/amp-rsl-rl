# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Import required libraries
from pathlib import Path  # For path manipulation
import tempfile  # For creating temporary directories
from amp_rsl_rl.utils import (
    AMPLoader,
    download_amp_dataset_from_hf,
)  # Core AMP utilities
import torch  # PyTorch for tensor operations

# =============================================
# CONFIGURATION SECTION
# =============================================
# Define the dataset source and files to download
repo_id = "ami-iit/amp-dataset"  # Hugging Face repository ID
robot_folder = "ergocub"  # Subfolder containing robot-specific datasets

# List of motion dataset files to download
files = [
    "ergocub_stand_still.npy",  # Standing still motion
    "ergocub_walk_left0.npy",  # Walking left motion
    "ergocub_walk.npy",  # Straight walking motion
    "ergocub_walk_right2.npy",  # Walking right motion
]

# =============================================
# DATASET DOWNLOAD AND LOADING
# =============================================
# Create a temporary directory to store downloaded datasets
# This ensures clean up after we're done
with tempfile.TemporaryDirectory() as tmpdirname:
    local_dir = Path(tmpdirname)  # Convert to Path object for easier handling

    # Download datasets from Hugging Face Hub
    # Returns the base names of the downloaded files (without .npy extension)
    dataset_names = download_amp_dataset_from_hf(
        local_dir,  # Where to save the files
        robot_folder=robot_folder,  # Which robot dataset to use
        files=files,  # Which specific motion files to download
        repo_id=repo_id,  # Repository ID on Hugging Face Hub
    )

    # =============================================
    # DATASET PROCESSING WITH AMPLoader
    # =============================================
    # Initialize the AMPLoader to process and manage the motion data
    datasets = {name: 1.0 for name in dataset_names}
    loader = AMPLoader(
        device="cpu",  # Use CPU for processing (change to "cuda" for GPU)
        simulation_dt=1 / 60.0,  # Simulation timestep (60Hz)
        slow_down_factor=1,  # Don't slow down the motions
        expected_joint_names=None,  # Use default joint ordering
        sources=[{"dataset_path": str(local_dir), "motions": datasets}],
    )

    # =============================================
    # EXAMPLE USAGE
    # =============================================
    # Get the first motion sequence from the loader
    motion = loader.motion_data[0]

    # Print basic information about the loaded motion
    print("Loaded dataset with", len(motion), "frames.")

    # Get and print a sample observation (first frame)
    sample_obs = motion.get_amp_dataset_obs(torch.tensor([0]))  # Get frame 0
    print("Sample AMP observation:", sample_obs)

    # The motion data contains:
    # - Joint positions and velocities
    # - Base linear/angular velocities (local and world frames)
    # - Base orientation (quaternion)

    # Typical usage patterns:
    # 1. For training: Use loader.feed_forward_generator() to get batches
    # 2. For reset: Use loader.get_state_for_reset() to initialize robot states
    # 3. For observation: Use motion.get_amp_dataset_obs() to get specific frames

# The temporary directory is automatically deleted when the 'with' block ends
