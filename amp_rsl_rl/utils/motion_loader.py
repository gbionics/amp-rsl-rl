# SPDX-FileCopyrightText: Generative Bionics S.R.L.
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2025, Istituto Italiano di Tecnologia
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from enum import Enum
import inspect
from pathlib import Path
from typing import List, Union, Tuple, Generator, Dict, Optional, Any
from dataclasses import dataclass
from amp_rsl_rl.utils._compat import RSL_RL_V3_3_PLUS

if RSL_RL_V3_3_PLUS:
    from rsl_rl.utils import resolve_callable
else:
    from rsl_rl.utils import string_to_callable as resolve_callable

import torch
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d


class VelocityRepresentation(Enum):
    """Velocity representation convention for base velocities.

    Attributes:
        BODY_FIXED_REPRESENTATION: Linear and angular velocities expressed in the
            body-fixed (local) frame, i.e. using axes that rotate with the body.
        MIXED_REPRESENTATION: Linear and angular velocities expressed using
            world-aligned axes, but taken at the body origin. In other words, the
            reference frame is attached to the body origin while its orientation
            remains aligned with the world frame. This differs from
            BODY_FIXED_REPRESENTATION, where the axes are attached to and rotate
            with the body.
    """

    BODY_FIXED_REPRESENTATION = "body_fixed"
    MIXED_REPRESENTATION = "mixed"


def download_amp_dataset_from_hf(
    destination_dir: Path,
    robot_folder: str,
    files: list,
    repo_id: str = "ami-iit/amp-dataset",
) -> list:
    """
    Downloads AMP dataset files from Hugging Face and saves them to `destination_dir`.
    Ensures real file copies (not symlinks or hard links).

    Args:
        destination_dir (Path): Local directory to save the files.
        robot_folder (str): Folder in the Hugging Face dataset repo to pull from.
        files (list): List of filenames to download.
        repo_id (str): Hugging Face repository ID. Default is "ami-iit/amp-dataset".

    Returns:
        List[str]: List of dataset names (without .npy extension).
    """
    from huggingface_hub import hf_hub_download

    destination_dir.mkdir(parents=True, exist_ok=True)
    dataset_names = []

    for file in files:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{robot_folder}/{file}",
            repo_type="dataset",
            local_files_only=False,
        )
        local_copy = destination_dir / file
        # Deep copy to avoid symlinks
        with open(file_path, "rb") as src_file, open(local_copy, "wb") as dst_file:
            dst_file.write(src_file.read())
        dataset_names.append(file.replace(".npy", ""))

    return dataset_names


def _call_augmentation_func(
    fn,
    *,
    obs: Optional[torch.Tensor] = None,
    actions: Optional[torch.Tensor] = None,
    obs_type: Any = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    parameters = inspect.signature(fn).parameters
    kwargs: Dict[str, Any] = {}

    if "obs" in parameters:
        kwargs["obs"] = obs

    if "actions" in parameters:
        kwargs["actions"] = actions

    if "obs_type" in parameters and obs_type is not None:
        kwargs["obs_type"] = obs_type

    result = fn(**kwargs)

    if isinstance(result, tuple):
        return result

    # Returning a tuple for consistency: (aug_obs, aug_actions)
    return result, None


_DEFAULT_OBS_COMPONENTS = ["joint_pos", "joint_vel", "base_lin_vel", "base_ang_vel"]


@dataclass
class MotionData:
    """
    Data class representing motion data for humanoid agents.

    This class stores joint positions and velocities, base velocities (both in local
    and mixed/world frames), and base orientation (as quaternion). It offers utilities
    for preparing data in AMP-compatible format, as well as environment reset states.

    Attributes:
        - joint_positions: shape (T, N)
        - joint_velocities: shape (T, N)
        - base_lin_velocities_mixed: linear velocity in world frame
        - base_ang_velocities_mixed: (currently zeros)
        - base_lin_velocities_local: linear velocity in local (body) frame
        - base_ang_velocities_local: (currently zeros)
        - base_quat: orientation quaternion as torch.Tensor in wxyz order
        - body_pos_b: (optional) relative body positions w.r.t. anchor, shape (T, (N_bodies-1)*3)
        - body_ori_b: (optional) relative body orientations w.r.t. anchor (6D repr), shape (T, (N_bodies-1)*6)
        - body_lin_vel_b: (optional) relative body linear velocities, shape (T, (N_bodies-1)*3)
        - body_ang_vel_b: (optional) relative body angular velocities, shape (T, (N_bodies-1)*3)

    Notes:
        - The quaternion is expected in the dataset as `xyzw` format (SciPy default),
          and it is converted internally to `wxyz` format to be compatible with IsaacLab conventions.
        - All data is converted to torch.Tensor on the specified device during initialization.
    """

    joint_positions: Union[torch.Tensor, np.ndarray]
    joint_velocities: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_mixed: Union[torch.Tensor, np.ndarray]
    base_lin_velocities_local: Union[torch.Tensor, np.ndarray]
    base_ang_velocities_local: Union[torch.Tensor, np.ndarray]
    base_quat: Union[Rotation, torch.Tensor]
    device: torch.device = torch.device("cpu")
    body_pos_b: Optional[Union[torch.Tensor, np.ndarray]] = None
    body_ori_b: Optional[Union[torch.Tensor, np.ndarray]] = None
    body_lin_vel_b: Optional[Union[torch.Tensor, np.ndarray]] = None
    body_ang_vel_b: Optional[Union[torch.Tensor, np.ndarray]] = None

    def __post_init__(self) -> None:
        # Convert numpy arrays (or SciPy Rotations) to torch tensors
        def to_tensor(x):
            return torch.tensor(x, device=self.device, dtype=torch.float32)

        if isinstance(self.joint_positions, np.ndarray):
            self.joint_positions = to_tensor(self.joint_positions)
        if isinstance(self.joint_velocities, np.ndarray):
            self.joint_velocities = to_tensor(self.joint_velocities)
        if isinstance(self.base_lin_velocities_mixed, np.ndarray):
            self.base_lin_velocities_mixed = to_tensor(self.base_lin_velocities_mixed)
        if isinstance(self.base_ang_velocities_mixed, np.ndarray):
            self.base_ang_velocities_mixed = to_tensor(self.base_ang_velocities_mixed)
        if isinstance(self.base_lin_velocities_local, np.ndarray):
            self.base_lin_velocities_local = to_tensor(self.base_lin_velocities_local)
        if isinstance(self.base_ang_velocities_local, np.ndarray):
            self.base_ang_velocities_local = to_tensor(self.base_ang_velocities_local)
        if isinstance(self.base_quat, Rotation):
            quat_xyzw = self.base_quat.as_quat()  # (T,4) xyzw
            # convert to wxyz
            self.base_quat = torch.tensor(
                quat_xyzw[:, [3, 0, 1, 2]],
                device=self.device,
                dtype=torch.float32,
            )
        if isinstance(self.body_pos_b, np.ndarray):
            self.body_pos_b = to_tensor(self.body_pos_b)
        if isinstance(self.body_ori_b, np.ndarray):
            self.body_ori_b = to_tensor(self.body_ori_b)
        if isinstance(self.body_lin_vel_b, np.ndarray):
            self.body_lin_vel_b = to_tensor(self.body_lin_vel_b)
        if isinstance(self.body_ang_vel_b, np.ndarray):
            self.body_ang_vel_b = to_tensor(self.body_ang_vel_b)

    def __len__(self) -> int:
        return self.joint_positions.shape[0]

    def get_amp_dataset_obs(
        self,
        indices: torch.Tensor,
        velocity_representation: VelocityRepresentation = VelocityRepresentation.BODY_FIXED_REPRESENTATION,
        obs_components: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Returns the AMP observation tensor for given indices.

        Args:
            indices: indices of samples to retrieve
            velocity_representation: which frame convention to use for the base
                velocities in the observation. Defaults to BODY_FIXED_REPRESENTATION.
            obs_components: list of observation component names to include.
                Valid values: "joint_pos", "joint_vel", "base_lin_vel", "base_ang_vel",
                "body_pos_b", "body_ori_b", "body_lin_vel_b", "body_ang_vel_b".
                If None, defaults to ["joint_pos", "joint_vel", "base_lin_vel", "base_ang_vel"].

        Returns:
            Concatenated observation tensor
        """
        if obs_components is None:
            obs_components = _DEFAULT_OBS_COMPONENTS

        use_mixed = (
            velocity_representation == VelocityRepresentation.MIXED_REPRESENTATION
        )

        component_map = {
            "joint_pos": self.joint_positions,
            "joint_vel": self.joint_velocities,
            "base_lin_vel": (
                self.base_lin_velocities_mixed
                if use_mixed
                else self.base_lin_velocities_local
            ),
            "base_ang_vel": (
                self.base_ang_velocities_mixed
                if use_mixed
                else self.base_ang_velocities_local
            ),
            "body_pos_b": self.body_pos_b,
            "body_ori_b": self.body_ori_b,
            "body_lin_vel_b": self.body_lin_vel_b,
            "body_ang_vel_b": self.body_ang_vel_b,
        }

        parts = []
        for comp in obs_components:
            tensor = component_map.get(comp)
            if tensor is None:
                raise ValueError(
                    f"Observation component '{comp}' is not available in this MotionData. "
                    f"Available components: {[k for k, v in component_map.items() if v is not None]}"
                )
            parts.append(tensor[indices])

        return torch.cat(parts, dim=1)

    def get_state_for_reset(
        self,
        indices: torch.Tensor,
        velocity_representation: VelocityRepresentation = VelocityRepresentation.BODY_FIXED_REPRESENTATION,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Returns the full state needed for environment reset.

        Args:
            indices: indices of samples to retrieve
            velocity_representation: which frame convention to use for the returned
                base velocities. Defaults to BODY_FIXED_REPRESENTATION for backward
                compatibility.

        Returns:
            Tuple of (quat, joint_positions, joint_velocities, base_lin_velocities, base_ang_velocities)
        """
        if velocity_representation == VelocityRepresentation.MIXED_REPRESENTATION:
            return (
                self.base_quat[indices],
                self.joint_positions[indices],
                self.joint_velocities[indices],
                self.base_lin_velocities_mixed[indices],
                self.base_ang_velocities_mixed[indices],
            )
        return (
            self.base_quat[indices],
            self.joint_positions[indices],
            self.joint_velocities[indices],
            self.base_lin_velocities_local[indices],
            self.base_ang_velocities_local[indices],
        )

    def get_random_sample_for_reset(
        self,
        items: int = 1,
        velocity_representation: VelocityRepresentation = VelocityRepresentation.BODY_FIXED_REPRESENTATION,
    ) -> Tuple[torch.Tensor, ...]:
        indices = torch.randint(0, len(self), (items,), device=self.device)
        return self.get_state_for_reset(indices, velocity_representation)


class AMPLoader:
    """
    Loader and processor for humanoid motion capture datasets in AMP format.

    Responsibilities:
      - Loading .npy files containing motion data
      - Building a unified joint ordering across all datasets
      - Resampling trajectories to match the simulator's timestep
      - Computing derived quantities (velocities, local-frame motion)
      - Returning torch-friendly MotionData instances

    Dataset format:
        Each .npy contains a dict with keys:
          - "joints_list": List[str]
          - "joint_positions": List[np.ndarray]
          - "root_position": List[np.ndarray]
          - "root_quaternion": List[np.ndarray] (xyzw)
          - "fps": float (frames/sec)

    Args:
        device: Target torch device ('cpu' or 'cuda')
        dataset_path_root: Directory containing the .npy motion files
        datasets: Dictionary mapping dataset names (without extension) to sampling weights (floats)
        simulation_dt: Timestep used by the simulator
        slow_down_factor: Integer factor to slow down original data
        expected_joint_names: (Optional) override for joint ordering
    """

    def __init__(
        self,
        device: str,
        dataset_path_root: Path,
        datasets: Dict[str, float],
        simulation_dt: float,
        slow_down_factor: int,
        expected_joint_names: Union[List[str], None] = None,
        symmetry_cfg: Optional[Dict[str, Any]] = None,
        velocity_representation: VelocityRepresentation = VelocityRepresentation.BODY_FIXED_REPRESENTATION,
        amp_obs_components: Optional[List[str]] = None,
        body_links_names: Optional[List[str]] = None,
        anchor_body_name: Optional[str] = None,
    ) -> None:
        self.device = device
        self.velocity_representation = velocity_representation
        self.obs_components = amp_obs_components
        if isinstance(dataset_path_root, str):
            dataset_path_root = Path(dataset_path_root)

        # ─── Validate body keyframe configuration ───
        _body_components = {"body_pos_b", "body_ori_b", "body_lin_vel_b", "body_ang_vel_b"}
        self._needs_body_keyframes = (
            self.obs_components is not None
            and bool(_body_components & set(self.obs_components))
        )
        if self._needs_body_keyframes:
            if body_links_names is None:
                raise ValueError(
                    "body_links_names must be provided when using body keyframe "
                    f"observation components: {_body_components & set(self.obs_components)}"
                )
            if anchor_body_name is None:
                raise ValueError(
                    "anchor_body_name must be provided when using body keyframe "
                    "observation components."
                )
            if anchor_body_name not in body_links_names:
                raise ValueError(
                    f"anchor_body_name '{anchor_body_name}' must be present in "
                    f"body_links_names: {body_links_names}"
                )
        self.body_links_names = body_links_names
        self.anchor_body_name = anchor_body_name
        # ─────────────────────────────────────────────

        # ─── Check symmetry augmentation configuration ───
        if symmetry_cfg is not None:
            aug_fn = symmetry_cfg.get("amp_dataset_augmentation_func", None)
            if isinstance(aug_fn, str):
                symmetry_cfg["amp_dataset_augmentation_func"] = resolve_callable(aug_fn)
            aug_fn = symmetry_cfg.get("amp_dataset_augmentation_func", None)
            if aug_fn is not None and not callable(aug_fn):
                raise ValueError(
                    "Symmetry configuration exists but the function is not callable: "
                    f"{aug_fn}"
                )
        self.symmetry_cfg = symmetry_cfg
        # ─────────────────────────────────────────────────

        # ─── Parse dataset names and weights ───
        dataset_names = list(datasets.keys())
        dataset_weights = list(datasets.values())

        # ─── Build union of all joint names if not provided ───
        if expected_joint_names is None:
            joint_union: List[str] = []
            seen = set()
            for name in dataset_names:
                p = dataset_path_root / f"{name}.npy"
                info = np.load(str(p), allow_pickle=True).item()
                for j in info["joints_list"]:
                    if j not in seen:
                        seen.add(j)
                        joint_union.append(j)
            expected_joint_names = joint_union
        # ─────────────────────────────────────────────────────────

        # Load and process each dataset into MotionData
        self.motion_data: List[MotionData] = []
        for dataset_name in dataset_names:
            dataset_path = dataset_path_root / f"{dataset_name}.npy"
            md = self.load_data(
                dataset_path,
                simulation_dt,
                slow_down_factor,
                expected_joint_names,
                body_links_names=self.body_links_names,
                anchor_body_name=self.anchor_body_name,
            )
            self.motion_data.append(md)

        # Normalize dataset-level sampling weights
        weights = torch.tensor(dataset_weights, dtype=torch.float32, device=self.device)
        self.dataset_weights = weights / weights.sum()

        # Precompute flat buffers for fast sampling
        obs_list, next_obs_list, reset_states = [], [], []
        augmented_lengths: List[int] = []
        original_lengths: List[int] = []
        for data, w in zip(self.motion_data, self.dataset_weights):
            T = len(data)
            idx = torch.arange(T, device=self.device)
            obs = data.get_amp_dataset_obs(
                idx, self.velocity_representation, self.obs_components
            )
            next_idx = torch.clamp(idx + 1, max=T - 1)
            next_obs = data.get_amp_dataset_obs(
                next_idx, self.velocity_representation, self.obs_components
            )

            if self.symmetry_cfg and self.symmetry_cfg.get(
                "use_amp_dataset_augmentation", False
            ):
                obs = self._apply_symmetry(obs=obs, obs_type="amp")
                next_obs = self._apply_symmetry(obs=next_obs, obs_type="amp")

            obs_list.append(obs)
            next_obs_list.append(next_obs)
            augmented_lengths.append(obs.shape[0])

            quat, jp, jv, blv, bav = data.get_state_for_reset(
                idx, self.velocity_representation
            )
            reset_states.append(torch.cat([quat, jp, jv, blv, bav], dim=1))
            original_lengths.append(T)

        self.all_obs = torch.cat(obs_list, dim=0)
        self.all_next_obs = torch.cat(next_obs_list, dim=0)
        self.all_states = torch.cat(reset_states, dim=0)

        # Build per-frame sampling weights for obs: weight_i / length_i
        per_frame = torch.cat(
            [
                torch.full((L,), w / L, device=self.device)
                for w, L in zip(self.dataset_weights, augmented_lengths)
            ]
        )
        self.per_frame_weights = per_frame / per_frame.sum()

        # Build separate weights for reset states (not augmented)
        per_frame_reset = torch.cat(
            [
                torch.full((L,), w / L, device=self.device)
                for w, L in zip(self.dataset_weights, original_lengths)
            ]
        )
        self.per_frame_weights_reset = per_frame_reset / per_frame_reset.sum()

    def _apply_symmetry(
        self,
        *,
        obs: Optional[torch.Tensor],
        obs_type: Optional[Any] = None,
    ) -> Optional[torch.Tensor]:
        if self.symmetry_cfg is None:
            return obs

        aug_fn = self.symmetry_cfg.get("amp_dataset_augmentation_func", None)
        if aug_fn is None:
            return obs

        aug_obs, _ = _call_augmentation_func(aug_fn, obs=obs, obs_type=obs_type)

        return aug_obs if aug_obs is not None else obs

    def _resample_data_Rn(
        self,
        data: List[np.ndarray],
        original_keyframes,
        target_keyframes,
    ) -> np.ndarray:
        f = interp1d(original_keyframes, data, axis=0)
        return f(target_keyframes)

    def _resample_data_SO3(
        self,
        raw_quaternions: List[np.ndarray],
        original_keyframes,
        target_keyframes,
    ) -> Rotation:

        # the quaternion is expected in the dataset as `xyzw` format (SciPy default)
        tmp = Rotation.from_quat(raw_quaternions)
        slerp = Slerp(original_keyframes, tmp)
        return slerp(target_keyframes)

    def _compute_ang_vel(
        self,
        data: List[Rotation],
        dt: float,
        local: bool = False,
    ) -> np.ndarray:
        R_prev = data[:-1]
        R_next = data[1:]

        if local:
            # Exp = R_i⁻¹ · R_{i+1}
            rel = R_prev.inv() * R_next
        else:
            # Exp = R_{i+1} · R_i⁻¹
            rel = R_next * R_prev.inv()

        # Log-map to rotation vectors and divide by Δt
        rotvec = rel.as_rotvec() / dt

        return np.vstack((rotvec, rotvec[-1]))

    def _compute_raw_derivative(self, data: np.ndarray, dt: float) -> np.ndarray:
        d = (data[1:] - data[:-1]) / dt
        return np.vstack([d, d[-1:]])

    def load_data(
        self,
        dataset_path: Path,
        simulation_dt: float,
        slow_down_factor: int = 1,
        expected_joint_names: Union[List[str], None] = None,
        body_links_names: Optional[List[str]] = None,
        anchor_body_name: Optional[str] = None,
    ) -> MotionData:
        """
        Loads and processes one motion dataset.

        Args:
            dataset_path: Path to the .npy motion file.
            simulation_dt: Target simulation timestep.
            slow_down_factor: Integer factor to slow down original data.
            expected_joint_names: Joint ordering to use.
            body_links_names: (Optional) List of body link names to extract from
                the dataset. Required when using body keyframe observations.
            anchor_body_name: (Optional) Name of the anchor body in body_links_names.
                Required when using body keyframe observations.

        Returns:
            MotionData instance
        """
        data = np.load(str(dataset_path), allow_pickle=True).item()
        dataset_joint_names = data["joints_list"]

        # build index map for expected_joint_names
        idx_map: List[Union[int, None]] = []
        for j in expected_joint_names:
            if j in dataset_joint_names:
                idx_map.append(dataset_joint_names.index(j))
            else:
                idx_map.append(None)

        # reorder & fill joint positions
        jp_list: List[np.ndarray] = []
        for frame in data["joint_positions"]:
            arr = np.zeros((len(idx_map),), dtype=frame.dtype)
            for i, src_idx in enumerate(idx_map):
                if src_idx is not None:
                    arr[i] = frame[src_idx]
            jp_list.append(arr)

        fps_val = float(np.asarray(data["fps"]).flat[0])
        dt = 1.0 / fps_val / float(slow_down_factor)
        T = len(jp_list)
        t_orig = np.linspace(0, T * dt, T)
        T_new = int(T * dt / simulation_dt)
        t_new = np.linspace(0, T * dt, T_new)

        resampled_joint_positions = self._resample_data_Rn(jp_list, t_orig, t_new)
        resampled_joint_velocities = self._compute_raw_derivative(
            resampled_joint_positions, simulation_dt
        )

        resampled_base_positions = self._resample_data_Rn(
            data["root_position"], t_orig, t_new
        )
        resampled_base_orientations = self._resample_data_SO3(
            data["root_quaternion"], t_orig, t_new
        )

        resampled_base_lin_vel_mixed = self._compute_raw_derivative(
            resampled_base_positions, simulation_dt
        )

        resampled_base_ang_vel_mixed = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=False
        )

        resampled_base_lin_vel_local = np.stack(
            [
                R.as_matrix().T @ v
                for R, v in zip(
                    resampled_base_orientations, resampled_base_lin_vel_mixed
                )
            ]
        )
        resampled_base_ang_vel_local = self._compute_ang_vel(
            resampled_base_orientations, simulation_dt, local=True
        )

        # ─── Process body keyframes if requested ───
        body_pos_b = None
        body_ori_b = None
        body_lin_vel_b = None
        body_ang_vel_b = None

        if self._needs_body_keyframes and body_links_names is not None:
            if "body_links_list" not in data:
                raise KeyError(
                    f"Dataset '{dataset_path}' does not contain 'body_links_list'. "
                    "Body keyframe observation components require datasets with "
                    "'body_links_list', 'body_links_pos_w', and 'body_links_quat_w' keys."
                )

            dataset_body_names = list(data["body_links_list"])

            # Build index map for body_links_names
            body_idx_map: List[int] = []
            for name in body_links_names:
                if name not in dataset_body_names:
                    raise ValueError(
                        f"Body link '{name}' not found in dataset '{dataset_path}'. "
                        f"Available body links: {dataset_body_names}"
                    )
                body_idx_map.append(dataset_body_names.index(name))

            # Extract body positions and quaternions (xyzw format in dataset)
            raw_body_pos_w = np.array(
                data["body_links_pos_w"][:, body_idx_map, :], dtype=np.float64
            )
            raw_body_quat_xyzw = np.array(
                data["body_links_quat_w"][:, body_idx_map, :], dtype=np.float64
            )

            # Resample body positions: shape (T_new, N_bodies, 3)
            n_bodies = len(body_links_names)
            resampled_body_pos_w = np.zeros((T_new, n_bodies, 3), dtype=np.float64)
            for b in range(n_bodies):
                resampled_body_pos_w[:, b, :] = self._resample_data_Rn(
                    raw_body_pos_w[:, b, :], t_orig, t_new
                )

            # Resample body quaternions via SLERP: shape (T_new, N_bodies, 4) as Rotation
            resampled_body_rotations = []  # list of Rotation objects, one per body
            for b in range(n_bodies):
                rot_b = self._resample_data_SO3(
                    raw_body_quat_xyzw[:, b, :], t_orig, t_new
                )
                resampled_body_rotations.append(rot_b)

            # Find anchor body index
            anchor_idx = body_links_names.index(anchor_body_name)

            # Get anchor rotation matrices: shape (T_new, 3, 3)
            anchor_rot_matrices = resampled_body_rotations[anchor_idx].as_matrix()
            # Anchor positions: shape (T_new, 3)
            anchor_pos = resampled_body_pos_w[:, anchor_idx, :]

            # Compute relative transforms for non-anchor bodies
            non_anchor_indices = [i for i in range(n_bodies) if i != anchor_idx]
            n_non_anchor = len(non_anchor_indices)

            # body_pos_b: relative position in anchor frame, shape (T_new, n_non_anchor, 3)
            rel_pos_b = np.zeros((T_new, n_non_anchor, 3), dtype=np.float64)
            # body_ori_b: relative orientation as 6D (first 2 cols of rot matrix),
            # shape (T_new, n_non_anchor, 6)
            rel_ori_b = np.zeros((T_new, n_non_anchor, 6), dtype=np.float64)

            for i, body_idx in enumerate(non_anchor_indices):
                body_pos = resampled_body_pos_w[:, body_idx, :]
                body_rot_matrices = resampled_body_rotations[body_idx].as_matrix()

                for t in range(T_new):
                    R_anchor_T = anchor_rot_matrices[t].T
                    # Relative position: R_anchor^T @ (p_body - p_anchor)
                    rel_pos_b[t, i, :] = R_anchor_T @ (body_pos[t] - anchor_pos[t])
                    # Relative orientation: R_anchor^T @ R_body
                    R_rel = R_anchor_T @ body_rot_matrices[t]
                    # 6D representation: first 2 columns of rotation matrix
                    rel_ori_b[t, i, :3] = R_rel[:, 0]
                    rel_ori_b[t, i, 3:] = R_rel[:, 1]

            # Compute velocities via numerical differentiation
            rel_pos_b_flat = rel_pos_b.reshape(T_new, n_non_anchor * 3)
            rel_ori_b_flat = rel_ori_b.reshape(T_new, n_non_anchor * 6)

            body_pos_b = rel_pos_b_flat
            body_ori_b = rel_ori_b_flat
            body_lin_vel_b = self._compute_raw_derivative(rel_pos_b_flat, simulation_dt)
            body_ang_vel_b = self._compute_raw_derivative(rel_ori_b_flat, simulation_dt)
        # ─────────────────────────────────────────────────

        return MotionData(
            joint_positions=resampled_joint_positions,
            joint_velocities=resampled_joint_velocities,
            base_lin_velocities_mixed=resampled_base_lin_vel_mixed,
            base_ang_velocities_mixed=resampled_base_ang_vel_mixed,
            base_lin_velocities_local=resampled_base_lin_vel_local,
            base_ang_velocities_local=resampled_base_ang_vel_local,
            base_quat=resampled_base_orientations,
            device=self.device,
            body_pos_b=body_pos_b,
            body_ori_b=body_ori_b,
            body_lin_vel_b=body_lin_vel_b,
            body_ang_vel_b=body_ang_vel_b,
        )

    def feed_forward_generator(
        self, num_mini_batch: int, mini_batch_size: int
    ) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
        """
        Yields mini-batches of (state, next_state) pairs for training,
        sampled directly from precomputed buffers.

        Args:
            num_mini_batch: Number of mini-batches to yield
            mini_batch_size: Size of each mini-batch
        Yields:
            Tuple of (state, next_state) tensors
        """
        for _ in range(num_mini_batch):
            idx = torch.multinomial(
                self.per_frame_weights, mini_batch_size, replacement=True
            )
            yield self.all_obs[idx], self.all_next_obs[idx]

    def get_state_for_reset(
        self,
        number_of_samples: int,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Randomly samples full states for environment resets,
        sampled directly from the precomputed state buffer.

        The velocity representation used is the one specified at construction time.

        Args:
            number_of_samples: Number of samples to retrieve
        Returns:
            Tuple of (quat, joint_positions, joint_velocities, base_lin_velocities, base_ang_velocities)
        """
        idx = torch.multinomial(
            self.per_frame_weights_reset, number_of_samples, replacement=True
        )
        full = self.all_states[idx]
        joint_dim = self.motion_data[0].joint_positions.shape[1]

        # The dimensions of the full state are:
        #   - 4 (quat) + joint_dim (joint_positions) + joint_dim (joint_velocities)
        #   + 3 (base_lin_velocities) + 3 (base_ang_velocities)
        #   = 4 + joint_dim + joint_dim + 3 + 3
        dims = [4, joint_dim, joint_dim, 3, 3]
        return torch.split(full, dims, dim=1)
