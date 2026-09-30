# AMP-RSL-RL

AMP-RSL-RL is a reinforcement learning library that extends the Proximal Policy Optimization (PPO) implementation of [RSL-RL](https://github.com/leggedrobotics/rsl_rl) to incorporate Adversarial Motion Priors (AMP). This framework enables humanoid agents to learn motor skills from motion capture data using adversarial imitation learning techniques.

---

## 📦 Installation

The repository is available on PyPI under the package name **amp-rl-rsl**. You can install it directly using pip:

```bash
pip install amp-rsl-rl
```

Alternatively, if you prefer to clone the repository and install it locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/gbionics/amp_rsl_rl.git
    cd amp_rsl_rl
    ```

2. Install the package:
    ```bash
    pip install .
    ```

For editable/development mode:

```bash
pip install -e .
```

If you want to run the examples, please install with:

```bash
pip install .[examples]
```

The required dependencies include:

- `numpy`
- `scipy`
- `torch`
- `rsl-rl-lib`

These will be automatically installed via pip.

---

## 📂 Project Structure

```
amp_rsl_rl/
│
├── algorithms/        # AMP and PPO implementations
├── networks/          # Neural networks for policy and discriminator
├── runners/           # Training and evaluation routines
├── storage/           # Replay buffer for experience collection
├── utils/             # Dataset loaders and motion tools
```

---

## 📁 Dataset Structure

AMP-RSL-RL expects each motion file to be a `.npy` containing a Python dictionary.

### Required keys (always used)

- **`joints_list`**: `List[str]`  
  Joint names in the same order expected by the robot configuration.

- **`joint_positions`**: array-like, shape `(T, N_joints)`  
  Joint positions over time.

- **`root_position`**: array-like, shape `(T, 3)`  
  Base position in world frame.

- **`root_quaternion`**: array-like, shape `(T, 4)`  
  Base orientation in quaternion **`xyzw`** format (SciPy convention).

- **`fps`**: `float` (or scalar array)  
  Original dataset framerate. The loader resamples to simulator `dt`.

### Optional keys (needed for body keyframe observations)

These keys are required only if you enable any of:
`body_pos_b`, `body_ori_b`, `body_lin_vel_b`, `body_ang_vel_b` in `amp_obs_components`.

- **`body_links_list`**: `List[str]`  
  Names of links for which body keyframes are provided.

- **`body_links_pos_w`**: array-like, shape `(T, N_bodies, 3)`  
  Body positions in world frame.

- **`body_links_quat_w`**: array-like, shape `(T, N_bodies, 4)`  
  Body orientations in world frame, quaternion **`xyzw`** format.

### Configuration fields for body keyframes

When using body keyframe components, set these in `dataset_cfg`:

- **`amp_obs_components`**: ordered list of discriminator components to concatenate.
- **`body_links_names`**: ordered list of body names to use from the dataset.
- **`anchor_body_name`**: reference body used to express relative body observations.

Notes:

- `anchor_body_name` must be present in `body_links_names`.
- The anchor body is excluded from the final body observation vectors.
- For backward compatibility, if `amp_obs_components` is not provided, the default is
  `["joint_pos", "joint_vel", "base_lin_vel", "base_ang_vel"]`.

---

## � Symmetry Augmentation
AMP-RSL-RL now exposes the symmetry-aware data augmentation and mirror-loss hooks from
[RSL-RL](https://github.com/leggedrobotics/rsl_rl). The implementation follows the design
described in:
> Mittal, M., Rudin, N., Klemm, V., Allshire, A., & Hutter, M. (2024).<br>
> *Symmetry Considerations for Learning Task Symmetric Robot Policies*. In IEEE International Conference on Robotics and Automation (ICRA).<br>
> https://doi.org/10.1109/ICRA57147.2024.10611493
Symmetry augmentation can be enabled through the `symmetry_cfg` section of the algorithm
configuration, providing both minibatch augmentation and optional mirror-loss regularisation
for the policy update. AMP-specific components (the discriminator and expert/policy motion
buffers) are augmented using the same configuration so that style rewards and adversarial
training remain consistent with their symmetric counterparts.

---

### Example

Example dataset dictionary with optional body keyframes:

```python
{
    "joints_list": ["hip", "knee", "ankle"],
    "joint_positions": np.ndarray(shape=(T, 3)),
    "root_position": np.ndarray(shape=(T, 3)),
    "root_quaternion": np.ndarray(shape=(T, 4)),  # xyzw
    "fps": 120.0,

    # Optional: required only for body_* discriminator components
    "body_links_list": ["root_link", "left_knee", "right_knee"],
    "body_links_pos_w": np.ndarray(shape=(T, 3, 3)),
    "body_links_quat_w": np.ndarray(shape=(T, 3, 4)),  # xyzw
}
```

`T` is the number of frames. All time-dependent fields must share the same `T`.

Example `dataset_cfg` enabling body keyframe observations:

```python
dataset_cfg = {
    "amp_data_path": "/path/to/datasets",
    "datasets": {"walk": 1.0},
    "slow_down_factor": 1,
    "amp_obs_components": [
        "joint_pos",
        "joint_vel",
        "base_lin_vel",
        "base_ang_vel",
        "body_pos_b",
        "body_ori_b",
    ],
    "body_links_names": ["root_link", "left_knee", "right_knee"],
    "anchor_body_name": "root_link",
}
```

---

## 📚 Supported Dataset

For a ready-to-use motion capture dataset, you can use the [AMP Dataset on Hugging Face](https://huggingface.co/datasets/ami-iit/amp-dataset). This dataset is curated to work seamlessly with the AMP-RSL-RL framework.

---

## 🧑‍💻 Authors

- **Giulio Romualdi** – [@GiulioRomualdi](https://github.com/GiulioRomualdi)
- **Giuseppe L'Erario** – [@Giulero](https://github.com/Giulero)

---

## 📄 License

BSD 3-Clause License © 2025 Istituto Italiano di Tecnologia
