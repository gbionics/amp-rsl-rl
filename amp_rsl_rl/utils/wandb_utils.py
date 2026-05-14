import os
import warnings
import wandb
from rsl_rl.utils.wandb_utils import WandbSummaryWriter as RslWandbSummaryWriter
from torch.utils.tensorboard import SummaryWriter

class WandbSummaryWriter(RslWandbSummaryWriter):
    def __init__(self, log_dir: str, flush_secs: int, cfg: dict) -> None:
        SummaryWriter.__init__(self, log_dir, flush_secs)

        # Get the run name
        run_name = os.path.split(log_dir)[-1]
        
        # Thanks to https://github.com/leggedrobotics/rsl_rl/pull/80/
        project = cfg.get('wandb_project') or cfg.get('wandb_kwargs', {}).get('project')
        if project is None:
            raise KeyError(
                "Please specify 'wandb_project' or 'wandb_kwargs.project' in the runner config."
            ) from None

        try:
            entity = cfg['wandb_kwargs']["entity"]
        except KeyError:
            entity = None
            warnings.warn("wandb_entity not specified in the runner config.")
        
        try:
            group = cfg['wandb_kwargs']["group"]
        except KeyError:
            group = None
            warnings.warn("wandb_group not specified in the runner config. Using default group.")

        notes = cfg.get('wandb_kwargs', {}).get('notes', None)

        # Initialize wandb
        wandb.init(
            project=project, 
            entity=entity, 
            name=run_name,
            group=group,
            notes=notes,
        )

        # Add log directory to wandb
        wandb.config.update({"log_dir": log_dir})

        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

        self.video_files = []

        self.update_run_name_with_sequence(
            prefix=cfg["wandb_kwargs"]["project"]
        )


    # To save video files to wandb explicitly
    # Thanks to https://github.com/leggedrobotics/rsl_rl/pull/84    
    def add_video_files(self, log_dir: str, step: int):
        # Check if there are video files in the video directory
        if os.path.exists(log_dir):
            # append the new video files to the existing list
            for root, dirs, files in os.walk(log_dir):
                for video_file in files:
                    if video_file.endswith(".mp4") and video_file not in self.video_files:
                        self.video_files.append(video_file)
                        # add the new video file to wandb only if video file is not updating
                        video_path = os.path.join(root, video_file)

                        # Log video to wandb the fps is not required here since wandb reads
                        # the fps from the video file itself
                        wandb.log(
                            {"Video": wandb.Video(video_path, format="mp4")},
                            step = step
                        )

    # Update the run name with a sequence number. This function is useful to
    # replicate the same behaviour of rsl-rl-lib before v2.3.0
    def update_run_name_with_sequence(self, prefix: str) -> None:
        # Retrieve the current wandb run details (project and entity)
        project = wandb.run.project
        entity = wandb.run.entity

        # Use wandb's API to list all runs in your project
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")

        max_num = 0
        # Iterate through runs to extract the numeric suffix after the prefix.
        for run in runs:
            if run.name.startswith(prefix):
                # Extract the numeric part from the run name.
                numeric_suffix = run.name[
                    len(prefix) :
                ]  # e.g., from "prefix564", get "564"
                try:
                    run_num = int(numeric_suffix)
                    if run_num > max_num:
                        max_num = run_num
                except ValueError:
                    continue

        # Increment to get the new run number
        new_num = max_num + 1
        new_run_name = f"{prefix}{new_num}"

        # Update the wandb run's name
        wandb.run.name = new_run_name
        print("Updated run name to:", wandb.run.name)

