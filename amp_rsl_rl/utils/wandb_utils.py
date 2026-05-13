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
