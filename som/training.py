import time
import pathlib
import os
import regex as re
import math
import functools

import torch

from src.som.model import ModelSaverLoader


class AlphaSchedule:
    """
    Controls the alpha parameter (i.e. the gain or the learning rate).
    Holds a constant value until reaching the kneel point step, then decreases linearly to zero.
    """

    def __init__(self, num_steps, kneel_pt_step=1000, initial_alpha=0.9, dtype=torch.float32, device=None):
        """
        :param num_steps: Total number of training steps.
        :param kneel_pt_step: Step number at which alpha will begin to decay.  Default value of 1000 is taken from rule of thumb from paper:
                              "Kohonen, Teuvo. "The self-organizing map." Proceedings of the IEEE 78.9 (1990): 1464-1480."
        :param initial_alpha: The starting value for alpha.
        """

        self._initial_alpha = initial_alpha
        self._alpha = torch.tensor([initial_alpha], dtype=dtype, device=device)
        self._current_step = 0
        self._final_step = num_steps
        self._kneel_pt_step = kneel_pt_step

    def step(self):
        self._current_step += 1
        self._current_step = min(self._current_step, self._final_step)

        self._alpha[0] = self._initial_alpha * (1.0 - max(0, self._current_step - self._kneel_pt_step) / (self._final_step - self._kneel_pt_step))

    def set_step(self, step_num):
        if 0 < step_num:
            self._current_step = min(step_num - 1, self._final_step)
            self.step()

    @property
    def alpha(self):
        return self._alpha


class SigmaSchedule:
    """
    Controls the sigma parameter (i.e. the neighbourhood radius in map space).
    Holds a constant value until reaching the kneel point step, then decreases linearly to zero.
    Recommended you set initial sigma value to half your map's largest dimension size.
    """

    def __init__(self, initial_sigma, num_steps, kneel_pt_step=1000, dtype=torch.float32, device=None):
        """
        :param initial_sigma: The starting value for sigma.
        :param num_steps: Total number of training steps.
        :param kneel_pt_step: Step number at which sigma will begin to decay.  Default value of 1000 is taken from rule of thumb from paper:
                              "Kohonen, Teuvo. "The self-organizing map." Proceedings of the IEEE 78.9 (1990): 1464-1480."
        """

        self._initial_sigma = initial_sigma
        self._sigma = torch.tensor([initial_sigma], dtype=dtype, device=device)
        self._sigma_squared = torch.pow(self._sigma, 2.0)
        self._current_step = 0
        self._final_step = num_steps
        self._kneel_pt_step = kneel_pt_step

    def step(self):
        self._current_step += 1
        self._current_step = min(self._current_step, self._final_step)

        self._sigma[0] = 1.0 + (self._initial_sigma - 1.0) * (1.0 - max(0, self._current_step - self._kneel_pt_step) / (self._final_step - self._kneel_pt_step))
        self._sigma_squared = torch.pow(self._sigma, 2.0)

    def set_step(self, step_num):
        if 0 < step_num:
            self._current_step = min(step_num - 1, self._final_step)
            self.step()

    @property
    def sigma(self):
        return self._sigma

    @property
    def sigma_squared(self):
        return self._sigma_squared


class ModelTraining:
    """
    Performs the model training.
    Contains logic for saving checkpoints during training and restoring from such checkpoints in the event of a crash.
    """

    CHECKPOINTS_FOLDER_NAME = "checkpoints"
    SAVED_MODEL_FOLDER_NAME = "saved-model"
    CHECKPOINT_FILENAME_STRING = "checkpoint-{:020d}"
    CHECKPOINT_FILENAME_REGEX_PATTERN = re.compile(r"(^checkpoint-)(\d{20}$)")

    @classmethod
    def run(cls, data, num_steps, model, path_output_dir, save_interval_min=15, print_period=1000, dtype=torch.float32, device=None):
        """
        Performs model training.  Saves checkpoint files every save_interval_min minutes.
        Will resume training from latest checkpoint (if one exists).   Therefore, if training is interrupted somehow,
        we can resume it simply by calling this method again with the original arguments.

        data is assumed to be a tensor of your training data already loaded into the device your are working with.  If you are working on GPU (cuda)
        then this assumes your training data set fits into GPU memory.
        """
        model.train(True)

        path_output_dir = pathlib.Path(path_output_dir)
        path_checkpoints_dir = path_output_dir.joinpath(cls.CHECKPOINTS_FOLDER_NAME)
        path_model_save_dir = path_output_dir.joinpath(cls.SAVED_MODEL_FOLDER_NAME)
        if not path_checkpoints_dir.exists():
            path_checkpoints_dir.mkdir(parents=True)

        alpha_schedule = AlphaSchedule(num_steps, dtype=dtype, device=device)
        sigma_schedule = SigmaSchedule(cls.initial_sigma(model.shape), num_steps, dtype=dtype, device=device)

        # Resume from checkpoint if one exists
        starting_step, state_dict = cls._load_latest_checkpoint(path_checkpoints_dir)
        if state_dict:
            model.load_state_dict(state_dict)
            alpha_schedule.set_step(starting_step)
            sigma_schedule.set_step(starting_step)
            data_sampler = torch.utils.data.RandomSampler(data, replacement=True, num_samples=max(0, num_steps - starting_step))
        else:
            data_sampler = torch.utils.data.RandomSampler(data, replacement=True, num_samples=num_steps)

        save_interval_sec = save_interval_min * 60
        time_last_save = time.time()

        # Training loop
        step_num = starting_step
        for sample_index in data_sampler:
            if step_num % print_period == 0:
                print("Step {} of {}".format(step_num, num_steps))

            # Save a checkpoint every save_interval_min minutes
            if save_interval_sec < time.time() - time_last_save:
                cls._save_checkpoint(model, step_num, path_checkpoints_dir)
                time_last_save = time.time()
                print("Checkpoint saved")

            # One step of training
            alpha_schedule.step()
            sigma_schedule.step()
            model(data[sample_index], alpha_schedule.alpha, sigma_schedule.sigma_squared)
            step_num += 1

        # Save the model when training is finished
        model.train(False)
        ModelSaverLoader.save(model, path_model_save_dir)
        print("Model saved to {}".format(path_model_save_dir))

    @staticmethod
    def recommended_num_steps(map_shape):
        """
        The recommended number of steps is equal to the number of map nodes times 500.
        This is a rule of thumb taken from the paper: Kohonen, Teuvo. "The self-organizing map." Proceedings of the IEEE 78.9 (1990): 1464-1480.
        """
        return 500 * functools.reduce(lambda x, y: x * y, map_shape)

    @staticmethod
    def initial_sigma(map_shape):
        """
        We take initial sigma as just over half the length of the map diagonal.
        """
        return math.ceil(functools.reduce(lambda x, y: x + y ** 2, map_shape) ** 0.5)

    @classmethod
    def _save_checkpoint(cls, model, step_num, path_checkpoint_dir):
        file_path = path_checkpoint_dir.joinpath(cls.CHECKPOINT_FILENAME_STRING.format(step_num))
        torch.save(model.state_dict(), file_path)

    @classmethod
    def _load_latest_checkpoint(cls, path_checkpoints_dir):
        checkpoint_file_names = [
            item_name for item_name in os.listdir(path_checkpoints_dir)
            if cls.CHECKPOINT_FILENAME_REGEX_PATTERN.match(item_name)]
        if checkpoint_file_names:
            latest_checkpoint_file_name = max(checkpoint_file_names)
            step_num = int(cls.CHECKPOINT_FILENAME_REGEX_PATTERN.match(latest_checkpoint_file_name).group(2))
            state_dict = torch.load(path_checkpoints_dir.joinpath(latest_checkpoint_file_name))
            return step_num, state_dict
        return 0, None
