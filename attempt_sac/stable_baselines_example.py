from stable_baselines import HER
from stable_baselines.sac import SAC
from attempt_sac.environments import *
from rrc_simulation.gym_wrapper.envs.cube_env import RandomInitializer
from rrc_simulation.gym_wrapper.envs import cube_env
from rrc_simulation.tasks import move_cube

import os
import numpy as np


class Initializer(RandomInitializer):

    """Initializer that samples random initial states and goals."""

    def __init__(self, difficulty):
        """Initialize.

        Args:
            difficulty (ndarray):  Difficulty level for sampling goals.
        """
        self.difficulties = difficulty
        self.difficulty = None

    def get_initial_state(self):
        """Get a random initial object pose (always on the ground)."""
        return move_cube.sample_goal(difficulty=-1)

    def get_goal(self):
        """Get a random goal depending on the difficulty."""
        self.difficulty = np.random.choice(self.difficulties)
        return move_cube.sample_goal(difficulty=self.difficulty)


if __name__ == "__main__":

    difficulties = np.arange(1, 5)
    env = gym.make('Example_CubeEnv-v0',
                   initializer=Initializer(difficulty=difficulties),
                   action_type=cube_env.ActionType.POSITION,
                   visualization=False,
                   )

    model_kwargs = {'ent_coef': 'auto',
                    'buffer_size': int(1e6),
                    'gamma': 0.95,
                    'learning_starts': 1000,
                    'train_freq': 1}

    model = HER('MlpPolicy', env, SAC,
                verbose=True, **model_kwargs)

    model.learn(int(8e6))
    model.save("./hersac_CubeEnv_diffall")

    os.system("shutdown now")
