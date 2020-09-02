from stable_baselines import HER
from stable_baselines.sac import SAC
from attempt_sac.environments import *
from rrc_simulation.gym_wrapper.envs.cube_env import RandomInitializer
from rrc_simulation.gym_wrapper.envs import cube_env

import os

if __name__ == "__main__":

    env = gym.make('Example_CubeEnv-v0',
                   initializer=RandomInitializer(difficulty=1),
                   action_type=cube_env.ActionType.POSITION,
                   visualization=False,
                   )
    logdir = os.path.join(os.getcwd(), 'training_logs')
    os.makedirs(logdir, exist_ok=True)

    model_kwargs = {'ent_coef': 'auto',
                    'buffer_size': int(1e6),
                    'gamma': 0.95,
                    'learning_starts': 1000,
                    'train_freq': 1}

    model = HER('MlpPolicy', env, SAC,
                verbose=True, **model_kwargs)

    model.learn(int(5e6))
    model.save("./hersac_CubeEnv_diffone")
