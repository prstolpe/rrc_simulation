#!/usr/bin/env python3
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import HER
from stable_baselines import SAC
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common import set_global_seeds
from stable_baselines.common.callbacks import CheckpointCallback

from example_pushing_training_env import ExamplePushingTrainingEnv
from example_pushing_training_env import FlatObservationWrapper


import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
import numpy as np


def get_multi_process_env(num_of_envs):
    def _make_env(rank):
        def _init():
            env = ExamplePushingTrainingEnv(frameskip=3, visualization=False)
            env.seed(seed=rank)
            env.action_space.seed(seed=rank)
            env = FlatObservationWrapper(env)
            return env

        return _init

    return SubprocVecEnv([_make_env(rank=i) for i in range(num_of_envs)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", required=True, help="output path")
    args = vars(parser.parse_args())
    output_path = str(args["output_path"])

    total_time_steps = 80000000
    validate_every_timesteps = 2000000
    model_path = os.path.join(output_path, "training_checkpoints")

    os.makedirs(model_path, exist_ok=True)

    set_global_seeds(0)
    num_of_active_envs = 1
    policy_kwargs = dict(layer=[256, 256])
    #env = gym.make("real_robot_challenge_phase_1-v1")
    env = FlatObservationWrapper(ExamplePushingTrainingEnv(frameskip=20, visualization=False))

    train_configs = {
        "gamma": 0.99,
        "n_steps": int(120000 / 20),
        "ent_coef": 0.01,
        "learning_rate": 0.00025,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "nminibatches": 40,
        "noptepochs": 4,
    }

    model = HER(
        MlpPolicy,
        env,
        SAC,
        verbose=1,
        tensorboard_log=model_path
    )

    ckpt_frequency = int(validate_every_timesteps / num_of_active_envs)
    checkpoint_callback = CheckpointCallback(
        save_freq=ckpt_frequency, save_path=model_path, name_prefix="model"
    )

    model.learn(int(total_time_steps), callback=checkpoint_callback)
    env.close()
