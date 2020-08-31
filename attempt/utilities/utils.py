#!/usr/bin/env python
"""Helper functions."""

from typing import Tuple, Union
from collections import OrderedDict

import gym
import numpy as np

from gym.spaces import Discrete, Box, Dict


def env_extract_dims(env: gym.Env) -> Tuple[Union[int, Tuple[int]], int]:
    """Returns state and action space dimensionality for given environment."""

    # observation space
    if isinstance(env.observation_space, Dict):
        # dict observation with observation field
        if isinstance(env.observation_space["observation"], gym.spaces.Box):
            if isinstance(env.observation_space.spaces, OrderedDict):
                obs_dim = env.observation_space["observation"].shape[0] + env.observation_space['desired_goal'].shape[0] + \
                    env.observation_space['achieved_goal'].shape[0]
            else:
                raise ValueError(f"Cannot extract the dimensionality from a Dict observation space "
                                                      f"where the observation is of type "
                                                      f"{type(env.observation_space['observation']).__name__}")
    else:
        # standard observation in box form
        obs_dim = env.observation_space.shape[0]

    # action space
    if isinstance(env.action_space, Discrete):
        act_dim = env.action_space.n
    elif isinstance(env.action_space, Box):
        act_dim = env.action_space.shape[0]
    else:
        raise NotImplementedError(f"Environment has unknown Action Space Typ: {env.action_space}")

    return obs_dim, act_dim


def flatten_goal_observation(obs, obs_names):
    observation = [obs[name].flatten() for name in obs_names]

    return np.concatenate(observation)
