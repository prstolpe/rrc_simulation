from attempt.ddpg import HERDDPG, RemoteHERDDPG
from attempt.utilities.wrappers import StateNormalizationWrapper
import gym
import numpy as np
import matplotlib.pyplot as plt
import ray
import os
from tqdm import tqdm
from collections import deque
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    env = gym.make('FetchReach-v1')
    agent = HERDDPG(env)
    agent.major_gather()
