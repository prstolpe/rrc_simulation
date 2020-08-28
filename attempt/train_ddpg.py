import gym

from attempt.ddpg import DDPG

import os
from tqdm import tqdm
from collections import deque
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    env = gym.make("Pendulum-v0")
    agent = DDPG(env)
    agent.drill()