from attempt.ddpg import HERDDPG
import gym
import numpy as np
import matplotlib.pyplot as plt
import ray
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    env = gym.make('FetchPickAndPlace-v1')
    agent = HERDDPG(env)
    agent.drill()

    plt.plot(np.vstack(agent.rewards))
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.show()

