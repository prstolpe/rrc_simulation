
from attempt.ddpg import HERDDPG, DDPG
import gym
import os
import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    env = gym.make('FetchReach-v1')
    agent = HERDDPG(env)
    agent.drill()

    plt.plot(np.vstack(agent.rewards))
    plt.title('Rewards')
    plt.show()

    plt.plot(np.vstack(agent.policy_losses))
    plt.title('Policy Losses')
    plt.show()

    plt.plot(np.vstack(agent.value_losses))
    plt.title('Value Losses')
    plt.show()