
from attempt.ddpg import HERDDPG, DDPG
import gym
import os
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    env = gym.make('Pendulum-v0')
    agent = DDPG(env)
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