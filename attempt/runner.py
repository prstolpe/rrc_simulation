
from attempt.ddpg import HERDDPG, DDPG
import gym
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    env = gym.make('FetchReach-v1')
    agent = HERDDPG(env)
    for epoch in range(5):
        for cycle in tqdm(range(10)):
            agent.gather_cycle()

        # target_agent.train()

        agent.test_env(10)
    env.close()

    plt.plot(np.vstack(agent.rewards))
    plt.title('Rewards')
    plt.show()

    plt.plot(np.vstack(agent.policy_losses))
    plt.title('Policy Losses')
    plt.show()

    plt.plot(np.vstack(agent.value_losses))
    plt.title('Value Losses')
    plt.show()