from attempt.ddpg import HERDDPG
import gym
import numpy as np
import matplotlib.pyplot as plt
import ray
import os
from collections import deque
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    env = gym.make('FetchPickAndPlace-v1')
    ray.init(num_cpus=8)
    agents = [HERDDPG.remote(env) for _ in range(8)]
    target_agent = HERDDPG(env)
    for i in range(30):
        rewards = ray.get([agent.drill.remote() for agent in agents])
        for agent in agents:
            # unload buffer
            target_agent.replay_buffer.append(agent.replay_buffer)
            agent.replay_buffer = deque()
        target_agent.train()
        target_agent.test_env(10)
        print("Epoch: " + str(i) + " mean reward: " + str(np.mean(np.vstack(rewards)[-2000:])))

    plt.plot(np.vstack(rewards))
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.show()
    env.close()
    ray.shutdown()

