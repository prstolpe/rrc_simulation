from attempt.ddpg import HERDDPG, RemoteHERDDPG
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
    agents = [RemoteHERDDPG.remote(env) for _ in range(8)]
    target_agent = HERDDPG(env)
    for epoch in range(200):
        for cycle in range(50):
            rewards = ray.get([agent.major_gather.remote() for agent in agents])
            for agent in agents:
                # unload buffer
                experience = ray.get(agent.unload.remote())
                for exp in experience:
                    target_agent.replay_buffer.append(exp)
                agent.clear_buffer.remote()
            target_agent.train()
            for agent in agents:
                agent.get_updated_policy.remote(target_agent.policy.get_weights(), target_agent.value.get_weights(),
                                                target_agent.value_target.get_weights(),
                                                target_agent.policy_target.get_weights())

        target_agent.test_env(5)
        print("Epoch: " + str(epoch+1) + " mean reward: " + str(np.mean(np.vstack(rewards[0])[-2000:])) +
              " Success rate: " + str(1 - (np.abs(np.mean(np.vstack(rewards[0])[-2000:]))/50)))
        if len(target_agent.replay_buffer) > int(1e6):
            target_agent.clear_buffer()




    env.close()
    ray.shutdown()

