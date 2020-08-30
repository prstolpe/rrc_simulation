from attempt.ddpg import HERDDPG, RemoteHERDDPG
from attempt.utilities.wrappers import StateNormalizationWrapper
import gym

import ray
import os

from tqdm import tqdm
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":


    env = gym.make('FetchReach-v1')
    ray.init(num_cpus=8)
    agents = [RemoteHERDDPG.remote(env) for _ in range(8)]
    target_agent = HERDDPG(env)

    for agent in agents:
        agent.get_updated_policy.remote(target_agent.policy.get_weights(), target_agent.value.get_weights(),
                                        target_agent.policy_target.get_weights(),
                                        target_agent.value_target.get_weights())
    for epoch in range(80):
        for cycle in tqdm(range(50)):
            experience = ray.get([agent.unload.remote() for agent in agents])

            for s, a, r, sn, d in experience:
                for i in range(len(s)):
                    target_agent.replay_buffer.replay_buffer.add(s[i], a[i], r[i], sn[i], d[i])
            target_agent.train()
            for agent in agents:
                agent.get_updated_policy.remote(target_agent.policy.get_weights(), target_agent.value.get_weights(),
                                                target_agent.policy_target.get_weights(), target_agent.value_target.get_weights())
        target_agent.test_env(10)
    env.close()
    ray.shutdown()




