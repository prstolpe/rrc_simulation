from attempt.ddpg import HERDDPG, RemoteHERDDPG
import gym

import ray
import os

from tqdm import tqdm


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":

    env = gym.make('FetchReach-v1')

    target_agent = HERDDPG(env)

    for epoch in range(5):
        for cycle in tqdm(range(10)):
            target_agent.gather_cycle()

           # target_agent.train()

        target_agent.test_env(10)
    env.close()





