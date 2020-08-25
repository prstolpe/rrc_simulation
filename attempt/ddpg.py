import numpy as np
import tensorflow as tf
import gym

from attempt.utilities.utils import env_extract_dims
from attempt.models.models import build_ffn_models

class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, buffer_size):
        self.obs1_buf = np.zeros([buffer_size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([buffer_size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([buffer_size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(buffer_size, dtype=np.float32)
        self.done_buf = np.zeros(buffer_size, dtype=np.float32)
        self.itr, self.size, self.max_size = 0, 0, buffer_size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.itr] = obs
        self.obs2_buf[self.itr] = next_obs
        self.acts_buf[self.itr] = act
        self.rews_buf[self.itr] = rew
        self.done_buf[self.itr] = done
        self.itr = (self.itr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class DDPG:

    def __init__(self, env:gym.Env, buffer_size:int=50000, seed:int=5, update_every=5000,
                 batch_size=1000, gamma:int=0.99):

        # env
        self.obs_dim, self.act_dim = env_extract_dims(env)
        self.env = env
        self.act_low, self.act_high = self.env.action_space.low, self.env.action_space.high


        # replay buffer
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, buffer_size)

        # networks
        self.policy, self.value, self.policy_target, self.value_target = build_ffn_models(env)

        # ddpg
        self.gamma = gamma


        self.seed = np.random.seed(seed)
        self.step_counter = 0
        self.update_every = update_every
        self.batch_size = batch_size

    def explore(self, observation):
        return np.clip((self.policy(observation) + np.random.randn(self.act_dim)), self.act_low, self.act_high)

    def _learn_on_batch(self, batch):
        states, next_states, actions, rewards, done = batch

        #with tf.GradientTape() as tape:
        Qvals = self.value([states, actions])
        next_actions = self.policy_target(next_states)
        next_Q = self.value_target([next_states, next_actions])

        Qprime = rewards + self.gamma * next_Q

        value_loss = tf.keras.losses.MSE(Qvals, Qprime)
        print(value_loss)


    def drill(self):


        is_done = False
        observation = self.env.reset()
        while not is_done:
            self.step_counter += 1
            action = self.explore(observation)
            next_observation, reward, is_done, info = self.env.step(action)

            # update buffer
            self.replay_buffer.store(observation, action, reward, next_observation, is_done)
            observation = next_observation

            if is_done:
                observation = self.env.reset()
                is_done = False

            if self.step_counter == self.update_every:
                for i in range(self.update_every):

                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self._learn_on_batch(batch)


