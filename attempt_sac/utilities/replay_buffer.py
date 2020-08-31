import numpy as np


class PlainReplayBuffer:

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
        idxs = np.random.randint(0, self.size, batch_size)
        temp_dict = dict(s=self.obs1_buf[idxs],
                         s2=self.obs2_buf[idxs],
                         a=self.acts_buf[idxs],
                         r=self.rews_buf[idxs],
                         d=self.done_buf[idxs])
        return (temp_dict['s'], temp_dict['a'], temp_dict['r'].reshape(-1, 1), temp_dict['s2'], temp_dict['d'])
