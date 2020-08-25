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

    def __init__(self, env:gym.Env, buffer_size:int=500000, seed:int=5, update_every=1000,
                 batch_size=64, gamma:int=0.99, tau:int=1e-2):

        # env
        self.obs_dim, self.act_dim = env_extract_dims(env)
        self.env = env
        self.act_low, self.act_high = self.env.action_space.low, self.env.action_space.high


        # replay buffer
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, buffer_size)

        # networks
        self.policy, self.value, self.policy_target, self.value_target = build_ffn_models(env)
        self.policy_optimizer = tf.keras.optimizers.Adam(lr=1e-4)
        self.value_optimizer = tf.keras.optimizers.Adam(lr=1e-3)

        # ddpg
        self.gamma = gamma
        self.tau = tau
        self.g = 1

        self.seed = np.random.seed(seed)
        self.step_counter = 0
        self.update_every = update_every
        self.batch_size = batch_size
        self.rewards = []

    def explore(self, s):

        a = self.policy.predict(s.reshape(1, -1))[0]
        a += 0.1 * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_low, self.act_high)

    def _learn_on_batch(self, batch):
        states, next_states, actions, rewards, done = batch['obs1'], batch['obs2'], batch['acts'], batch['rews'], batch['done']
        states_actions = tf.concat((states, actions), axis=-1)
        # policy optimization
        with tf.GradientTape() as tape2:
            next_policy_actions = self.policy(tf.convert_to_tensor(states, dtype=tf.float32))
            states_next_policy_actions = tf.concat((states, next_policy_actions), axis=-1)
            policy_loss = - tf.reduce_mean(self.value(states_next_policy_actions))
            policy_gradients = tape2.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy.trainable_variables))
        # value optimization
        with tf.GradientTape() as tape:
            Qvals = self.value(tf.convert_to_tensor(states_actions, dtype=tf.float32))

            next_actions = self.policy_target(tf.convert_to_tensor(next_states, dtype=tf.float32))

            next_states_next_actions = tf.concat((next_states, next_actions), axis=-1)
            next_Q = self.value_target(next_states_next_actions)

            Qprime = rewards + self.gamma * (1-done) * next_Q

            value_loss = tf.reduce_mean((Qvals - Qprime)**2)
            value_gradients = tape.gradient(value_loss, self.value.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_gradients, self.value.trainable_variables))

        temp1 = np.array(self.value_target.get_weights())
        temp2 = np.array(self.value.get_weights())
        temp3 = self.tau * temp1 + (1 - self.tau) * temp2
        self.value_target.set_weights(temp3)

        # updating Actor network
        temp1 = np.array(self.policy_target.get_weights())
        temp2 = np.array(self.policy.get_weights())
        temp3 = self.tau * temp1 + (1 - self.tau) * temp2
        self.policy_target.set_weights(temp3)
    def drill(self):


        is_done = False
        observation = self.env.reset()
        for i in range(self.buffer_size):
            self.step_counter += 1
            action = self.explore(observation)

            next_observation, reward, is_done, info = self.env.step(action)
            self.rewards.append(reward)
            # update buffer
            self.replay_buffer.store(observation, action, reward, next_observation, is_done)
            observation = next_observation

            if is_done:
                observation = self.env.reset()
                is_done = False

            if self.step_counter % self.update_every == 0:
                for k in range(100):

                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self._learn_on_batch(batch)

                self.test_env(3)

    def test_env(self, num_episodes=1):
        #t0 = datetime.now()
        n_steps = 0
        for j in range(num_episodes):
            s, episode_return, episode_length, d = self.env.reset(), 0, 0, False
            while not d:
                # Take deterministic actions at test time (noise_scale=0)
                s, r, d, _ = self.env.step(self.policy(tf.convert_to_tensor([s]))[0])
                episode_return += r
                episode_length += 1
                n_steps += 1
            print('test return:', episode_return, 'episode_length:', episode_length)



