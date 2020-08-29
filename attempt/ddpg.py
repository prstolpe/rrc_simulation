import numpy as np
import tensorflow as tf
import gym
import ray
from collections import deque
import random

from attempt.utilities.utils import env_extract_dims, flatten_goal_observation
from attempt.models.models import Critic_gen, Actor_gen
from attempt.utilities.her import HER


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
        idxs = np.random.randint(0, self.size, batch_size)
        temp_dict = dict(s=self.obs1_buf[idxs],
                         s2=self.obs2_buf[idxs],
                         a=self.acts_buf[idxs],
                         r=self.rews_buf[idxs],
                         d=self.done_buf[idxs])
        return (temp_dict['s'], temp_dict['a'], temp_dict['r'].reshape(-1, 1), temp_dict['s2'], temp_dict['d'])


class DDPG:

    def __init__(self, env: gym.Env, buffer_size: int=int(1e6), gamma: int = 0.99, tau: int = 1e-2, start_steps: int=100,
                 noise_scale: float=0.1, batch_size=32, actor_lr=1e-3, value_lr=1e-3, seed: int=5):

        # env
        self.obs_dim, self.act_dim = env_extract_dims(env)
        self.env = env
        self.act_low, self.act_high = self.env.action_space.low, self.env.action_space.high

        # replay buffer
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, buffer_size)

        # networks
        self.policy = Actor_gen(self.obs_dim, self.act_dim, hidden_layers=(512, 200, 128), action_mult=self.act_high)
        self.value = Critic_gen(self.obs_dim, self.act_dim, hidden_layers=(1024, 512, 300, 1))

        self.policy_target = Actor_gen(self.obs_dim, self.act_dim, hidden_layers=(512, 200, 128))
        self.value_target = Critic_gen(self.obs_dim, self.act_dim, hidden_layers=(1024, 512, 300, 1))
        self.policy_target.set_weights(self.policy.get_weights())
        self.value_target.set_weights(self.value.get_weights())

        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=value_lr)

        # ddpg hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.noise_scale = noise_scale

        self.seed = np.random.seed(seed)
        self.start_steps = start_steps
        self.batch_size = batch_size

        # monitoring
        self.rewards = []
        self.value_losses = []
        self.policy_losses = []

    def get_action(self, s):

        a = self.policy(s.reshape(1, -1))[0]
        a += self.noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, self.act_low, self.act_high)

    def _policy_loss(self, states):
        next_policy_actions = self.policy(states)
        return - tf.reduce_mean(self.value([states, next_policy_actions]))

    def _value_loss(self, states, actions, next_states, rewards, done):

        Qvals = self.value([states, actions])
        next_actions = self.policy_target(next_states)
        next_Q = self.value_target([next_states, next_actions])
        next_Q = tf.clip_by_value(next_Q, -(1/(1 - self.gamma)), 0)
        Qprime = rewards + self.gamma * (1 - done) * next_Q

        return tf.reduce_mean(tf.square(Qvals - Qprime))

    def _learn_on_batch(self, batch):
        states, actions, rewards, next_states, done = batch

        # value optimization
        with tf.GradientTape() as tape:
            value_loss = self._value_loss(states, actions, next_states, rewards, done)
            self.value_losses.append(value_loss)
            value_gradients = tape.gradient(value_loss, self.value.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_gradients, self.value.trainable_variables))

        # policy optimization
        with tf.GradientTape() as tape2:
            policy_loss = self._policy_loss(states)
            self.policy_losses.append(policy_loss)
            policy_gradients = tape2.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy.trainable_variables))

        # updating target network
        temp1 = np.array(self.value_target.get_weights())
        temp2 = np.array(self.value.get_weights())
        temp3 = self.tau * temp2 + (1 - self.tau) * temp1
        self.value_target.set_weights(temp3)

        # updating Actor network
        temp1 = np.array(self.policy_target.get_weights())
        temp2 = np.array(self.policy.get_weights())
        temp3 = self.tau * temp2 + (1 - self.tau) * temp1
        self.policy_target.set_weights(temp3)

    def drill(self, num_episodes=25):
        num_steps = 0
        for episode in range(num_episodes):

            is_done = False
            observation, episode_reward = self.env.reset(), 0

            while not is_done:
                num_steps += 1
                if num_steps > self.start_steps:
                    action = self.get_action(observation)
                else:
                    action = self.env.action_space.sample()

                next_observation, reward, is_done, info = self.env.step(action)
                episode_reward += reward
                # update buffer
                self.replay_buffer.store(observation, action, reward, next_observation, is_done)
                observation = next_observation

                #n_samples = len(self.replay_buffer)
                #K = np.min([n_samples, self.batch_size])
                batch = self.replay_buffer.sample_batch(self.batch_size)
                self._learn_on_batch(batch)


                if is_done:
                    self.rewards.append(episode_reward)
                    print("Episode " + str(episode) + ": " + str(episode_reward))

        self.test_env(5)

    def test_env(self, num_episodes=1):

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


class HERDDPG(DDPG):

    def __init__(self, env: gym.GoalEnv, buffer_size: int=int(1e5), gamma: int = 0.99, tau: int = 1e-2, start_steps: int=1000,
                 noise_scale: float=0.1, batch_size=int(128), actor_lr=1e-3, value_lr=1e-3, seed: int=5):

        super().__init__(env, buffer_size, gamma, tau, start_steps, noise_scale,
                         batch_size, actor_lr, value_lr, seed)


        self.obs_names = ['observation', 'desired_goal']
        self.her_buffer = HER()

        self.ep_len = self.env.spec.max_episode_steps

    def drill(self, num_episodes=30):
        num_steps = 0
        self.actions = []
        for epoch in range(10):
            for episode in range(num_episodes):

                observation, episode_reward = self.env.reset(), 0
                ac_obs = observation['achieved_goal']
                observation = flatten_goal_observation(observation, self.obs_names)
                for _ in range(self.ep_len):
                    num_steps += 1
                    if num_steps > self.start_steps:
                        action = self.get_action(observation)
                    else:
                        action = self.env.action_space.sample()

                    next_observation, reward, is_done, info = self.env.step(action)
                    self.actions.append(action)
                    next_ac_obs = next_observation['achieved_goal']
                    next_observation = flatten_goal_observation(next_observation, self.obs_names)
                    episode_reward += reward
                    # update buffer
                    self.replay_buffer.store(observation, action, reward, next_observation, is_done)
                    self.her_buffer.keep([np.concatenate((observation, ac_obs), axis=-1)
                                             , action, reward,
                                          np.concatenate((next_observation, next_ac_obs), axis=-1), is_done])
                    observation = next_observation
                    ac_obs = next_ac_obs

                    batch = self.replay_buffer.sample_batch(self.batch_size)
                    self._learn_on_batch(batch)
                self.rewards.append(episode_reward)
                print("Epoch " + str(epoch) + "Episode " + str(episode) + ": " + str(episode_reward))

                her_list = self.her_buffer.backward()
                for observation, action, reward, next_observation, is_done in her_list:
                    self.replay_buffer.store(observation[0:13], action, reward, next_observation[0:13], is_done)
                self.her_buffer.reset()

            self.test_env(3)

    def test_env(self, num_episodes=1):

        n_steps = 0
        for j in range(num_episodes):
            s, episode_return, episode_length, d = self.env.reset(), 0, 0, False
            s = flatten_goal_observation(s, self.obs_names)
            while not d:
                # Take deterministic actions at test time (noise_scale=0)
                s, r, d, _ = self.env.step(self.policy_target(s.reshape(1, -1))[0])
                s = flatten_goal_observation(s, self.obs_names)
                episode_return += r
                episode_length += 1
                n_steps += 1
            print('test return:', episode_return, 'episode_length:', episode_length)