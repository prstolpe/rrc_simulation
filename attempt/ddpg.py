import numpy as np
import tensorflow as tf
import gym

from collections import deque
from attempt.utilities.utils import env_extract_dims, flatten_goal_observation
from attempt.models.models import Critic_gen, Actor_gen
from attempt.utilities.her import HER
from attempt.utilities.wrappers import StateNormalizationWrapper
import random

import ray

<<<<<<< HEAD
=======
from attempt.utilities.utils import env_extract_dims
from attempt.models.models import Critic_gen, Actor_gen
>>>>>>> 16a1ffa... DDPG working fine

#############This noise code is copied from openai baseline #########OrnsteinUhlenbeckActionNoise############# Openai Code#########

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

<<<<<<< HEAD
    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
=======
    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        temp_dict = dict(s=self.obs1_buf[idxs],
                         s2=self.obs2_buf[idxs],
                         a=self.acts_buf[idxs],
                         r=self.rews_buf[idxs],
                         d=self.done_buf[idxs])
        return (temp_dict['s'], temp_dict['a'], temp_dict['r'].reshape(-1, 1), temp_dict['s2'], temp_dict['d'])
>>>>>>> 16a1ffa... DDPG working fine


#########################################################################################################

<<<<<<< HEAD

class DDPG(object):

    def __init__(self, env:gym.Env, buffer_size:int=int(1e6), seed:int=5, num_episodes:int=25,
                 batch_size=32, gamma:int=0.99, tau:int=1e-2, start_steps:int=1000, actor_lr=1e-4,
=======
    def __init__(self, env:gym.Env, buffer_size:int=int(1e5), seed:int=5, num_episodes:int=30,
                 batch_size=16, gamma:int=0.99, tau:int=1e-2, start_steps:int=1000, actor_lr=1e-3,
>>>>>>> 16a1ffa... DDPG working fine
                 value_lr=1e-3):

        # env
        self.obs_dim, self.act_dim = env_extract_dims(env)
        self.env = env
        self.act_low, self.act_high = self.env.action_space.low, self.env.action_space.high
<<<<<<< HEAD
        self.ep_len = env.spec.max_episode_steps
=======
>>>>>>> 16a1ffa... DDPG working fine

        # replay buffer
        self.buffer_size = buffer_size
        self.replay_buffer = deque(maxlen=buffer_size)

        # networks
        self.policy = Actor_gen(self.obs_dim, self.act_dim, hidden_layers=(512, 200, 128), action_mult=self.act_high)
        self.value = Critic_gen(self.obs_dim, 1, hidden_layers=(1024, 512, 300, 1))

<<<<<<< HEAD
        self.policy_target = Actor_gen(self.obs_dim, self.act_dim, hidden_layers=(512, 200, 128), action_mult=self.act_high)
=======
        self.policy_target = Actor_gen(self.obs_dim, self.act_dim, hidden_layers=(512, 200, 128))
>>>>>>> 16a1ffa... DDPG working fine
        self.value_target = Critic_gen(self.obs_dim, 1, hidden_layers=(1024, 512, 300, 1))
        self.policy_target.set_weights(self.policy.get_weights())
        self.value_target.set_weights(self.value.get_weights())

        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=value_lr)

        # ddpg hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.num_episodes = num_episodes

        self.num_episodes = num_episodes
        np.random.seed(seed)
        self.step_counter = 0
        self.start_steps = start_steps
        self.batch_size = batch_size

        # monitoring
        self.rewards = []
        self.value_losses = []
        self.policy_losses = []

        self.action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.act_dim))

    def get_action(self, s):

        a = self.policy(s.reshape(1, -1).astype(np.float32))[0]
        a += 0.1 * np.random.randn(self.act_dim)
        return np.clip(a, self.act_low, self.act_high)

    def _policy_loss(self, states):
        next_policy_actions = self.act_high * self.policy(states)
        return - tf.reduce_mean(self.value([states, next_policy_actions]))

    def _value_loss(self, states, actions, next_states, rewards, done):
        Qvals = self.value([states, actions])
        next_actions = self.policy_target(next_states)
        next_Q = self.value_target([next_states, next_actions])
        Qprime = rewards + self.gamma * (1 - done) * next_Q

        return tf.reduce_mean(tf.square(Qvals - Qprime))

    def _learn_on_batch(self, batch):
<<<<<<< HEAD
        states, actions, rewards, next_states, done = zip(*batch)

=======
        states, actions, rewards, next_states, done = batch
>>>>>>> 16a1ffa... DDPG working fine
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
<<<<<<< HEAD
        done = np.asarray(done, dtype=np.float32)
=======
>>>>>>> 16a1ffa... DDPG working fine
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

    def drill(self):
<<<<<<< HEAD

        for episode in range(self.num_episodes):

            is_done = False
            observation, episode_reward = self.env.reset(), 0

            for _ in range(200):
                self.step_counter += 1
                if self.step_counter > self.start_steps:
                    action = self.get_action(observation)
                else:
                    action = self.env.action_space.sample()

                next_observation, reward, is_done, info = self.env.step(action)
                episode_reward += reward
                # update buffer
                self.replay_buffer.append([observation, action, reward, next_observation, is_done])
                observation = next_observation
                if len(self.replay_buffer) > self.batch_size:

                    batch = random.sample(self.replay_buffer, self.batch_size)
=======

        for episode in range(self.num_episodes):

            is_done = False
            observation, episode_reward = self.env.reset(), 0

            while not is_done:
                self.step_counter += 1
                if self.step_counter > self.start_steps:
                    action = self.get_action(observation)
                else:
                    action = self.env.action_space.sample()

                next_observation, reward, is_done, info = self.env.step(action)
                episode_reward += reward
                # update buffer
                self.replay_buffer.store(observation, action, reward, next_observation, is_done)
                observation = next_observation
                if self.replay_buffer.size > self.batch_size:
                    batch = self.replay_buffer.sample_batch(self.batch_size)
>>>>>>> 16a1ffa... DDPG working fine
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

                s, r, d, i = self.env.step(self.policy(s.reshape(1, -1).astype(np.float32))[0])

                episode_return += r
                episode_length += 1
                n_steps += 1
            print('test return:', episode_return, 'episode_length:', episode_length)


class HERDDPG(DDPG):

    def __init__(self, env:gym.GoalEnv, buffer_size:int=int(1e6), seed:int=5, num_episodes:int=800,
                 batch_size=128, gamma:int=0.98, tau:int=1e-2, start_steps:int=500, actor_lr=1e-3,
                 value_lr=1e-3, epochs:int=100):

        super().__init__(env=env, buffer_size=buffer_size, seed=seed, num_episodes=num_episodes,
                         batch_size=batch_size, gamma=gamma, tau=tau, start_steps=start_steps,
                         actor_lr=actor_lr, value_lr=value_lr)
        self.observation_names = ["observation", "achieved_goal", "desired_goal"]

        self.obs_dim_separate = [
                            self.env.observation_space[name].shape[0]
                            for name in self.observation_names
                        ]
        self.obs_dim = sum(self.obs_dim_separate)

        self.obs_indices = np.arange(0, self.obs_dim_separate[0])
        self.achieved_indices = np.arange(self.obs_dim_separate[0],
                                          (self.obs_dim_separate[0] +
                                          self.obs_dim_separate[1]))
        self.desired_indices = np.arange((self.obs_dim_separate[0] +
                                          self.obs_dim_separate[1]),
                                         (self.obs_dim_separate[0] +
                                          self.obs_dim_separate[1] +
                                          self.obs_dim_separate[2]))

        # networks
        self.policy = Actor_gen(self.obs_dim, self.act_dim, hidden_layers=(64, 64), action_mult=self.act_high)
        self.value = Critic_gen(self.obs_dim, self.act_dim, hidden_layers=(128, 64, 64, 1))

        self.policy_target = Actor_gen(self.obs_dim, self.act_dim, hidden_layers=(64, 64), action_mult=self.act_high)
        self.value_target = Critic_gen(self.obs_dim, self.act_dim, hidden_layers=(128, 64, 64, 1))
        self.policy_target.set_weights(self.policy.get_weights())
        self.value_target.set_weights(self.value.get_weights())

        self.epochs = epochs
        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.her = HER(self.desired_indices, self.achieved_indices, self.obs_indices)

    def unload(self):

        return self.replay_buffer

    def clear_buffer(self):

        self.replay_buffer.clear()

    def update_wrapper(self, wrapper):

        self.wrapper = wrapper

    def get_updated_policy(self, policy, value, value_target, policy_target):

        self.policy.set_weights(policy)
        self.value.set_weights(value)
        self.value_target.set_weights(value_target)
        self.policy_target.set_weights(policy_target)

    def test_env(self, num_episodes=1):

        n_steps = 0
        all_returns = []
        for j in range(num_episodes):
            s, episode_return, episode_length, d = self.env.reset(), 0, 0, False
            s = flatten_goal_observation(s, self.observation_names)
            for _ in range(self.ep_len):
                # Take deterministic actions at test time (noise_scale=0)
                s, r, d, i = self.env.step(self.policy(s.reshape(1, -1).astype(np.float32))[0])
                episode_return += r
                episode_length += 1
                n_steps += 1
                s = flatten_goal_observation(s, self.observation_names)
            print('test return:', episode_return, 'episode_length:', episode_length)
            all_returns.append(episode_return)

        return np.vstack(all_returns)

    def gather(self):

        for episode in range(4):

            is_done = False
            observation, episode_reward = self.env.reset(), 0
            observation = flatten_goal_observation(observation, self.observation_names)
            for _ in range(self.ep_len):
                self.step_counter += 1

                action = self.get_action(observation)


                next_observation, reward, is_done, info = self.env.step(action)
                episode_reward += reward
                # update buffer
                next_observation = flatten_goal_observation(next_observation, self.observation_names)

                self.replay_buffer.append([observation, action, reward, next_observation, is_done])
                observation = next_observation

    def train(self):
        for i in range():
            num = len(self.replay_buffer)
            K = np.min([num, self.batch_size])
            batch = random.sample(self.replay_buffer, K)
            self._learn_on_batch(batch)

    def gather_her(self):

        for epoch in range(2000):
            self.her.reset()
            is_done = False
            observation, episode_reward = self.env.reset(), 0
            observation = flatten_goal_observation(observation, self.observation_names)
            for _ in range(self.ep_len):
                self.step_counter += 1
                action = self.get_action(observation)

                next_observation, reward, is_done, info = self.env.step(action)
                episode_reward += reward
                # update buffer
                next_observation = flatten_goal_observation(next_observation, self.observation_names)
                self.replay_buffer.append([observation, action, reward, next_observation, is_done])
                self.her.keep([observation, action, reward, next_observation, is_done])
                observation = next_observation
                her_list = self.her.backward()
                for item in her_list:
                    self.replay_buffer.append(item)
                self.her.reset()

                num = len(self.replay_buffer)
                K = np.min([num, self.batch_size])
                batch = random.sample(self.replay_buffer, K)
                self._learn_on_batch(batch)



            self.rewards.append(episode_reward)

            print(f"Epsiode {epoch}; Reward {episode_reward} Policy Loss {self.policy_losses[-1]} Value Loss {self.value_losses[-1]}")

    def major_gather(self):

        self.gather_her()
        self.test_env(5)


@ray.remote
class RemoteHERDDPG(HERDDPG):
    pass













