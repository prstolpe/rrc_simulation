import numpy as np
import tensorflow as tf
import gym
import ray

from attempt.utilities.utils import env_extract_dims
from attempt.models.models import Critic_gen, Actor_gen

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
        temp_dict = dict(s=self.obs1_buf[idxs],
                         s2=self.obs2_buf[idxs],
                         a=self.acts_buf[idxs],
                         r=self.rews_buf[idxs],
                         d=self.done_buf[idxs])
        return (temp_dict['s'], temp_dict['a'], temp_dict['r'].reshape(-1, 1), temp_dict['s2'], temp_dict['d'])


class DDPG:

    def __init__(self, env:gym.Env, buffer_size:int=int(1e5), seed:int=5, num_episodes:int=30,
                 batch_size=16, gamma:int=0.99, tau:int=1e-2, start_steps:int=1000, actor_lr=1e-3,
                 value_lr=1e-3):

        # env
        self.obs_dim, self.act_dim = env_extract_dims(env)
        self.env = env
        self.act_low, self.act_high = self.env.action_space.low, self.env.action_space.high

        # replay buffer
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, buffer_size)

        # networks
        self.policy = Actor_gen(self.obs_dim, self.act_dim, hidden_layers=(512, 200, 128), action_mult=self.act_high)
        self.value = Critic_gen(self.obs_dim, 1, hidden_layers=(1024, 512, 300, 1))

        self.policy_target = Actor_gen(self.obs_dim, self.act_dim, hidden_layers=(512, 200, 128))
        self.value_target = Critic_gen(self.obs_dim, 1, hidden_layers=(1024, 512, 300, 1))
        self.policy_target.set_weights(self.policy.get_weights())
        self.value_target.set_weights(self.value.get_weights())

        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=value_lr)

        # ddpg hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.num_episodes = num_episodes

        self.seed = np.random.seed(seed)
        self.step_counter = 0
        self.start_steps = start_steps
        self.batch_size = batch_size

        # monitoring
        self.rewards = []
        self.value_losses = []
        self.policy_losses = []

    def get_action(self, s):

        a = self.policy(s.reshape(1, -1))[0]
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
        states, actions, rewards, next_states, done = batch
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
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



