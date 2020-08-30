import numpy as np
import tensorflow as tf
import gym
import ray
from collections import deque
import random

from attempt.utilities.utils import env_extract_dims, flatten_goal_observation
from attempt.models.models import Critic_gen, Actor_gen
from attempt.utilities.her import HERGoalEnvWrapper
from attempt.utilities.replay_buffer import PlainReplayBuffer, HindsightExperienceReplayWrapper, ReplayBuffer, GoalSelectionStrategy


class DDPG:

    def __init__(self, env: gym.Env, buffer_size: int=int(1e6), gamma: int = 0.99, tau: int = 1e-2, start_steps: int=100,
                 noise_scale: float=0.1, batch_size=32, actor_lr=1e-3, value_lr=1e-3, seed: int=5):

        # env
        self.obs_dim, self.act_dim = env_extract_dims(env)
        self.env = env
        self.act_low, self.act_high = self.env.action_space.low, self.env.action_space.high

        # replay buffer
        self.buffer_size = buffer_size
        self.replay_buffer = PlainReplayBuffer(self.obs_dim, self.act_dim, buffer_size)

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

    def __init__(self, env: gym.GoalEnv, goal_selection_strategy=GoalSelectionStrategy.RANDOM, buffer_size: int=int(1e5),
                 gamma: int = 0.99, tau: int = 1e-2, start_steps: int=1000, noise_scale: float=0.1,
                 batch_size=int(128), actor_lr=1e-3, value_lr=1e-3, seed: int=5, k=4):

        super().__init__(env, buffer_size, gamma, tau, start_steps, noise_scale,
                         batch_size, actor_lr, value_lr, seed)

        self.env = HERGoalEnvWrapper(env)
        self.goal_selection_strategy = goal_selection_strategy
        self.k = k
        self.replay_buffer = HindsightExperienceReplayWrapper(ReplayBuffer(self.buffer_size), k,
                                                              goal_selection_strategy, self.env)
        self.ep_len = self.env.env.spec.max_episode_steps

        print(f"Training HER + DDPG Agent on {self.env.env.spec.id} using \n"
              f"{goal_selection_strategy} \n"
              f"and k = {k}")

    def drill(self, epochs=10, num_episodes=50):
        num_steps = 0
        self.actions = []
        for epoch in range(epochs):
            for episode in range(num_episodes):

                observation, episode_reward = self.env.reset(), 0
                for _ in range(self.ep_len):
                    num_steps += 1
                    if num_steps > self.start_steps:
                        action = self.get_action(observation)
                    else:
                        action = self.env.action_space.sample()

                    next_observation, reward, is_done, info = self.env.step(action)
                    self.actions.append(action)

                    episode_reward += reward
                    # update buffer
                    self.replay_buffer.add(observation, action, reward, next_observation, is_done, info)
                    observation = next_observation

                    if self.replay_buffer.can_sample(self.batch_size):
                        batch = self.replay_buffer.sample(self.batch_size)
                        self._learn_on_batch(batch)

                self.rewards.append(episode_reward)
                print("Epoch " + str(epoch) + " Episode " + str(episode) + ": " + str(episode_reward) + " reward")

            self.test_env(3)

    def test_env(self, num_episodes=1, render=False):

        n_steps = 0
        for j in range(num_episodes):
            s, episode_return, episode_length, d = self.env.reset(), 0, 0, False
            if render:
                self.env.render()
            while not d:
                # Take deterministic actions at test time (noise_scale=0)
                s, r, d, _ = self.env.step(self.policy_target(s.reshape(1, -1))[0])
                if render:
                    self.env.render()
                episode_return += r
                episode_length += 1
                n_steps += 1
            print('test return:', episode_return, 'episode_length:', episode_length)