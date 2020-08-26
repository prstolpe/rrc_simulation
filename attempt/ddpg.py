import numpy as np
import tensorflow as tf
import gym
import ray

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

    def __init__(self, env:gym.Env, buffer_size:int=int(1e5), seed:int=5, update_every=1000,
                 batch_size=16, gamma:int=0.99, tau:int=0.99, start_steps:int=1000):

        # env
        self.obs_dim, self.act_dim = env_extract_dims(env)
        self.env = env
        self.act_low, self.act_high = self.env.action_space.low, self.env.action_space.high


        # replay buffer
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, buffer_size)

        # networks
        self.policy, self.value, self.policy_target, self.value_target = build_ffn_models(env)
        self.policy_target.set_weights(self.policy.get_weights())
        self.value_target.set_weights(self.value.get_weights())
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # ddpg hyperparameters
        self.gamma = gamma
        self.tau = tau

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
        next_policy_actions = self.act_high *   self.policy(states)
        return - tf.reduce_mean(self.value(tf.concat((states, next_policy_actions), axis=-1)))

    def _value_loss(self, states, actions, next_states, rewards, done):
        Qvals = self.value(tf.concat((states, actions), axis=-1))
        next_actions = self.policy_target(next_states)
        next_Q = self.value_target(tf.concat((next_states, next_actions), axis=-1))

        Qprime = rewards + self.gamma * (1 - done) * next_Q

        return tf.reduce_mean(tf.square(Qvals - Qprime))

    def _learn_on_batch(self, batch):
        states, next_states, actions, rewards, done = batch['obs1'], batch['obs2'], \
                                                      batch['acts'], batch['rews'], \
                                                      batch['done']

        # value optimization
        with tf.GradientTape() as tape:
            value_loss = self._value_loss(states, actions, next_states, rewards, done)

            value_gradients = tape.gradient(value_loss, self.value.trainable_variables)
        self.value_optimizer.apply_gradients(zip(value_gradients, self.value.trainable_variables))

        # policy optimization
        with tf.GradientTape() as tape2:
            policy_loss = self._policy_loss(states)
            policy_gradients = tape2.gradient(policy_loss, self.policy.trainable_variables)
        self.policy_optimizer.apply_gradients(zip(policy_gradients, self.policy.trainable_variables))



        # updating target network
        temp1 = np.array(self.value_target.get_weights())
        temp2 = np.array(self.value.get_weights())
        temp3 = self.tau * temp1 + (1 - self.tau) * temp2
        self.value_target.set_weights(temp3)

        # updating Actor network
        temp1 = np.array(self.policy_target.get_weights())
        temp2 = np.array(self.policy.get_weights())
        temp3 = self.tau * temp1 + (1 - self.tau) * temp2
        self.policy_target.set_weights(temp3)

    @ray.remote
    def gather(self):
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


    def drill(self):

        ray.init(num_cpus=10)

        h = [self.gather.remote() for episode in range(100)]
        ray.get(h)
        if self.step_counter % 200 == 0:
            for i in range(100):
                batch = self.replay_buffer.sample_batch(self.batch_size)
                self._learn_on_batch(batch)
        self.rewards.append(episode_reward)
        print(episode_reward)
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



