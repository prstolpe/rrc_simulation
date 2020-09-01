from attempt_sac.models.models import *

from attempt_sac.utilities.replay_buffer import PlainReplayBuffer, HindsightExperienceReplayWrapper, \
        ReplayBuffer, GoalSelectionStrategy
from attempt_sac.utilities.her import HERGoalEnvWrapper
from attempt_sac.utilities.utils import env_extract_dims
import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import datetime
import imageio

tfd = tfp.distributions

tf.keras.backend.set_floatx('float64')

# paper https://arxiv.org/pdf/1812.05905.pdf
# code references https://github.com/anita-hu/TF2-RL/blob/master/SAC/TF2_SAC.py#L294
                 #https://github.com/StepNeverStop/RLs, https://github.com/rail-berkeley/softlearning


class SAC:

    def __init__(self, env: gym.Env, lr_actor=3e-4, lr_critic=3e-3, actor_units=(64, 64), critic_units=(64, 64),
                 auto_alpha=True, alpha=0.2, tau=0.005, gamma=0.99, batch_size=128, buffer_size=100000):

        # env
        self.env = env
        self.obs_dims, self.act_dims = env_extract_dims(env)
        self.act_low = env.action_space.low
        self.act_high = env.action_space.high
        self.action_bound = (env.action_space.high - env.action_space.low) / 2
        self.action_shift = (env.action_space.high + env.action_space.low) / 2

        # network params
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.actor_units = actor_units
        self.critic_units = critic_units

        # networks
        self.actor = actor(self.obs_dims, self.act_dims, actor_units)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        self.log_std_min, self.log_std_max = -20, 2
        print(self.actor.summary())

        # Define and initialize critic networks
        self.critic_1 = critic(self.obs_dims, self.act_dims, critic_units)
        self.critic_target_1 = critic(self.obs_dims, self.act_dims, critic_units)
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        update_target_weights(self.critic_1, self.critic_target_1, tau=1.)

        self.critic_2 = critic(self.obs_dims, self.act_dims, critic_units)
        self.critic_target_2 = critic(self.obs_dims, self.act_dims, critic_units)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(learning_rate=lr_critic)
        update_target_weights(self.critic_2, self.critic_target_2, tau=1.)
        print(self.critic_1.summary())

        # algo params
        self.auto_alpha = auto_alpha
        self.alpha = alpha
        if auto_alpha:
            self.target_entropy = -np.prod(self.act_dims)
            self.log_alpha = tf.Variable(0., dtype=tf.float64)
            self.alpha = tf.Variable(0., dtype=tf.float64)
            self.alpha.assign(tf.exp(self.log_alpha))
            self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
        else:
            self.alpha = tf.Variable(alpha, dtype=tf.float64)

        self.tau = tau
        self.gamma = gamma

        # buffer
        self.buffer_size = buffer_size
        self.replay_buffer = PlainReplayBuffer(self.obs_dims, self.act_dims, self.buffer_size)

        self.summaries = {}

    def process_actions(self, mean, log_std, test=False, eps=1e-6):
        std = tf.math.exp(log_std)
        raw_actions = mean

        if not test:
            raw_actions += tf.random.normal(shape=mean.shape, dtype=tf.float64) * std

        log_prob_u = tfd.Normal(loc=mean, scale=std).log_prob(raw_actions)
        actions = tf.math.tanh(raw_actions)

        log_prob = tf.reduce_sum(log_prob_u - tf.math.log(1 - actions ** 2 + eps))

        actions = actions * self.action_bound + self.action_shift

        return actions, log_prob

    def act(self, state, test=False, use_random=False):
        state = np.expand_dims(state, axis=0).astype(np.float64)

        if use_random:
            a = tf.random.uniform(shape=(1, self.act_dims), minval=-1, maxval=1, dtype=tf.float64)
        else:
            means, log_stds = self.actor.predict(state)
            log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)

            a, log_prob = self.process_actions(means, log_stds, test=test)

        q1 = self.critic_1.predict([state, a])[0][0]
        q2 = self.critic_2.predict([state, a])[0][0]
        self.summaries['q_min'] = tf.math.minimum(q1, q2)
        self.summaries['q_mean'] = np.mean([q1, q2])

        return a

    def save_model(self, a_fn, c_fn):
        self.actor.save(a_fn)
        self.critic_1.save(c_fn)

    def load_actor(self, a_fn):
        self.actor.load_weights(a_fn)
        print(self.actor.summary())

    def load_critic(self, c_fn):
        self.critic_1.load_weights(c_fn)
        self.critic_target_1.load_weights(c_fn)
        self.critic_2.load_weights(c_fn)
        self.critic_target_2.load_weights(c_fn)
        print(self.critic_1.summary())

    def replay(self):

        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(self.batch_size)

        with tf.GradientTape(persistent=True) as tape:
            # next state action log probs
            means, log_stds = self.actor(next_states)
            log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)
            next_actions, log_probs = self.process_actions(means, log_stds)

            # critics loss
            current_q_1 = self.critic_1([states, actions])
            current_q_2 = self.critic_2([states, actions])
            next_q_1 = self.critic_target_1([next_states, next_actions])
            next_q_2 = self.critic_target_2([next_states, next_actions])
            next_q_min = tf.math.minimum(next_q_1, next_q_2)
            state_values = next_q_min - self.alpha * log_probs
            target_qs = tf.stop_gradient(rewards + state_values * self.gamma * (1. - dones))
            critic_loss_1 = tf.reduce_mean(0.5 * tf.math.square(current_q_1 - target_qs))
            critic_loss_2 = tf.reduce_mean(0.5 * tf.math.square(current_q_2 - target_qs))

            # current state action log probs
            means, log_stds = self.actor(states)
            log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)
            actions, log_probs = self.process_actions(means, log_stds)

            # actor loss
            current_q_1 = self.critic_1([states, actions])
            current_q_2 = self.critic_2([states, actions])
            current_q_min = tf.math.minimum(current_q_1, current_q_2)
            actor_loss = tf.reduce_mean(self.alpha * log_probs - current_q_min)

            # temperature loss
            if self.auto_alpha:
                alpha_loss = -tf.reduce_mean(
                    (self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy)))

        critic_grad = tape.gradient(critic_loss_1, self.critic_1.trainable_variables)  # compute actor gradient
        self.critic_optimizer_1.apply_gradients(zip(critic_grad, self.critic_1.trainable_variables))

        critic_grad = tape.gradient(critic_loss_2, self.critic_2.trainable_variables)  # compute actor gradient
        self.critic_optimizer_2.apply_gradients(zip(critic_grad, self.critic_2.trainable_variables))

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  # compute actor gradient
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        # tensorboard info
        self.summaries['q1_loss'] = critic_loss_1
        self.summaries['q2_loss'] = critic_loss_2
        self.summaries['actor_loss'] = actor_loss

        if self.auto_alpha:
            # optimize temperature
            alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
            self.alpha.assign(tf.exp(self.log_alpha))
            # tensorboard info
            self.summaries['alpha_loss'] = alpha_loss

    def train(self, max_epochs=8000, random_epochs=1000, max_steps=1000, save_freq=50):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        done, use_random, episode, steps, epoch, episode_reward = False, True, 0, 0, 0, 0
        cur_state = self.env.reset()

        while epoch < max_epochs:
            if steps > max_steps:
                done = True

            if done:
                episode += 1
                print("episode {}: {} total reward, {} alpha, {} steps, {} epochs".format(
                    episode, episode_reward, self.alpha.numpy(), steps, epoch))

                with summary_writer.as_default():
                    tf.summary.scalar('Main/episode_reward', episode_reward, step=episode)
                    tf.summary.scalar('Main/episode_steps', steps, step=episode)

                summary_writer.flush()

                done, cur_state, steps, episode_reward = False, self.env.reset(), 0, 0
                #if episode % save_freq == 0:
                    #self.save_model("sac_actor_episode{}.h5".format(episode),
                     #               "sac_critic_episode{}.h5".format(episode))

            if epoch > random_epochs and self.buffer_size > self.batch_size:
                use_random = False

            action = self.act(cur_state, use_random=use_random)  # determine action
            next_state, reward, done, _ = self.env.step(action[0])  # act on env
            #self.env.render(mode='rgb_array')

            self.replay_buffer.store(cur_state, action, reward, next_state, done)  # add to memory
            self.replay()  # train models through memory replay

            update_target_weights(self.critic_1, self.critic_target_1, tau=self.tau)  # iterates target model
            update_target_weights(self.critic_2, self.critic_target_2, tau=self.tau)

            cur_state = next_state
            episode_reward += reward
            steps += 1
            epoch += 1

            # Tensorboard update
            with summary_writer.as_default():
                if self.buffer_size > self.batch_size:
                    tf.summary.scalar('Loss/actor_loss', self.summaries['actor_loss'], step=epoch)
                    tf.summary.scalar('Loss/q1_loss', self.summaries['q1_loss'], step=epoch)
                    tf.summary.scalar('Loss/q2_loss', self.summaries['q2_loss'], step=epoch)
                    if self.auto_alpha:
                        tf.summary.scalar('Loss/alpha_loss', self.summaries['alpha_loss'], step=epoch)

                tf.summary.scalar('Stats/alpha', self.alpha, step=epoch)
                if self.auto_alpha:
                    tf.summary.scalar('Stats/log_alpha', self.log_alpha, step=epoch)
                tf.summary.scalar('Stats/q_min', self.summaries['q_min'], step=epoch)
                tf.summary.scalar('Stats/q_mean', self.summaries['q_mean'], step=epoch)
                tf.summary.scalar('Main/step_reward', reward, step=epoch)

            summary_writer.flush()

        self.save_model("sac_actor_final_episode{}.h5".format(episode),
                        "sac_critic_final_episode{}.h5".format(episode))

    def test(self, render=True, fps=30, filename='test_render.mp4'):
        cur_state, done, rewards = self.env.reset(), False, 0
        video = imageio.get_writer(filename, fps=fps)
        while not done:
            action = self.act(cur_state, test=True)
            next_state, reward, done, _ = self.env.step(action[0])
            cur_state = next_state
            rewards += reward
            if render:
                video.append_data(self.env.render(mode='rgb_array'))
        video.close()
        return rewards


class HERSAC(SAC):

    def __init__(self, env: gym.Env, lr_actor=3e-4, lr_critic=3e-4, actor_units=(64, 64, 64), critic_units=(64, 64, 64),
                 auto_alpha=True, alpha=0.2, tau=0.005, gamma=0.95, batch_size=128, buffer_size=int(1e6),
                 goal_selection_strategy=GoalSelectionStrategy.FUTURE, k = 4):

        super().__init__(env, lr_actor=lr_actor, lr_critic=lr_critic, actor_units=actor_units,
                         critic_units=critic_units, auto_alpha=auto_alpha, alpha=alpha, tau=tau,
                         gamma=gamma, batch_size=batch_size, buffer_size=buffer_size)

        self.env = HERGoalEnvWrapper(env)
        self.replay_buffer = HindsightExperienceReplayWrapper(ReplayBuffer(self.buffer_size), k, goal_selection_strategy,
                                                              self.env)

    def replay(self):

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        with tf.GradientTape(persistent=True) as tape:
            # next state action log probs
            means, log_stds = self.actor(next_states)
            log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)
            next_actions, log_probs = self.process_actions(means, log_stds)

            # critics loss
            current_q_1 = self.critic_1([states, actions])
            current_q_2 = self.critic_2([states, actions])
            next_q_1 = self.critic_target_1([next_states, next_actions])
            next_q_2 = self.critic_target_2([next_states, next_actions])
            next_q_min = tf.math.minimum(next_q_1, next_q_2)
            state_values = next_q_min - self.alpha * log_probs
            target_qs = tf.stop_gradient(rewards + state_values * self.gamma * (1. - dones))
            critic_loss_1 = tf.reduce_mean(0.5 * tf.math.square(current_q_1 - target_qs))
            critic_loss_2 = tf.reduce_mean(0.5 * tf.math.square(current_q_2 - target_qs))

            # current state action log probs
            means, log_stds = self.actor(states)
            log_stds = tf.clip_by_value(log_stds, self.log_std_min, self.log_std_max)
            actions, log_probs = self.process_actions(means, log_stds)

            # actor loss
            current_q_1 = self.critic_1([states, actions])
            current_q_2 = self.critic_2([states, actions])
            current_q_min = tf.math.minimum(current_q_1, current_q_2)
            actor_loss = tf.reduce_mean(self.alpha * log_probs - current_q_min)

            # temperature loss
            if self.auto_alpha:
                alpha_loss = -tf.reduce_mean(
                    (self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy)))

        critic_grad = tape.gradient(critic_loss_1, self.critic_1.trainable_variables)  # compute actor gradient
        self.critic_optimizer_1.apply_gradients(zip(critic_grad, self.critic_1.trainable_variables))

        critic_grad = tape.gradient(critic_loss_2, self.critic_2.trainable_variables)  # compute actor gradient
        self.critic_optimizer_2.apply_gradients(zip(critic_grad, self.critic_2.trainable_variables))

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)  # compute actor gradient
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        # tensorboard info
        self.summaries['q1_loss'] = critic_loss_1
        self.summaries['q2_loss'] = critic_loss_2
        self.summaries['actor_loss'] = actor_loss

        if self.auto_alpha:
            # optimize temperature
            alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grad, [self.log_alpha]))
            self.alpha.assign(tf.exp(self.log_alpha))
            # tensorboard info
            self.summaries['alpha_loss'] = alpha_loss

    def train(self, max_epochs=8000, random_epochs=1000, max_steps=1000, save_freq=50):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time
        summary_writer = tf.summary.create_file_writer(train_log_dir)

        done, use_random, episode, steps, epoch, episode_reward = False, True, 0, 0, 0, 0
        cur_state = self.env.reset()

        while epoch < max_epochs:
            if steps > max_steps:
                done = True

            if done:
                episode += 1
                print("episode {}: {} total reward, {} alpha, {} steps, {} epochs".format(
                    episode, episode_reward, self.alpha.numpy(), steps, epoch))

                with summary_writer.as_default():
                    tf.summary.scalar('Main/episode_reward', episode_reward, step=episode)
                    tf.summary.scalar('Main/episode_steps', steps, step=episode)

                summary_writer.flush()

                done, cur_state, steps, episode_reward = False, self.env.reset(), 0, 0
                #if episode % save_freq == 0:
                    #self.save_model("sac_actor_episode{}.h5".format(episode),
                     #               "sac_critic_episode{}.h5".format(episode))

            if epoch > random_epochs and self.buffer_size > self.batch_size:
                use_random = False

            action = self.act(cur_state, use_random=use_random)  # determine action
            next_state, reward, done, _ = self.env.step(action[0])  # act on env
            #self.env.render(mode='rgb_array')

            self.replay_buffer.add(cur_state, action[0], reward, next_state, done, _)  # add to memory
            if self.replay_buffer.can_sample(self.batch_size):
                self.replay()  # train models through memory replay

                update_target_weights(self.critic_1, self.critic_target_1, tau=self.tau)  # iterates target model
                update_target_weights(self.critic_2, self.critic_target_2, tau=self.tau)

                cur_state = next_state
                episode_reward += reward
                steps += 1
                epoch += 1

                # Tensorboard update
                with summary_writer.as_default():
                    if self.buffer_size > self.batch_size:
                        tf.summary.scalar('Loss/actor_loss', self.summaries['actor_loss'], step=epoch)
                        tf.summary.scalar('Loss/q1_loss', self.summaries['q1_loss'], step=epoch)
                        tf.summary.scalar('Loss/q2_loss', self.summaries['q2_loss'], step=epoch)
                        if self.auto_alpha:
                            tf.summary.scalar('Loss/alpha_loss', self.summaries['alpha_loss'], step=epoch)

                    tf.summary.scalar('Stats/alpha', self.alpha, step=epoch)
                    if self.auto_alpha:
                        tf.summary.scalar('Stats/log_alpha', self.log_alpha, step=epoch)
                    tf.summary.scalar('Stats/q_min', self.summaries['q_min'], step=epoch)
                    tf.summary.scalar('Stats/q_mean', self.summaries['q_mean'], step=epoch)
                    tf.summary.scalar('Main/step_reward', reward, step=epoch)

                summary_writer.flush()

        self.save_model("sac_actor_final_episode{}.h5".format(episode),
                        "sac_critic_final_episode{}.h5".format(episode))



