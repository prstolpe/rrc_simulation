import gym
from attempt_sac.sac import SAC


if __name__ == "__main__":
    gym_env = gym.make("Pendulum-v0")
    sac = SAC(gym_env, lr_actor=1e-3, lr_critic=1e-2)
    sac.train(max_epochs=20000, random_epochs=10000, save_freq=50)
