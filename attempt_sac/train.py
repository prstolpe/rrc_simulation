import gym
from attempt_sac.sac import SAC


if __name__ == "__main__":
    gym_env = gym.make("MountainCarContinuous-v0")
    sac = SAC(gym_env)
    sac.train(max_epochs=200000, random_epochs=10000, save_freq=50)