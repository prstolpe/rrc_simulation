import gym
from attempt_sac.sac import HERSAC


if __name__ == "__main__":
    gym_env = gym.make("FetchReach-v1")
    sac = HERSAC(gym_env)
    sac.train(max_epochs=200000, random_epochs=1000, save_freq=50)
