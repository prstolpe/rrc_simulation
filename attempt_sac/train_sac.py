import gym
from attempt_sac.sac import HERSAC, SAC

from attempt_sac.environments import *
from rrc_simulation.gym_wrapper.envs.cube_env import RandomInitializer
from rrc_simulation.gym_wrapper.envs import cube_env
if __name__ == "__main__":
    env = gym.make('Example_CubeEnv-v0',
                   initializer=RandomInitializer(difficulty=1),
                   action_type=cube_env.ActionType.POSITION,
                   visualization=False,
                   )

    sac = HERSAC(env)
    sac.train(max_epochs=int(2e6), random_epochs=10000, save_freq=50)
