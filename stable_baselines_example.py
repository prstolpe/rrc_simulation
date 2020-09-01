from stable_baselines import HER
from stable_baselines.sac import SAC
import gym
from attempt_sac.environments import *
from rrc_simulation.gym_wrapper.envs.cube_env import RandomInitializer
from rrc_simulation.gym_wrapper.envs import cube_env


env = gym.make('Example_CubeEnv-v0',
               initializer=RandomInitializer(difficulty=1),
               action_type=cube_env.ActionType.POSITION,
               visualization=False,
               )

model = HER('MlpPolicy', env, SAC)

model.learn(int(1e4))
