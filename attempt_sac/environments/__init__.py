import gym

gym.envs.register(
    id='Example_CubeEnv-v0',
    entry_point='attempt_sac.environments.example_pushing_env:CubeEnv'
)