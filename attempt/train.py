from attempt.ddpg import DDPG
import gym
import ray

if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    agent = DDPG(env)
    agent.drill()
