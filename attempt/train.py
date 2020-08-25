from attempt.ddpg import DDPG
import gym

if __name__ == "__main__":
    env = gym.make('Pendulum-v0')
    agent = DDPG(env)
    agent.drill()