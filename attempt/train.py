from attempt.ddpg import DDPG
import gym
if __name__ == "__main__":
    env = gym.make("LunarLanderContinuous-v2")
    agent = DDPG(env)
    agent.drill()