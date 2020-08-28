from collections import deque
import numpy as np
import copy


class HER:
    def __init__(self, goal_idx, achieved_idx, observation_idx):
        self.buffer = deque()
        self.goal_idx = goal_idx
        self.achieved_idx = achieved_idx
        self.observation_idx = observation_idx

    def reset(self):
        self.buffer = deque()

    def keep(self, item):
        self.buffer.append(item)

    def backward(self):
        num = len(self.buffer)
        # goal = self.buffer[-1][-2][self.achieved_idx]
        print(num)
        for i in range(num):
            goal = self.buffer[-1 - i][-2][self.achieved_idx]
            self.buffer[-1 - i][-2][self.goal_idx] = goal
            self.buffer[-1 - i][0][self.goal_idx] = goal
            self.buffer[-1 - i][2] = -1.0
            self.buffer[-1 - i][4] = False
            if np.linalg.norm(self.buffer[-1 - i][-2][self.achieved_idx] - goal) < 0.05:
                self.buffer[-1 - i][2] = 0.0
                self.buffer[-1 - i][4] = True
        return self.buffer