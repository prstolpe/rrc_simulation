from collections import deque
import numpy as np
import copy


class HER:
    def __init__(self):
        self.buffer = deque()

    def reset(self):
        self.buffer = deque()

    def keep(self, item):
        self.buffer.append(item)

    def backward(self):
        num = len(self.buffer)

        for i in range(num):
            goal = self.buffer[-1 - i][-2][13:16]
            self.buffer[-1 - i][-2][10:13] = goal
            self.buffer[-1 - i][0][10:13] = goal
            self.buffer[-1 - i][2] = -1.0
            self.buffer[-1 - i][4] = False
            if np.linalg.norm(self.buffer[-1 - i][-2][13:16] - goal) < 0.05:
                self.buffer[-1 - i][2] = 0.0
                self.buffer[-1 - i][4] = True
        return self.buffer


