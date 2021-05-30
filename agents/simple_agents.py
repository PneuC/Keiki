from random import randrange
from utils.math import Vec2
from abc import abstractmethod
from logic.objects.player import Player

class Agent:
    # stop, up, down, left, right, up left, up right, down left, down right
    up_set, down_set = {1, 5, 6}, {2, 7, 8}
    left_set, right_set = {3, 5, 6}, {4, 7, 8}

    def __init__(self):
        self.action = randrange(17)

    @abstractmethod
    def update(self):
        pass

    @staticmethod
    def decode_action(action):
        slow_down = action > 8
        dire_id = action - 8 if slow_down else action
        dire_vec = Vec2(0, 0)
        if dire_id in Agent.up_set:
            dire_vec.y = 1.0
        elif dire_id in Agent.down_set:
            dire_vec.y = -1.0
        if dire_id in Agent.left_set:
            dire_vec.x = -1.0
        elif dire_id in Agent.right_set:
            dire_vec.x = 1.0
        return dire_vec, slow_down

    def get_dvec(self):
        return Agent.decode_action(self.action)


class RandomAgent(Agent):
    def __init__(self, freq=10):
        super(RandomAgent, self).__init__()
        self.freq = freq
        self.coldown = freq

    def update(self):
        if self.coldown == 0:
            self.action = randrange(17)
            self.coldown = self.freq
        else:
            self.coldown -= 1


class PreinstalledAgent(Agent):
    def __init__(self, action_seq):
        super(PreinstalledAgent, self).__init__()
        self.action_seq = action_seq
        self.action_seq.reverse()

    def update(self):
        if self.action_seq:
            self.action = self.action_seq.pop()
        else:
            self.action = 0

