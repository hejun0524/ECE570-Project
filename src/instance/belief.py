import numpy as np
from src.instance.constants import *

class BeliefController:
    def __init__(self, n_category):
        self.n_category = n_category
        self.belief_dict = {}  # belief price
        self.real_time_dict = {}  # real time price
        self.delta = 0.9
        self.learned_dict = {}  # ever learned?

        for i in range(n_category):
            self.hard_set(i, 0.0)
            self.learned_dict[i] = False

    def hard_set(self, i, data):
        self.real_time_dict[i] = data
        self.belief_dict[i] = data

    def update(self, i, data: dict):
        self.real_time_dict[i] = data[LMP]
        if not self.learned_dict[i]:
            self.belief_dict[i] = np.round(data[LMP])
            self.learned_dict[i] = True
        else:
            td = self.belief_dict[i] - data[LMP]
            belief = self.belief_dict[i] - self.delta * td / np.sqrt(data[DAY] + 1)
            self.belief_dict[i] = np.round(belief)

    def get_value(self, i, is_real_time=False):
        if is_real_time:
            return self.real_time_dict[i]
        return self.belief_dict[i]

    def get_values(self, shift=0, scale=1, is_real_time=False):
        return [
            (self.get_value(i, is_real_time) - shift) / scale
            for i in range(self.n_category)
        ]

    def get_belief_key(self):
        keys = [int(self.get_value(i)) for i in range(self.n_category)]
        return tuple(keys)
