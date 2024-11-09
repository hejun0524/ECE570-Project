import numpy as np


class Shape:
    def __init__(
        self, name, raw_data, T, time_step, noise_lb=0.9, noise_ub=1.1
    ) -> None:
        self.name = name  # name of the shape
        self.data = raw_data  # data adjusted with clock
        self.raw_data = raw_data  # original data
        self.T = T  # original total time in hours
        self.time_step = time_step  # original hours per step
        self.decimal = 5
        self.noise_lb = noise_lb
        self.noise_ub = noise_ub

    def average(self):
        return round(np.mean(self.data), self.decimal)

    def sample(self, time_counter, randomize=True):
        data_pt = self.data[time_counter]
        if randomize:
            data_pt *= np.random.triangular(self.noise_lb, 1.0, self.noise_ub)
        return round(data_pt, self.decimal)

    def get_value(self, time_counter):
        return self.sample(time_counter, False)

    def reload(self, clock):
        # for simplicity, make sure the following dimensions match
        assert self.T >= clock.T
        assert self.time_step == clock.time_step

        # randomly select a starting pt in between 0 and T
        if self.T == clock.T:
            p0, p1 = 0, clock.T
        else:
            p0 = np.random.choice(np.arange(0, self.T, clock.T)[:-1])
            p1 = p0 + clock.T

        # dimension adjustment
        self.data = self.raw_data[p0:p1]
