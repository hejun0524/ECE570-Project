import numpy as np

class Clock:
    def __init__(self, T, time_step) -> None:
        self.T = T # total time in hours (eg. 48)
        self.time_step = time_step # each step length (eg. 1)
        self.n_steps = T // time_step # total n of steps (eg. 48)
        self.n_steps_one_day = 24 // time_step # n steps in one day (eg. 24) 
        self.time_counter = 0 # current time counter
        self.time_now = 0 # current time in minutes 
        self.day = 0 # what day it is

    def proceed_time(self):
        # proceed time by one step, always proceed at the end
        self.time_counter += 1
        self.time_now += self.time_step
        return self.time_counter, self.get_time_counter_of_day(), self.get_day()

    def is_over(self):
        # return true if the clock is at its last time
        return self.time_now >= self.T 
    
    def reset(self):
        # reset the clock to the first time counter
        self.time_counter = 0
        self.time_now = 0

    def get_time_counter_of_day(self, time_counter=None):
        if time_counter is None:
            time_counter = self.time_counter
        return time_counter % self.n_steps_one_day
    
    def get_day(self, time_counter=None):
        if time_counter is None:
            time_counter = self.time_counter
        return time_counter // self.n_steps_one_day
    
    def synchronize_dimension(self, arr, arr_T, arr_time_step):
        # repeat outer (tile: [1,2] -> [1,2,1,2]) 
        outer = int(np.ceil(self.T / arr_T))
        new_arr = np.tile(arr, outer)
        # repeat or group inner (repeat: [1,2] -> [1,1,2,2])
        if arr_time_step >= self.time_step:
            inner = arr_time_step // self.time_step
            new_arr = np.repeat(new_arr, inner)
        else:
            group = self.time_step // arr_time_step
            new_arr = np.sum(np.reshape(new_arr, (-1, group)), axis=-1)
        return new_arr[:self.n_steps]