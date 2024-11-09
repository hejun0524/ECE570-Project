import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.instance.constants import *
from src.instance.belief import BeliefController as Belief
from src.instance.simulation import SimulationInstance


class AggregatorEnv(gym.Env):
    def __init__(self, *, config: dict):
        super().__init__()
        # unpack parameters from config
        self.debug_mode = config["debug_mode"]
        self.aid = config["prosumer_id"]
        self.n_steps_one_day = config["n_steps_one_day"]
        self.prosumer_shape = config["prosumer_shape"]
        self.prosumers_spec = config["prosumer_spec"]
        self.capacity = self.prosumers_spec[CAPACITY]
        self.ref_capacity = self.prosumers_spec[REF_CAPACITY]
        self.efficiency = self.prosumers_spec[EFFICIENCY]
        self.n_days_episode = config["n_days_episode"]
        # define spaces
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                CURRENT_BELIEF: gym.spaces.Box(low=-2, high=2, shape=(1,)),
                HOUR_OF_DAY: gym.spaces.Box(low=0, high=1, shape=(1,)),
                NET_DEMAND: gym.spaces.Box(low=0, high=2, shape=(1,)),
                BATTERY_LEVEL: gym.spaces.Box(low=0, high=1, shape=(1,)),
            }
        )
        # define env attributes
        self.decimal = 2
        self.price_shift = 150.0  # shift it (LMP = LMP - SHIFT)
        self.price_factor = 30.0  # tune this to normalize the LMPs
        self.reward_factor = 125  # tune this to normalize the rew
        self.nd_factor = (
            0.8  # tune this to normalize the nd (I think sometimes its too small)
        )
        # each hour keeps a log of 4 prices
        self.lmp_trace = Belief(self.n_steps_one_day)
        self.hour_of_day = 0
        self.net_demand = 0
        self.battery_level = 0
        self.t = 0  # to control episode termination
        self.legal_actions = self.set_action_masks()
        self.day = 0

    def ratio_to_quantity(self, ratio, use_true_cap):
        """
        Given the agent is charging `ratio` amount,
        convert it to actual quantity in MWh (/1000)
        """
        eta = self.efficiency
        cap = self.capacity if use_true_cap else self.ref_capacity
        q = ratio * cap
        if ratio < 0:
            return eta * q / 1000  # discharging #FIXME UNIT
        return q / eta / 1000  # charging #FIXME UNIT

    def sample_shape(self, t, randomize=True):
        data_pt = self.prosumer_shape[t]
        if randomize:
            data_pt *= np.random.triangular(0.9, 1.0, 1.1)
        return round(data_pt, self.decimal)

    def get_observation(self):
        current_belief = self.lmp_trace.get_value(self.hour_of_day)
        current_belief = (current_belief - self.price_shift) / self.price_factor
        current_belief = max(-2, min(2, current_belief))
        return {
            CURRENT_BELIEF: [current_belief],
            HOUR_OF_DAY: [self.hour_of_day / self.n_steps_one_day],
            NET_DEMAND: [self.net_demand / self.nd_factor],
            BATTERY_LEVEL: [self.battery_level],
        }

    def set_observation(self, obs: dict, update_LMP=True):
        """
        Set all observation
        """
        if HOUR_OF_DAY in obs:
            self.hour_of_day = obs[HOUR_OF_DAY]
        if NET_DEMAND in obs:
            self.net_demand = obs[NET_DEMAND]
        if BATTERY_LEVEL in obs:
            self.battery_level = obs[BATTERY_LEVEL]
        if TIME_COUNTER in obs:
            self.t = obs[TIME_COUNTER]
        if DAY in obs:
            self.day = obs[DAY]
        if LMP in obs and update_LMP:
            self.update_lmp(obs[LMP])
        if TAX_RATE in obs:
            self.tax_rate = obs[TAX_RATE]
        # update the action mask
        self.set_action_masks()

    def set_action_masks(self):
        """
        Give the env values of how much one can discharge/charge,
        set the action masks of those to True.
        """
        lb, ub = -self.battery_level, 1 - self.battery_level
        legals = [
            lb <= get_continuous_action(act) <= ub for act in range(self.action_space.n)
        ]
        self.legal_actions = np.array(legals)
        return self.legal_actions

    def get_action_masks(self):
        return self.legal_actions

    def update_lmp(self, lmp_dict: dict):
        """
        Update lmp where arg is {h: lmp, ...}
        """
        for h, lmp in lmp_dict.items():
            self.lmp_trace.update(h, {LMP: lmp, DAY: self.day})

    def get_lmp_trace_key(self):
        return self.lmp_trace.get_belief_key()

    def validate_action(self, new_bat, curr_bat, true_action, action):
        """
        Validate if the action taken follows the masks
        """
        out_of_range = 0 if 0 <= new_bat <= 1 else max(new_bat - 1, -new_bat)
        if out_of_range >= 1e-3:
            print(f"[ILLEGAL ACT ({self.aid})] @ step {self.t}")
            legals = np.where(self.legal_actions[self.aid])[0]  # needs to unpack
            print(f" - current legal actions: {legals[0]} to {legals[-1]}")
            print(f" - current transition: {curr_bat} + {true_action} -> {new_bat}")
            print(f" - where discrete action is {action}")
            raise ValueError

    def step(self, action):
        # take action and calc reward
        true_action = get_continuous_action(action)
        true_net_demand = self.sample_shape(self.hour_of_day)
        curr_battery = self.battery_level
        new_battery = curr_battery + true_action
        if self.debug_mode:
            self.validate_action(new_battery, curr_battery, true_action, action)
        new_battery = max(min(new_battery, 1), 0)
        q = self.ratio_to_quantity(true_net_demand, False)
        q = self.ratio_to_quantity(
            new_battery - curr_battery, True
        )  # let's just focus on actions
        curr_belief = self.lmp_trace.get_value(self.hour_of_day) - self.price_shift
        reward = -curr_belief * q / self.reward_factor

        # state transition
        self.hour_of_day += 1
        if self.hour_of_day == self.n_steps_one_day:
            self.hour_of_day = 0
        self.net_demand = self.sample_shape(self.hour_of_day, False)
        self.battery_level = new_battery
        self.set_action_masks()

        # termination happens at the end of Day 3
        self.t += 1
        terminated = self.t >= self.n_days_episode * self.n_steps_one_day
        # terminated = False  # never end
        truncated = terminated
        if terminated or truncated:
            self.t = 0  # reset
        # now get observation
        observation = self.get_observation()
        # additional info is always empty
        info = {}
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.set_observation(
            {
                HOUR_OF_DAY: 0,
                NET_DEMAND: self.sample_shape(0, False),
                BATTERY_LEVEL: np.random.rand(),
                TIME_COUNTER: 0,
                DAY: self.day,
            }
        )
        observation = self.get_observation()
        info = {}
        return observation, info

    def render(self):
        pass

    def close(self):
        pass


class EnvConfig:
    def __init__(self, ins: SimulationInstance):
        self.debug_mode = True  # change this to False to disable debugging
        self.n_steps_one_day = ins.clock.n_steps_one_day
        self.prosumer_shape = [nd for nd in ins.prosumer_shape.data]
        self.consumer_shape = [nd for nd in ins.consumer_shape.data]
        self.prosumers = ins.representatives
        self.prosumers_spec = {
            a.aid: {
                CAPACITY: a.storage_capacity,
                REF_CAPACITY: a.reference_capacity,
                EFFICIENCY: a.storage_efficiency,
            }
            for a in ins.representatives
        }
        self.n_days_episode = 3
        self.prosumers_id = [a.aid for a in ins.representatives]

    def as_dict(self, agent_id: str):
        config = {
            "debug_mode": self.debug_mode,
            "n_steps_one_day": self.n_steps_one_day,
            "prosumer_shape": self.prosumer_shape,
            "consumer_shape": self.consumer_shape,
            "prosumers": self.prosumers,
            "n_days_episode": self.n_days_episode,
            "prosumer_id": agent_id,
            "prosumer_spec": self.prosumers_spec[agent_id],
        }
        return config
