from src.instance.clock import Clock
from src.instance.shape import Shape
from src.instance.network import Network
from src.instance.agent import Agent
from src.instance.dispatch import EconomicDispatch
from numpy.random import randint

class SimulationInstance:
    def __init__(
        self,
        name: str,
        clock: Clock,
        network: Network,
        prosumer_shape: Shape,
        consumer_shape: Shape,
        prosumers: list[Agent],
        consumers: list[Agent],
        representatives: list[Agent],
        dispatch: EconomicDispatch,
    ) -> None:
        self.name = name
        self.clock = clock
        self.network = network
        self.prosumer_shape = prosumer_shape
        self.consumer_shape = consumer_shape
        self.prosumers = prosumers
        self.consumers = consumers
        self.representatives = representatives
        self.dispatch = dispatch

    def set_rl_configs(self, *, random_seed: int, n_episodes: int, evaluate: bool):
        self.random_seed = random_seed
        self.n_episodes = n_episodes
        self.evaluate = evaluate

    def reset_and_reload(self):
        self.clock.reset()
        self.prosumer_shape.reload(self.clock)
        self.consumer_shape.reload(self.clock)

    def reset_lmp(self, random_lmps=False):
        total_hourly_bids = [0 for _ in range(self.clock.n_steps_one_day)]

        # start with random lmps
        if random_lmps:
            for bus in self.network.buses:
                for t in range(self.clock.n_steps_one_day):
                    lmp = randint(self.dispatch.lb_lmp, self.dispatch.ub_lmp)
                    self.dispatch.lmps[bus.index][t] = lmp
            return total_hourly_bids # nobody bids
        
        # start with solving a vanilla dispatch
        for t in range(self.clock.n_steps_one_day):
            demands = {bus.index: 0.0 for bus in self.network.buses}
            for agent in self.prosumers:
                ratio = self.prosumer_shape.sample(t, False)
                demands[agent.bus.index] += agent.ratio_to_quantity(ratio, False)
            for agent in self.consumers:
                ratio = self.consumer_shape.sample(t, False)
                demands[agent.bus.index] += agent.ratio_to_quantity(ratio, False)
            total_hourly_bids[t] += sum(list(demands.values()))
            self.dispatch.solve_model(demands, demands, t)
        return total_hourly_bids
