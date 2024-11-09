class Agent:
    def __init__(
        self,
        index,
        bus,
        capacity,
        ref_capacity,
        storage_efficiency,
        env,
        policy,
        tol=1e-6,
    ) -> None:
        self.index = index
        self.aid = f"agent_{index}"  # agent id, for rllib multi agent env
        self.bus = bus
        self.current_net_demand = 0.0

        self.storage_capacity = capacity  # max energy
        self.reference_capacity = ref_capacity
        self.storage_efficiency = storage_efficiency  # i/o efficiency
        self.current_level = 0.0  # ratio from 0 to 1

        # env & policy are for representatives only
        # each repre maintains its own env & policy (Q)
        self.env = env
        self.policy = policy

        self.representative = None # link later

        self.tol = tol

    def link_representative(self, rep):
        """
        For non-representatives, we need to link them to the rep
        """
        self.representative = rep 

    def setup_rl(self, env, policy):
        """
        For representatives, we need to provide env and policy (PPO)
        """
        self.env = env
        self.policy = policy

    def update_storage(self, ratio):
        """
        Given the agent is charging `ratio` amount:
            eg. 0.25 = charge 25% of the battery
            eg. -0.1 = discharge 10% of the battery
        update the storage and clip it to [0, 1]
        """
        new_ratio = self.current_level + ratio
        self.set_storage(new_ratio)

    def set_storage(self, ratio):
        """
        Hard set the storage level 
        """
        self.current_level = max(min(ratio, 1.0), 0.0)

    def ratio_to_quantity(self, ratio, use_true_storage):
        """
        Given the agent is charging `ratio` amount,
        convert it to actual quantity in MWh (/1000)
        """
        eta = self.storage_efficiency
        cap = self.storage_capacity if use_true_storage else self.reference_capacity
        q = ratio * cap
        if ratio < 0:
            return eta * q / 1000  # discharging #FIXME UNIT
        return q / eta / 1000 # charging #FIXME UNIT

    def get_net_demand(self, shape, clock, randomize=True):
        t = clock.time_counter
        self.current_net_demand = shape.sample(t, randomize)
        return self.current_net_demand
