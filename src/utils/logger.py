import json
from src.instance.constants import *


class Logger:

    def __init__(self, agents, network, clock) -> None:
        self.agents = [a.aid for a in agents]
        self.buses = [b.name for b in network.buses]
        self.lines = [l.name for l in network.lines]
        self.wind_units = [u.name for u in network.units if u.fuel_type == WIND]
        self.solar_units = [u.name for u in network.units if u.fuel_type == SOLAR]
        self.agent_to_bus = {a.aid: a.bus.name for a in agents}
        self.n_steps_one_day = clock.n_steps_one_day
        self.imv_T = self.n_steps_one_day * 3
        # build lists of the logger
        self.bats = self._build_dict(self.agents, AGGREGATE)
        self.rews = self._build_dict(self.agents, AGGREGATE)
        self.acts = self._build_dict(self.agents, AGGREGATE)
        self.bids = self._build_dict(self.buses, HUB)
        self.bids0 = self._build_dict(self.buses, HUB)
        self.lmps = self._build_dict(self.buses, HUB)  # not 1step info
        self.lmps0 = self._build_dict(self.buses, HUB)  # not 1step info
        self.beliefs = self._build_dict(self.agents, AGGREGATE)
        self.wind = self._build_dict(self.wind_units, AGGREGATE)
        self.solar = self._build_dict(self.solar_units, AGGREGATE)
        self.congestion = self._build_dict(self.lines, AGGREGATE)
        self.costs = self._build_dict(self.buses, HUB) # ex-post
        self.costs0 = self._build_dict(self.buses, HUB) # ex-post
        self.rates = []
        self.eei_prosumers = self._build_dict(self.agents, AGGREGATE)
        self.eei_consumers = self._build_dict(self.agents, AGGREGATE)
        

    def _build_dict(self, key_list, key_aggregate, use_list=True):
        d = {k: [] if use_list else 0 for k in key_list}
        d[key_aggregate] = [] if use_list else 0
        return d

    def _build_scalar_dict(self, key_list, key_aggregate):
        return self._build_dict(key_list, key_aggregate, use_list=False)

    def get_collector_dict(self):
        # set use_list to False to collect one step info
        return {
            BATTERY_LEVEL: self._build_scalar_dict(self.agents, AGGREGATE),
            REWARD: self._build_scalar_dict(self.agents, AGGREGATE),
            ACTION: self._build_scalar_dict(self.agents, AGGREGATE),
            BID: self._build_scalar_dict(self.buses, HUB),
            BID0: self._build_scalar_dict(self.buses, HUB),
            COST: self._build_scalar_dict(self.buses, HUB),
            COST0: self._build_scalar_dict(self.buses, HUB),
            LMP: self._build_scalar_dict(self.buses, HUB),
            LMP0: self._build_scalar_dict(self.buses, HUB),
            BELIEF: self._build_scalar_dict(self.agents, AGGREGATE),
            TAX_RATE: None,
            EEI_PROSUMER: None,
            EEI_CONSMER: None,
            WIND: None,
            SOLAR: None,
            CONGESTION: None,
        }

    def update(self, data: dict):
        key_to_lists = {
            BATTERY_LEVEL: self.bats,
            REWARD: self.rews,
            ACTION: self.acts,
            BID: self.bids,
            BID0: self.bids0,
            LMP: self.lmps,
            LMP0: self.lmps0,
            BELIEF: self.beliefs,
            WIND: self.wind,
            SOLAR: self.solar,
            CONGESTION: self.congestion,
            COST: self.costs,
            COST0: self.costs0,
            TAX_RATE: self.rates,
            EEI_PROSUMER: self.eei_prosumers,
            EEI_CONSMER: self.eei_consumers,
        }
        # make sure the input is correct
        for key in data.keys():
            assert key in key_to_lists.keys()
        # iteratively add in the value
        for key, data_collection in data.items():
            # if my information is stored in a list
            if data_collection is None:
                continue
            if isinstance(data_collection, list):
                if isinstance(v, list):
                    key_to_lists[key].extend(v)
                else:
                    key_to_lists[key].append(v)
            # if my information is stored in a dict
            else:
                for id, v in data_collection.items():
                    if isinstance(v, list):
                        key_to_lists[key][id].extend(v)
                    else:
                        key_to_lists[key][id].append(v)

    def _compute_imvs(self, lmp_dict: dict):
        T = self.imv_T
        imv_dict = {}
        for bname, lmp in lmp_dict.items():
            lmps_in_T = [lmp[i:i+T] for i in range(len(lmp) - T + 1)]
            imvs = []
            for ps in lmps_in_T:
                imvs.append(sum(abs(ps[i+1] - ps[i])/T for i in range(T-1)))
            imv_dict[bname] = imvs
        return imv_dict

    def save(self, fname: str):
        """
        fname: .json file name, eg: 'result.json'
        """
        with open(fname, "w") as f:
            json.dump(
                {
                    N_AGENTS: len(self.agents),
                    N_STEPS_ONE_DAY: self.n_steps_one_day,
                    BATTERY_LEVEL: self.bats,
                    BID: self.bids,
                    BID0: self.bids0,
                    REWARD: self.rews,
                    LMP: self.lmps,
                    ACTION: self.acts,
                    LMP0: self.lmps0,
                    BELIEF: self.beliefs,
                    WIND: self.wind,
                    SOLAR: self.solar,
                    CONGESTION: self.congestion,
                    COST: self.costs,
                    COST0: self.costs0,
                    IMV: self._compute_imvs(self.lmps),
                    IMV0: self._compute_imvs(self.lmps0),
                    TAX_RATE: self.rates,
                    EEI_PROSUMER: self.eei_prosumers,
                    EEI_CONSMER: self.eei_consumers,
                },
                f,
            )

    def display_collection(self, h):
        for aid in self.agents:
            print(
                f"AG {aid} | ",
                f"HR {h+1} |",
                f"ACT {round(100*self.acts[aid][-1])}% |",
                f"BAT {round(100*self.bats[aid][-1])}% |",
                f"REW ${self.rews[aid][-1]:.2f}",
            )

    def display_net_info(self, h):
        """
        h: current hour
        """
        n = len(self.agents)
        agg_acts = self.acts[AGGREGATE][-1]
        agg_bats = self.bats[AGGREGATE][-1]
        agg_rews = self.rews[AGGREGATE][-1]
        agg_blfs = self.beliefs[AGGREGATE][-1]
        print(
            f"HR {h+1} |",
            f"LMP ${int(agg_blfs/n)} |",
            f"ACT {round(100*agg_acts/n)}% |",
            f"BAT {round(100*agg_bats/n)}% |",
            f"REW ${agg_rews/n:.3f}",
        )

    def display_beliefs(self, agents):
        for a in agents:
            beliefs = [str(int(blf)) for blf in a.env.lmp_trace.get_values()]
            print(f"{a.aid} Day {a.env.day}: " + ", ".join(beliefs))
