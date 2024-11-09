import pyomo.environ as pe
import pyomo.opt as po
from tabulate import tabulate
import numpy as np
from src.instance.constants import SOLAR, WIND, AGGREGATE, HUB


class EconomicDispatch:
    def __init__(self, network, clock, solar, wind) -> None:
        self.buses = network.buses
        self.lines = network.lines
        self.units = network.units
        self.G = network.isf
        self.model = self._construct_model()
        self.model0 = self._construct_model()
        # lmps index is (bus, hour)
        self.lmps = {
            bus.name: [0 for _ in range(clock.n_steps_one_day)] for bus in self.buses
        }
        self.lmps0 = {
            bus.name: [0 for _ in range(clock.n_steps_one_day)] for bus in self.buses
        }
        self.hub = [0 for _ in range(clock.n_steps_one_day)]
        self.hub0 = [0 for _ in range(clock.n_steps_one_day)]
        self.set_lmp_range()
        # solar and wind shapes
        self.solar = solar
        self.wind = wind
        # solar and wind unit weights (for cap frac aggregate)
        self.solar_caps = {
            u.name: u.max_prod for u in self.units if u.fuel_type == SOLAR
        }
        self.wind_caps = {u.name: u.max_prod for u in self.units if u.fuel_type == WIND}
        self.total_solar_caps = sum(self.solar_caps.values())
        self.total_wind_caps = sum(self.wind_caps.values())

    def set_lmp_range(self, lb=0, ub=999):
        self.lb_lmp = lb
        self.ub_lmp = ub

    def _construct_model(self):
        buses = self.buses
        lines = self.lines
        units = self.units
        G = self.G
        m = pe.ConcreteModel()
        # define sets
        m.N = pe.Set(initialize=[bus.index for bus in buses])
        m.L = pe.Set(initialize=[line.index for line in lines])
        m.U = pe.Set(initialize=[unit.index for unit in units])

        # define parameters
        m.demands = pe.Param(
            m.N, initialize={bus.index: 0.0 for bus in buses}, mutable=True
        )
        m.unit_max_prod = pe.Param(
            m.U, initialize={unit.index: unit.max_prod for unit in units}
        )
        m.unit_cap_frac = pe.Param(
            m.U,
            initialize={unit.index: 1.0 for unit in units},
            mutable=True,
        )
        m.line_max_flow = pe.Param(
            m.L, initialize={line.index: line.normal_flow_limit for line in lines}
        )
        m.cost_curves = pe.Param(
            m.U,
            initialize={unit.index: unit.cost_curve for unit in units},
            within=pe.Any,
        )
        m.shift_factor = pe.Param(
            m.L,
            m.N,
            initialize={
                (line.index, bus.index): G[line.index, bus.index]
                for line in lines
                for bus in buses
            },
        )
        m.penalty = pe.Param(initialize=5000)

        # define variable
        m.prod = pe.Var(m.U, domain=pe.NonNegativeReals)
        m.line_flow = pe.Var(m.L, domain=pe.NonNegativeReals)
        m.excess_prod = pe.Var(domain=pe.NonNegativeReals)
        m.unserved_load = pe.Var(domain=pe.NonNegativeReals)

        # define constraints
        m.power_balance_constraint = pe.Constraint(
            rule=lambda m: sum(m.prod[i] for i in m.U) + m.unserved_load
            == sum(m.demands[i] for i in m.N) + m.excess_prod
        )

        m.line_upper_flow_constraint = pe.Constraint(
            m.L,
            rule=lambda m, l: sum(
                m.shift_factor[l, i]
                * (sum(m.prod[u.index] for u in buses[i].units) - m.demands[i])
                for i in m.N
            )
            <= m.line_max_flow[l],
        )

        m.line_lower_flow_constraint = pe.Constraint(
            m.L,
            rule=lambda m, l: sum(
                m.shift_factor[l, i]
                * (sum(m.prod[u.index] for u in buses[i].units) - m.demands[i])
                for i in m.N
            )
            >= -m.line_max_flow[l],
        )

        m.unit_max_prod_constraint = pe.Constraint(
            m.U, rule=lambda m, i: m.prod[i] <= m.unit_max_prod[i] * m.unit_cap_frac[i]
        )

        # define objective function
        def get_unit_cost(i):
            unit_cost = 0.0
            order = len(m.cost_curves[i]) - 1
            for cost in m.cost_curves[i]:
                unit_cost += cost * (m.prod[i] ** order)
                order -= 1
            return unit_cost

        m.obj = pe.Objective(
            expr=sum(get_unit_cost(i) for i in m.U)
            + m.penalty * (m.unserved_load + m.excess_prod),
            sense=pe.minimize,
        )

        # add dual variable
        m.dual = pe.Suffix(direction=pe.Suffix.IMPORT)

        return m

    def update_prices(self, m, t, hub_list, lmps_dict):
        # record price into a dict
        price_dict = {}
        # get lmp and update the lmp list
        hub_list[t] = m.dual[m.power_balance_constraint]
        price_dict[HUB] = hub_list[t]
        mu_mins = [m.dual[m.line_lower_flow_constraint[l]] for l in m.L]
        mu_maxs = [m.dual[m.line_upper_flow_constraint[l]] for l in m.L]

        for i in m.N:
            bus = self.buses[i]
            lmps_dict[bus.name][t] = hub_list[t] - sum(
                self.G[l, i] * (mu_mins[l] - mu_maxs[l]) for l in m.L
            )
            price_dict[bus.name] = lmps_dict[bus.name][t]
            # lmps_dict[i][t] = max(lmps_dict[i][t], self.lb_lmp)
            # lmps_dict[i][t] = min(lmps_dict[i][t], self.ub_lmp)

        return price_dict, hub_list[t], mu_mins, mu_maxs

    def solve_model(self, demands, demands0, t):
        """
        demands = {bus.index: total-demand}
        returns: solar_caps, wind_caps, congestions for logger
        """
        m = self.model
        m0 = self.model0
        total_demand = sum(demands[i] for i in m.N)
        total_demand0 = sum(demands0[i] for i in m.N)

        for i in m.N:
            m.demands[i] = 0 if total_demand <= 0 else demands[i]  # this is mutable
            m0.demands[i] = 0 if total_demand0 <= 0 else demands0[i]

        # renewable capacity fractions
        solar_fracs = {AGGREGATE: 0}
        wind_fracs = {AGGREGATE: 0}
        for i in m.U:
            unit = self.units[i]
            cap_frac = 1.0
            if unit.fuel_type == "Solar":
                cap_frac = self.solar.sample(t)
                solar_fracs[unit.name] = cap_frac
                solar_fracs[AGGREGATE] += (
                    cap_frac * self.solar_caps[unit.name] / self.total_solar_caps
                )
            if unit.fuel_type == "Wind":
                cap_frac = self.wind.sample(t)
                wind_fracs[unit.name] = cap_frac
                wind_fracs[AGGREGATE] += (
                    cap_frac * self.wind_caps[unit.name] / self.total_wind_caps
                )
            m.unit_cap_frac[i] = cap_frac
            m0.unit_cap_frac[i] = cap_frac

        # solve the model
        results = po.SolverFactory("ipopt").solve(m)
        results0 = po.SolverFactory("ipopt").solve(m0)

        # update prices
        lmps, _, mu_mins, mu_maxs = self.update_prices(m, t, self.hub, self.lmps)
        lmps0, _, _, _ = self.update_prices(m0, t, self.hub0, self.lmps0)

        # congestions
        congestions = {}
        for line in self.lines:
            l = line.index
            congestions[line.name] = abs(mu_mins[l]) > 1e-3 or abs(mu_maxs[l]) > 1e-3

        if results.solver.termination_condition == po.TerminationCondition.infeasible:
            print("Entering infeasibility checking procedure...")
            print("Infeasibility check has not been implemented!")
            # check generation
            total_gen = sum(m.prod[i].value for i in m.U)
            total_load = sum(m.demands[i].value for i in m.N)
            if abs(total_gen - total_load) > 1e-3:
                print(
                    f"Unbalanced power with generation={total_gen}, load={total_load}"
                )
            for line in self.lines:
                if congestions[line.name]:
                    print(f"Line {line.name} is congested")

        return lmps, lmps0, solar_fracs, wind_fracs, congestions

    def display(self, day):
        """
        Display current prices, on `day` (int).
        If buses are supplied (list[Bus]), those LMPs will be displayed.
        """
        _round = lambda p: f"{p:.2f}"
        headers = [
            f"Day {day}",
            "HUB Learned ($/MW)",
            "HUB Unlearned ($/MW)",
            "L-UL Difference ($/MW)",
        ]
        T = len(self.hub)
        diffs = [(self.hub[i] - self.hub0[i]) for i in range(T)]
        contents = list(
            zip(
                [f"Hour {h+1}" for h in range(T)] + ["Mean", "Std"],
                [_round(p) for p in self.hub]
                + [_round(np.mean(self.hub)), _round(np.std(self.hub))],
                [_round(p) for p in self.hub0]
                + [_round(np.mean(self.hub0)), _round(np.std(self.hub0))],
                [_round(r) for r in diffs]
                + [_round(np.mean(diffs)), _round(np.std(diffs))],
            )
        )
        print(tabulate(contents, headers=headers))
