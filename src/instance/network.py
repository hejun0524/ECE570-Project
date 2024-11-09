import numpy as np
import numpy.linalg as LA


class Bus:
    def __init__(
        self, name, index, load, q_load, v_max, v_min, v_mag, v_ang, basekv
    ) -> None:
        self.name = name
        self.index = index
        self.load = load
        self.q_load = q_load
        self.v_max = v_max
        self.v_min = v_min
        self.v_mag = v_mag
        self.v_ang = v_ang
        self.initial_v_mag = v_mag
        self.initial_v_ang = v_ang
        self.basekv = basekv
        self.units = []

    def reset(self):
        # reset bus voltage
        self.v_mag = self.initial_v_mag
        self.v_ang = self.initial_v_ang

    def add_unit(self, unit):
        self.units.append(unit)


class Line:
    def __init__(
        self,
        name,
        index,
        source,
        target,
        reactance,
        susceptance,
        normal_flow_limit,
        current_flow,
    ) -> None:
        self.name = name
        self.index = index
        self.source = source
        self.target = target
        self.reactance = reactance
        self.susceptance = susceptance
        self.normal_flow_limit = normal_flow_limit * 2
        self.current_flow = current_flow

    def reset(self):
        # reset current line flow
        self.current_flow = 0.0


class Unit:
    def __init__(self, name, index, bus, max_prod, cost_curve, fuel_type) -> None:
        self.name = name
        self.index = index
        self.bus = bus
        self.max_prod = max_prod
        self.cost_curve = cost_curve
        self.fuel_type = fuel_type


class Network:
    def __init__(self, buses, lines, units, name_to_bus, tol=1e-6) -> None:
        self.buses = buses
        self.lines = lines
        self.units = units
        self.name_to_bus = name_to_bus
        self.tol = tol
        self.isf = self.injection_shift_factors()

    def injection_shift_factors(self):
        # L*L diagonal matrix for susceptance
        B = np.diag([l.susceptance for l in self.lines])
        # reduced incidence matrix
        A = np.zeros((len(self.lines), len(self.buses)))
        for l in self.lines:
            if l.source.index > 0:
                A[l.index, l.source.index] = 1
            if l.target.index > 0:
                A[l.index, l.target.index] = -1
        # drop the first column (slack bus)
        A = A[:, 1:]
        # laplacian matrix
        L = A.T @ B @ A
        # get ISF
        isf = B @ A @ LA.inv(L)
        isf[abs(isf) < self.tol] = 0.0
        # append the first column (all zeros, slack bus)
        isf = np.hstack((np.zeros((len(self.lines), 1)), isf))
        return isf

    def reset(self):
        for bus in self.buses:
            bus.reset()
        for line in self.lines:
            line.reset()
