import json
import numpy as np
import os, pathlib

# import all classes
from src.instance.clock import Clock
from src.instance.shape import Shape
from src.instance.network import Bus, Line, Unit, Network
from src.instance.agent import Agent
from src.instance.dispatch import EconomicDispatch
from src.instance.simulation import SimulationInstance


def load(name: str) -> SimulationInstance:
    """
    `name`: str, folder name of the testcase
    """
    testcase = pathlib.Path(os.getcwd(), name)
    instance = read(
        name,
        network_path=pathlib.Path(testcase, "network.json"),
        shape_path=pathlib.Path(testcase, "shape.json"),
        agents_path=pathlib.Path(testcase, "agents.json"),
    )
    return instance


def read(
    name: str,
    *,
    network_path,
    shape_path,
    agents_path,
) -> SimulationInstance:

    # open the file
    with open(network_path) as f_network:
        data_network = json.load(f_network)
        # read parameters
        clock = read_parameters(data_network["Parameters"])
        # read network
        network = read_network(
            data_network["Buses"],
            data_network["Transmission lines"],
            data_network["Generators"],
        )

    with open(agents_path) as f_agents:
        data_agents = json.load(f_agents)
        # read agents
        prosumers, consumers, representatives = read_agents(
            data_agents["Agents"], network
        )

    with open(shape_path) as f_shape:
        data_shape = json.load(f_shape)
        capacity = prosumers[0].reference_capacity
        # read shape files
        prosumer_shape = read_shape(
            data_shape["Net demand (prosumer)"], "prosumer", capacity
        )
        consumer_shape = read_shape(
            data_shape["Net demand (consumer)"], "consumer", capacity
        )
        solar_shape = read_shape(
            data_shape["Renewable energy (solar)"], "solar", capacity
        )
        wind_shape = read_shape(data_shape["Renewable energy (wind)"], "wind", capacity)

    # construct simulation instance
    dispatch = EconomicDispatch(network, clock, solar_shape, wind_shape)
    instance = SimulationInstance(
        name,
        clock,
        network,
        prosumer_shape,
        consumer_shape,
        prosumers,
        consumers,
        representatives,
        dispatch,
    )

    # finally, setup all RL envs & policies
    bus_rep_dict = {}
    for a in instance.representatives:
        bus_rep_dict[a.bus.name] = a
        a.storage_capacity = 0.0
        a.reference_capacity = 0.0
    # each prosumer links with a representative
    for a in instance.prosumers:
        a.link_representative(bus_rep_dict[a.bus.name])
        a.representative.storage_capacity += a.storage_capacity
        a.representative.reference_capacity += a.reference_capacity
    return instance


def read_parameters(data: dict) -> Clock:
    return Clock(
        T=data["Total time (d)"] * 24,
        time_step=data["Time step (h)"],
    )


def read_network(
    bus_dict: dict, line_dict: dict, unit_dict: dict, tol: float = 1e-6
) -> Network:
    buses = []
    lines = []
    units = []
    name_to_bus = {}

    for bus_name, bus_data in bus_dict.items():
        bus = Bus(
            bus_name,
            len(buses),  # index (0-idx)
            bus_data["P Load (kW)"],
            bus_data["Q Load (kVar)"],
            bus_data["Maximum voltage (p.u.)"] * bus_data["Base KV"],
            bus_data["Minimum voltage (p.u.)"] * bus_data["Base KV"],
            bus_data["Voltage magnitude"] * bus_data["Base KV"],
            bus_data["Voltage angle"],
            bus_data["Base KV"],
        )
        name_to_bus[bus_name] = bus
        buses.append(bus)

    for line_name, line_data in line_dict.items():
        line = Line(
            line_name,
            len(lines),  # index (0-idx)
            name_to_bus[line_data["Source bus"]],
            name_to_bus[line_data["Target bus"]],
            line_data["Reactance (ohms)"],
            line_data["Susceptance (S)"],
            line_data["Normal flow limit (MW)"],
            0.0,
        )
        lines.append(line)

    for unit_name, unit_data in unit_dict.items():
        fuel_type = unit_data["Fuel type"]
        unit = Unit(
            unit_name,
            len(units),  # index (0-idx)
            name_to_bus[unit_data["Bus"]],
            unit_data["Maximum production (MW)"],
            unit_data["Cost curve coefficients"],
            fuel_type,
        )
        units.append(unit)
        name_to_bus[unit_data["Bus"]].add_unit(unit)

    # construct and return network instance
    return Network(buses, lines, units, name_to_bus, tol)


def read_shape(data: dict, name: str, capacity: float) -> Shape:
    params = data["Parameters"]
    T = params["Total time (d)"] * 24
    time_step = params["Time step (h)"]
    data_type = params.get("Type", "Percentage")
    shape_data = np.array(data["Data"])
    if data_type == "kW":
        shape_data /= capacity
    noise_bounds = params.get("Noise bounds", (0.9, 1.1))
    # construct shape
    return Shape(
        name=name,
        raw_data=shape_data,  # dim will be adjusted upon reload
        T=T,
        time_step=time_step,
        noise_lb=noise_bounds[0],
        noise_ub=noise_bounds[1],
    )


def read_agents(
    data: dict, network: Network
) -> tuple[list[Agent], list[Agent], list[Agent]]:
    agents = []
    # construct assets for producer or prosumer
    capacity = data["Storage capacity (kW)"]
    ref_capacity = data.get("Reference capacity (kW)", capacity)

    agents = {"Prosumers": [], "Consumers": [], "Representatives": []}

    for agent_type in agents.keys():
        if data[agent_type].get("Each bus") is not None:
            n_agents = data[agent_type]["Each bus"]
            construct_items = [(bus, n_agents) for bus in network.buses]
        else:
            construct_items = [
                (network.name_to_bus[bus_name], n_agents)
                for bus_name, n_agents in data[agent_type].items()
            ]
        for bus, n_agents in construct_items:
            for _ in range(n_agents):
                agent = Agent(
                    len(agents[agent_type]),
                    bus,
                    capacity,
                    ref_capacity,
                    data["Storage efficiency"],
                    env=None,  # set up later
                    policy=None,  # set up later
                )
                agents[agent_type].append(agent)

    return agents["Prosumers"], agents["Consumers"], agents["Representatives"]
