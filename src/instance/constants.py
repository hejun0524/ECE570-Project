# Logger constants
PROSUMER = "Prosumer"
CONSUMER = "Consumer"
REPRESENTATIVE = "Representative"
ACTION = "Action"
LMP = "LMP"
LMP0 = "LMP without battery"
NET_DEMAND = "Net demand"
HOUR_OF_DAY = "Hour of the day"
BATTERY_LEVEL = "Battery level"
DAY = "Day"
TIME_COUNTER = "Time counter"
TIME_COUNTER_START = "Start time counter"
CURRENT_LMP = "Current LMP"
CURRENT_BELIEF = "Current belief"
BID = "Bid"
BID0 = "Bid without battery"
REWARD = "Reward"
REWARD0 = "Reward without battery"
COST = "Cost"
COST0 = "Cost without battery"
HUB = "Hub"
AGGREGATE = "Aggregate"
BELIEF = "Belief"
IMV = "IMV"
IMV0 = "IMV without battery"

# Parameter constants
N_STEPS_ONE_DAY = "Number of steps in one day"
N_AGENTS = "Number of agents"
EFFICIENCY = "Efficiency"
REF_CAPACITY = "Reference battery capacity"
CAPACITY = "Battery capacity"

# Dispatch constants
RENEWABLE = "Renewable"
SOLAR = "Solar"
WIND = "Wind"
CONGESTION = "Congestion"

# Planner constants
TOTAL_CONSUMER_BIDS = "Total consumer bids"
TOTAL_PROSUMER_BIDS = "Total prosumer bids"
TAX_RATE = "Tax rate"
EEI_PROSUMER = "Prosumer EEI"
EEI_CONSMER = "Consumer EEI"

# Numerical constants
STEP_ACTION = 1 # X% each step
N_ACTIONS = 200 // STEP_ACTION + 1
N_ACTIONS_HALF = N_ACTIONS // 2

STEP_ACTION_PLANNER = 1 # X% each step
N_ACTIONS_PLANNER = 100 // STEP_ACTION + 2 # Last is NOOP
N_ACTIONS_HALF_PLANNER = N_ACTIONS_PLANNER // 2

def get_continuous_action(discrete):
    return (discrete - N_ACTIONS_HALF) / N_ACTIONS_HALF