# from stable_baselines3 import PPO
from src.instance.simulation import SimulationInstance as Sim
from src.utils.logger import Logger
from src.learning.env import AggregatorEnv, EnvConfig
from src.instance.constants import *

from src.learning.ppo import build_policy, take_action
import time
import tqdm

def simulate(
    instance: Sim,
    *,
    n_days: int = 100,
    n_train: int = 2_400,
    save_as: str = "result.json",
):
    # initialize logger to save information
    logger = Logger(instance.representatives, instance.network, instance.clock)

    # reset everything in instance
    instance.reset_and_reload()
    instance.reset_lmp()
    # record in logger
    init_lmp = {**instance.dispatch.lmps, HUB: instance.dispatch.hub}
    init_lmp0 = {**instance.dispatch.lmps0, HUB: instance.dispatch.hub0}
    logger.update({LMP: init_lmp, LMP0: init_lmp0})

    # setup rl for representatives
    obs_dict = {}  # this is the actual play obs
    reps = instance.representatives
    n_steps_one_day = instance.clock.n_steps_one_day
    env_config = EnvConfig(instance)
    for a in reps:
        my_env = AggregatorEnv(config=env_config.as_dict(a.aid))
        my_env, my_policy = build_policy(my_env)
        a.setup_rl(my_env, my_policy)
        obs, _ = a.env.reset()
        # construct first obs
        curr_lmps = instance.dispatch.lmps[a.bus.name]
        obs_dict[a.aid] = {
            LMP: {h: curr_lmps[h] for h in range(n_steps_one_day)},
            HOUR_OF_DAY: 0,
            NET_DEMAND: instance.prosumer_shape.get_value(0),
            BATTERY_LEVEL: 0,
            TIME_COUNTER: 0,
            DAY: 0,
        }

    t0 = time.time()
    for k in tqdm.tqdm(range(n_days * instance.clock.n_steps_one_day)):
        day = instance.clock.get_day()
        h = instance.clock.get_time_counter_of_day()

        # first let all agents learn
        for a in reps:
            a.env.set_observation(obs_dict[a.aid])
            a.policy.learn(total_timesteps=n_train)
            a.env.set_observation(obs_dict[a.aid], update_LMP=False)

        # real play using learned policy
        ed_bids = {bus.index: 0.0 for bus in instance.network.buses}
        ed0_bids = {bus.index: 0.0 for bus in instance.network.buses}
        info_collector = logger.get_collector_dict()
        # 1. prosumers play
        for a in reps:
            # rollback to pre-learning states
            a.env.set_observation(obs_dict[a.aid], update_LMP=False)

            # take action
            action = take_action(a)
            obs, rew, _, _, _ = a.env.step(action)
            a.set_storage(obs[BATTERY_LEVEL][0])
            cts_act = get_continuous_action(action)
            nd = obs_dict[a.aid][NET_DEMAND]
            bat_lv = obs_dict[a.aid][BATTERY_LEVEL]
            my_bid0 = a.ratio_to_quantity(nd, False)
            my_bid = my_bid0 + a.ratio_to_quantity(cts_act, True)
            ed_bids[a.bus.index] += my_bid
            ed0_bids[a.bus.index] += my_bid0

            # log the following: act, bid, bat, rew, belief
            info_collector[ACTION][a.aid] = cts_act
            info_collector[ACTION][AGGREGATE] += cts_act
            info_collector[BID][a.bus.name] = my_bid
            info_collector[BID][HUB] += my_bid
            info_collector[BID0][a.bus.name] = my_bid0
            info_collector[BID0][HUB] += my_bid0
            info_collector[BATTERY_LEVEL][a.aid] = bat_lv
            info_collector[BATTERY_LEVEL][AGGREGATE] += bat_lv
            info_collector[REWARD][a.aid] = rew
            info_collector[REWARD][AGGREGATE] += rew
            info_collector[BELIEF][a.aid] = int(a.env.lmp_trace.get_value(h))
            info_collector[BELIEF][AGGREGATE] += int(a.env.lmp_trace.get_value(h))

        # 2. consumers play
        for a in instance.consumers:
            consumer_bid = a.ratio_to_quantity(instance.consumer_shape.sample(h), False)
            ed_bids[a.bus.index] += consumer_bid
            ed0_bids[a.bus.index] += consumer_bid

        # solve the new economic dispatch
        lmps, lmps0, solar_caps, wind_caps, congestions = instance.dispatch.solve_model(
            ed_bids, ed0_bids, h
        )
        info_collector[SOLAR] = solar_caps
        info_collector[WIND] = wind_caps
        info_collector[CONGESTION] = congestions
        info_collector[LMP] = lmps
        info_collector[LMP0] = lmps0
        info_collector[COST] = {
            bname: lmps[bname] * info_collector[BID][bname] for bname in lmps.keys()
        }
        info_collector[COST0] = {
            bname: lmps0[bname] * info_collector[BID0][bname] for bname in lmps0.keys()
        }
        logger.update(info_collector)

        # set the observation for each agent
        _, next_h, day = instance.clock.proceed_time()

        # update observation for the next round
        for a in reps:
            obs_dict[a.aid] = {
                LMP: {h: instance.dispatch.lmps[a.bus.name][h]},
                HOUR_OF_DAY: next_h,
                NET_DEMAND: instance.prosumer_shape.get_value(next_h),
                BATTERY_LEVEL: a.current_level,
                TIME_COUNTER: 0,
                DAY: day,
            }

    t1 = time.time()
    dt = t1 - t0
    print(f"MFG Done! Total time spent: {dt // 60}min {dt % 60:.2f}s")
    # finally save the file
    logger.save(save_as)
