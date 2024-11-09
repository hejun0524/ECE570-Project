# from stable_baselines3 import PPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from src.instance.constants import *
import numpy as np


def build_policy(my_env):
    if not isinstance(my_env, ActionMasker):
        my_env = ActionMasker(my_env, lambda env: env.get_action_masks())
    my_policy = MaskablePPO(
        "MultiInputPolicy",
        my_env,
        learning_rate=0.0003,
        gamma=0.9999,
        ent_coef=0.01,
        batch_size=128,
        n_epochs=10,
        n_steps=1200,
        clip_range=0.2,
        policy_kwargs={"net_arch": dict(pi=[12, 24], vf=[12, 24])},
        device="cpu"
    )
    return my_env, my_policy


def take_action(a):
    inner_policy: MaskableMultiInputActorCriticPolicy = a.policy.policy
    tensor_obs, _ = inner_policy.obs_to_tensor(
        a.env.get_observation())
    act_dist = inner_policy.get_distribution(
        tensor_obs,
        action_masks=a.env.get_action_masks()
    )  # action masks assigns 0 prob to invalid acts
    # act probs is in shape of [[.1, .2, ...]]
    act_probs = act_dist.distribution.probs.detach().numpy()
    # must add [0] to get the scalar out (its a 1x1 matrix)
    action = np.inner(act_probs, np.array(
        [i for i in range(a.env.action_space.n)]))[0]
    return action