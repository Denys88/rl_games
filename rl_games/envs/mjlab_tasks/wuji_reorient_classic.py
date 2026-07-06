"""Classic in-hand reorientation formulation on the wuji-mjlab WujiHand.

The IsaacGymEnvs/DexLab reward design — reciprocal rotation kernel, large
one-shot success bonus, cube-position penalty, fall reset — on wuji-mjlab's
hand, scene, uniform-SO(3) switch-on-success goal command and domain
randomization. Registered as 'WujiHand_Reorient_Classic' on import
(requires wuji-mjlab installed).

mjlab scales reward weights by step_dt; the weights below are the classic
per-step scales divided by step_dt (0.05 s) so per-step magnitudes match
the IsaacGymEnvs originals.
"""

import torch

from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task

_ROT_EPS = 0.1
_FALL_DIST = 0.24


def _ori_error(env, command_name):
    return env.command_manager.get_term(command_name).metrics["ori_error"]


def rot_reciprocal(env, command_name: str):
    """Classic 1/(|rot_dist| + eps) rotation reward."""
    return 1.0 / (torch.abs(_ori_error(env, command_name)) + _ROT_EPS)


def goal_reached_bonus(env, command_name: str):
    """One-shot bonus when the goal switches after a success."""
    term = env.command_manager.get_term(command_name)
    return term.goal_switched.float()


def _palm_object_dist(env, asset_cfg: SceneEntityCfg):
    palm = env.scene[asset_cfg.name].data.body_link_pos_w[:, asset_cfg.body_ids[0]]
    obj = env.scene["object"].data.root_link_pos_w
    return torch.linalg.norm(obj - palm, dim=-1)


def object_palm_distance(env, asset_cfg: SceneEntityCfg):
    """Classic distance penalty: cube drifting from the hold point."""
    return _palm_object_dist(env, asset_cfg)


def action_l2(env):
    return torch.sum(env.action_manager.action ** 2, dim=-1)


def object_out_of_reach(env, asset_cfg: SceneEntityCfg):
    """Classic fall condition: cube left the hand's neighborhood."""
    return _palm_object_dist(env, asset_cfg) >= _FALL_DIST


def _classic_cfg(play: bool = False, num_envs: int = 8192):
    from wuji_mjlab.tasks.reorient.config.wuji_hand.env_cfgs import wuji_hand_reorient_env_cfg

    cfg = wuji_hand_reorient_env_cfg(play=play, num_envs=num_envs)
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation  # 0.05

    cmd = cfg.commands["reorient_command"]
    cmd.success_threshold = 0.2
    cmd.success_hold_steps = 1
    cmd.goal_switch_delay = 0

    palm = SceneEntityCfg("robot", body_names=("right_palm_link",))
    cfg.rewards = {
        "rot_reward": RewardTermCfg(
            func=rot_reciprocal, weight=1.0 / step_dt,
            params={"command_name": "reorient_command"}),
        "reach_goal_bonus": RewardTermCfg(
            func=goal_reached_bonus, weight=250.0 / step_dt,
            params={"command_name": "reorient_command"}),
        "dist_penalty": RewardTermCfg(
            func=object_palm_distance, weight=-10.0 / step_dt,
            params={"asset_cfg": palm}),
        "action_penalty": RewardTermCfg(
            func=action_l2, weight=-0.0002 / step_dt, params={}),
    }
    cfg.terminations = {
        "time_out": cfg.terminations["time_out"],
        "out_of_reach": TerminationTermCfg(
            func=object_out_of_reach, params={"asset_cfg": palm}),
    }
    # classic formulation: no curriculum; keep the DR events as-is (harder
    # setting than the DexLab PhysX runs, matching the reference policy's DR)
    cfg.curriculum = {}
    return cfg


def _register():
    from wuji_mjlab.tasks.reorient.config.wuji_hand.rsl_rl.ppo import wuji_hand_reorient_ppo_runner_cfg

    register_mjlab_task(
        "WujiHand_Reorient_Classic",
        _classic_cfg(play=False),
        _classic_cfg(play=True),
        wuji_hand_reorient_ppo_runner_cfg(),
    )


_register()
