"""Live viewer for trained rl_games policies on mjlab tasks.

Watch a checkpoint drive any registered mjlab task in real time::

    python -m rl_games.envs.mjlab_play \
        --file rl_games/configs/mjlab/ppo_go1_velocity.yaml \
        --checkpoint runs/MJLab_Go1_Velocity/nn/MJLab_Go1_Velocity.pth

Uses mjlab's own viewers (mjlab >= 1.6):

- ``NativeMujocoViewer``: local window; SPACE = pause, ENTER = reset
  (built-ins), plus keyboard velocity-command control on velocity tasks
  (see :class:`CommandController`).
- ``ViserPlayViewer``: browser UI, works headless (prints a local URL).

Known v1 limitation: the viewer's ENTER reset offers no post-reset hook, so
RNN policy hidden states are not zeroed on reset -- an RNN policy recovers
over a few steps after a reset instead of instantly.
"""

import argparse
import copy
import os

import yaml

# native-viewer key codes: prefer mjlab's canonical table, fall back to the
# GLFW values it wraps so this module imports (and is testable) without mjlab
try:
    from mjlab.viewer.native import keys as _keys
except Exception:  # mjlab absent (or viewer deps unavailable) -- fall back
    _keys = None


def _key(name, default):
    return getattr(_keys, name, default) if _keys is not None else default


KEY_UP = _key('KEY_UP', 265)
KEY_DOWN = _key('KEY_DOWN', 264)
KEY_LEFT = _key('KEY_LEFT', 263)
KEY_RIGHT = _key('KEY_RIGHT', 262)


def pick_policy_group(obs_dict):
    """Name of the policy obs group in an mjlab observation dict.

    mjlab's own tasks call it 'actor'; Isaac-Lab-style task plugins use
    'policy' (same detection as MjlabVecEnv).
    """
    return 'actor' if 'actor' in obs_dict else 'policy'


class CommandController:
    """Keyboard override for an mjlab velocity command term.

    A one-shot write to ``term.vel_command_b`` silently reverts: the term's
    ``_update_command`` rewrites heading/standing envs and the resample timer
    overwrites the rest. The pattern that sticks (mjlab's own viser joystick
    uses it) is to re-assert the command around every ``env.step`` and
    suppress the machinery that would revert it: standing/heading flags
    cleared, resample timer pushed far out. :class:`PolicyAdapter` calls
    :meth:`apply` on every policy evaluation, i.e. once per viewer step.
    """

    SPEED_STEP = 0.1

    def __init__(self, env, term_name='twist'):
        manager = getattr(env, 'command_manager', None)
        if manager is None:
            raise ValueError('env has no command_manager')
        # velocity tasks name the term 'twist'; fall back to the first term
        # that carries a body-frame velocity command
        try:
            term = manager.get_term(term_name)
        except Exception:
            term = None
        if term is None or not hasattr(term, 'vel_command_b'):
            names = list(getattr(manager, 'active_terms', []) or [])
            for name in names:
                candidate = manager.get_term(name)
                if hasattr(candidate, 'vel_command_b'):
                    term = candidate
                    break
            else:
                raise ValueError(
                    f'no velocity command term found (command terms: {names})')
        self.term = term
        self.command = [0.0, 0.0, 0.0]  # vx, vy, wz -- body frame

    def set_velocity(self, vx, vy, wz):
        self.command = [float(vx), float(vy), float(wz)]
        self.apply()

    def apply(self):
        """Re-assert the override on ALL envs; call after/around every step."""
        term = self.term
        cmd = term.vel_command_b  # (num_envs, 3) = [vx, vy, wz]
        cmd[:, 0] = self.command[0]
        cmd[:, 1] = self.command[1]
        cmd[:, 2] = self.command[2]
        # flag/timer attributes vary by term class -- guard each
        if hasattr(term, 'is_standing_env'):
            term.is_standing_env[:] = False
        if hasattr(term, 'is_heading_env'):
            term.is_heading_env[:] = False
        if hasattr(term, 'time_left'):
            term.time_left[:] = 1e9

    def key_callback(self, keycode):
        """NativeMujocoViewer key hook (runs after the built-in ENTER/SPACE)."""
        vx, vy, wz = self.command
        step = self.SPEED_STEP
        if keycode in (KEY_UP, ord('W')):
            vx += step
        elif keycode in (KEY_DOWN, ord('S')):
            vx -= step
        elif keycode in (KEY_LEFT, ord('A')):
            wz += step
        elif keycode in (KEY_RIGHT, ord('D')):
            wz -= step
        elif keycode == ord('Q'):
            vy += step
        elif keycode == ord('E'):
            vy -= step
        elif keycode == ord('X'):
            vx, vy, wz = 0.0, 0.0, 0.0
        else:
            return
        self.set_velocity(vx, vy, wz)
        print(f'command: vx={vx:+.2f} m/s  vy={vy:+.2f} m/s  wz={wz:+.2f} rad/s')


class PolicyAdapter:
    """mjlab obs-group dict -> action tensor, via a restored rl_games player.

    Reuses ``PpoPlayerContinuous.get_action`` so observation normalization
    runs inside the model's forward path exactly as during training. If a
    :class:`CommandController` is attached, its override is re-asserted on
    every call (i.e. right before the viewer steps the env), so no viewer
    internals are touched.
    """

    def __init__(self, player, group_key, deterministic=True, controller=None):
        self.player = player
        self.group_key = group_key
        self.deterministic = deterministic
        self.controller = controller

    def __call__(self, obs_dict):
        actions = self.player.get_action(
            obs_dict[self.group_key], is_deterministic=self.deterministic)
        if self.controller is not None:
            self.controller.apply()
        return actions


def _build_player(params, obs_dim, num_actions, num_envs, device):
    """Restore-ready PpoPlayerContinuous from runner params + live env dims.

    env_info is injected so BasePlayer skips env construction entirely; the
    player is batch-mode (whole vectorized obs per get_action call).
    """
    import numpy as np
    from gymnasium import spaces
    from rl_games.algos_torch.players import PpoPlayerContinuous

    params = copy.deepcopy(params)
    config = params['config']
    config['env_info'] = {
        'observation_space': spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        'action_space': spaces.Box(
            low=-1.0, high=1.0, shape=(num_actions,), dtype=np.float32),
    }
    config['device_name'] = str(device)

    player = PpoPlayerContinuous(params=params)
    player.has_batch_dimension = True
    player.batch_size = num_envs
    if player.is_rnn:
        player.init_rnn()
    return player


def run_play(yaml_config_path, checkpoint, task_id_override=None, num_envs=4,
             viewer='auto', deterministic=True, device=None):
    """Build play-mode env + restored player + mjlab viewer, then block in run().

    viewer: 'native' (local window, keyboard command control), 'viser'
    (browser UI, headless-safe), or 'auto' (native if DISPLAY/WAYLAND_DISPLAY
    is set, else viser).
    """
    with open(yaml_config_path) as f:
        params = yaml.safe_load(f)['params']
    config = params['config']
    env_config = config.get('env_config', {})
    task_id = task_id_override or env_config.get('task_name') or env_config.get('task')
    if not task_id:
        raise ValueError(
            f'no task id: pass task_id_override or set env_config.task_name '
            f'in {yaml_config_path}')
    if device is None:
        device = env_config.get('device', 'cuda')

    import warp as wp
    wp.init()
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

    # the separately-registered play cfg: infinite episodes, corruption off
    cfg = load_env_cfg(task_id, play=True)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg, device=device)

    obs_dict, _ = env.reset()
    group_key = pick_policy_group(obs_dict)
    player = _build_player(
        params,
        obs_dim=obs_dict[group_key].shape[-1],
        num_actions=env.action_space.shape[-1],
        num_envs=env.num_envs,
        device=device,
    )
    player.restore(checkpoint)

    if viewer == 'auto':
        has_display = bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
        viewer = 'native' if has_display else 'viser'

    # keyboard command control is native-only: the viser viewer ships its own
    # play UI and a competing every-step re-assert would fight it
    controller = None
    if viewer == 'native':
        try:
            controller = CommandController(env)
        except ValueError as e:
            print(f'command control disabled: {e}')

    policy = PolicyAdapter(
        player, group_key, deterministic=deterministic, controller=controller)

    if viewer == 'native':
        from mjlab.viewer import NativeMujocoViewer
        key_cb = controller.key_callback if controller is not None else None
        ui = NativeMujocoViewer(env, policy, key_callback=key_cb)
    elif viewer == 'viser':
        from mjlab.viewer import ViserPlayViewer
        ui = ViserPlayViewer(env, policy)
    else:
        raise ValueError(f"unknown viewer '{viewer}' (expected auto|native|viser)")

    try:
        ui.run()  # blocks; ENTER = reset, SPACE = pause (viewer built-ins)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(
        description='Watch a trained rl_games policy live on an mjlab task')
    p.add_argument('--file', required=True, help='training yaml config')
    p.add_argument('--checkpoint', required=True, help='.pth checkpoint to restore')
    p.add_argument('--task', default=None,
                   help="task id override (default: the config's env_config.task_name)")
    p.add_argument('--num-envs', type=int, default=4)
    p.add_argument('--viewer', choices=['auto', 'native', 'viser'], default='auto')
    p.add_argument('--stochastic', action='store_true',
                   help='sample actions instead of taking the deterministic mean')
    p.add_argument('--device', default=None,
                   help="sim/policy device (default: the config's env_config.device)")
    args = p.parse_args()
    run_play(args.file, args.checkpoint, task_id_override=args.task,
             num_envs=args.num_envs, viewer=args.viewer,
             deterministic=not args.stochastic, device=args.device)


if __name__ == '__main__':
    main()
