"""Live viewer for trained rl_games policies on mjlab tasks.

Watch a checkpoint drive any registered mjlab task in real time::

    python -m rl_games.envs.mjlab_play \
        --file rl_games/configs/mjlab/ppo_go1_velocity.yaml \
        --checkpoint runs/MJLab_Go1_Velocity/nn/MJLab_Go1_Velocity.pth

Uses mjlab's own viewers (mjlab >= 1.5.3):

- ``NativeMujocoViewer``: local window; SPACE = pause, ENTER = reset
  (built-ins), plus keypad velocity-command control on velocity tasks
  (see :class:`CommandController`).
- ``ViserPlayViewer``: browser UI, works headless (prints a local URL).

ENTER reaches :meth:`PolicyAdapter.reset` (the viewer calls
``policy.reset()`` after ``env.reset()``), so RNN hidden states are zeroed
with the env. Env-internal per-env resets (a fall, MicroDuck's 20 s
truncation) and viser's per-env GUI reset hand the policy observations
only: an RNN policy carries stale hidden state across those and recovers
over a few steps.
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



# command keys live on the keypad on purpose: the native viewer binds A
# (show all envs) and RIGHT (single-step while paused) itself and forwards
# the key afterwards, and the MuJoCo window underneath toggles a render or
# visualization flag on EVERY letter (W wireframe, S shadows, D static
# bodies, ...); the keypad is free in both layers
KEY_KP_0 = _key('KEY_KP_0', 320)
KEY_KP_2 = _key('KEY_KP_2', 322)
KEY_KP_4 = _key('KEY_KP_4', 324)
KEY_KP_6 = _key('KEY_KP_6', 326)
KEY_KP_7 = _key('KEY_KP_7', 327)
KEY_KP_8 = _key('KEY_KP_8', 328)
KEY_KP_9 = _key('KEY_KP_9', 329)

# per-press command delta (vx, vy, wz) in units of
# CommandController.SPEED_STEP; None zeroes the command
COMMAND_KEYS = {
    KEY_KP_8: (1, 0, 0),    # forward
    KEY_KP_2: (-1, 0, 0),   # back
    KEY_KP_4: (0, 0, 1),    # yaw left (+wz)
    KEY_KP_6: (0, 0, -1),   # yaw right
    KEY_KP_7: (0, 1, 0),    # strafe left (+vy)
    KEY_KP_9: (0, -1, 0),   # strafe right
    KEY_KP_0: None,         # zero
}


def pick_policy_group(obs_dict):
    """Name of the policy obs group in an mjlab observation dict.

    mjlab's own tasks call it 'actor'; Isaac-Lab-style task plugins use
    'policy' (same detection as MjlabVecEnv).
    """
    return 'actor' if 'actor' in obs_dict else 'policy'


class CommandController:
    """Keyboard override for an mjlab velocity command term.

    A one-shot write to ``term.vel_command_b`` silently reverts: the term's
    ``_update_command`` rewrites heading/standing/world-frame envs, the
    resample timer overwrites the rest, and a mid-episode reset (a fall, or
    the episode timer on tasks whose play cfg keeps finite episodes)
    resamples that env's command INSIDE ``env.step`` -- after any write made
    before the step. So the override is enforced on two fronts:

    - :meth:`apply` re-asserts the command around every ``env.step`` with the
      standing/heading/world flags cleared and the resample timer pushed far
      out (mjlab's own viser joystick pins its override at the same
      per-step cadence, from the term's ``compute``);
    - the term's sampling distribution is collapsed onto the commanded
      values (degenerate ranges, special-env fractions zeroed), so a
      reset-time resample reproduces the override instead of drawing a
      random command. Re-pinned every :meth:`apply`, because curricula (the
      MicroDuck play cfg keeps its standing-envs curriculum) rewrite the
      fractions at runtime.

    :class:`PolicyAdapter` calls :meth:`apply` on every policy evaluation,
    i.e. once per viewer step.
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
        # (vx, vy, wz) body frame; an immutable tuple rebound whole, so the
        # viewer thread's key_callback and the main thread's apply() never
        # see a half-written command
        self.command = (0.0, 0.0, 0.0)
        self._saved_distribution = self._snapshot_distribution()
        # establish the (zero) override immediately: pins the distribution AND
        # overwrites the live commands drawn by the initial reset -- without
        # this the policy chases a leftover random command until the first
        # policy evaluation runs apply()
        self.apply()

    # resample-fraction knobs that inject special-cased commands on reset
    # (standing / heading / world-frame / forward-only / turn-in-place envs)
    _REL_FRACTION_ATTRS = (
        'rel_standing_envs', 'rel_heading_envs', 'rel_world_envs',
        'rel_forward_envs', 'rel_turn_in_place_envs')

    def set_velocity(self, vx, vy, wz):
        """Main-thread setter: rebind the command and re-assert it now."""
        self.command = (float(vx), float(vy), float(wz))
        self.apply()

    def _snapshot_distribution(self):
        """Record the term's original sampling distribution (for restore)."""
        cfg = getattr(self.term, 'cfg', None)
        if cfg is None:
            return None
        snap = {'ranges': {}, 'fractions': {}}
        ranges = getattr(cfg, 'ranges', None)
        if ranges is not None:
            for attr in ('lin_vel_x', 'lin_vel_y', 'ang_vel_z'):
                val = getattr(ranges, attr, None)
                if val is not None:
                    snap['ranges'][attr] = tuple(val)
        for attr in self._REL_FRACTION_ATTRS:
            val = getattr(cfg, attr, None)
            if val is not None:
                snap['fractions'][attr] = val
        return snap

    def restore_distribution(self):
        """Put the term's original sampling distribution back.

        The pinning in :meth:`apply` mutates the live term cfg. Call this
        before handing the env over to mjlab's own play UI: the viser GUI
        derives its slider bounds from ``cfg.ranges``, and the degenerate
        pinned ranges crash its construction.
        """
        snap = self._saved_distribution
        if snap is None:
            return
        cfg = self.term.cfg
        ranges = getattr(cfg, 'ranges', None)
        for attr, val in snap['ranges'].items():
            setattr(ranges, attr, val)
        for attr, val in snap['fractions'].items():
            setattr(cfg, attr, val)

    def _pin_resample_distribution(self, vx, vy, wz):
        """Collapse the term's sampling distribution onto the override.

        A mid-episode reset resamples that env's command inside ``env.step``
        -- after :meth:`apply` already ran -- so re-assertion alone leaves the
        policy acting on a random command for a step after every reset. With
        the ranges degenerate at the override and the special-env fractions
        zeroed, any resample reproduces the override instead. Config
        attributes vary by term class: guard each.
        """
        cfg = getattr(self.term, 'cfg', None)
        if cfg is None:
            return
        ranges = getattr(cfg, 'ranges', None)
        if ranges is not None:
            for attr, val in (('lin_vel_x', vx), ('lin_vel_y', vy),
                              ('ang_vel_z', wz)):
                if getattr(ranges, attr, None) is not None:
                    setattr(ranges, attr, (val, val))
            # ranges.heading stays: heading envs are disabled via the fraction
        for attr in self._REL_FRACTION_ATTRS:
            if getattr(cfg, attr, None) is not None:
                setattr(cfg, attr, 0.0)

    def apply(self):
        """Re-assert the override on ALL envs; main-loop thread, every step."""
        # one snapshot per call: key_callback rebinds self.command from the
        # viewer thread at any time, and the pinned ranges must agree with
        # the tensor written below
        vx, vy, wz = self.command
        # re-pin every call: curricula rewrite the fractions at runtime
        self._pin_resample_distribution(vx, vy, wz)
        term = self.term
        cmd = term.vel_command_b  # (num_envs, 3) = [vx, vy, wz]
        cmd[:, 0] = vx
        cmd[:, 1] = vy
        cmd[:, 2] = wz
        # world-frame envs recompute vel_command_b from vel_command_w every
        # step: keep the reference copy in sync (and clear the flag below)
        if hasattr(term, 'vel_command_w'):
            term.vel_command_w[:] = cmd
        # flag/timer attributes vary by term class -- guard each
        for flag in ('is_standing_env', 'is_heading_env', 'is_world_env',
                     'is_forward_env'):
            if hasattr(term, flag):
                getattr(term, flag)[:] = False
        if hasattr(term, 'time_left'):
            term.time_left[:] = 1e9

    def key_callback(self, keycode):
        """NativeMujocoViewer key hook, called after the viewer's own binding.

        Runs on the viewer thread while the main loop may be inside
        ``env.step``: only rebinds ``self.command``. The tensor and cfg
        writes happen in :meth:`apply`, which :class:`PolicyAdapter` runs on
        the main thread before the next step.
        """
        if keycode not in COMMAND_KEYS:
            return
        delta = COMMAND_KEYS[keycode]
        if delta is None:
            vx, vy, wz = 0.0, 0.0, 0.0
        else:
            vx, vy, wz = (c + d * self.SPEED_STEP
                          for c, d in zip(self.command, delta))
        self.command = (vx, vy, wz)
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

    def reset(self):
        """Viewer reset hook: ``BaseViewer.reset_environment`` calls it after
        ``env.reset()`` -- zero the player's RNN state with the env."""
        self.player.reset()


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
