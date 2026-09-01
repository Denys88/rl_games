"""CI-safe tests for the mjlab live-play module -- no mjlab required.

rl_games.envs.mjlab_play keeps all mjlab imports function-local or guarded,
so its pure logic (obs-group pick, command re-assert pattern, policy adapter)
is testable against stubs whether or not mjlab is installed. The two tests
that read the installed viewer (reserved keys, reset hook) importorskip it.
"""

import inspect
import os
import re
import subprocess
import sys
import types

import pytest
import torch
import yaml

from rl_games.envs import mjlab_play
from rl_games.envs.mjlab_play import (CommandController, PolicyAdapter,
                                      pick_policy_group)


# ---------------------------------------------------------------- stubs

class FakeVelocityTerm:
    """Mimics an mjlab velocity command term with all revert machinery armed."""

    def __init__(self, num_envs=4):
        self.vel_command_b = torch.zeros(num_envs, 3)
        self.vel_command_w = torch.zeros(num_envs, 3)
        self.is_standing_env = torch.ones(num_envs, dtype=torch.bool)
        self.is_heading_env = torch.ones(num_envs, dtype=torch.bool)
        self.is_world_env = torch.ones(num_envs, dtype=torch.bool)
        self.is_forward_env = torch.ones(num_envs, dtype=torch.bool)
        self.time_left = torch.full((num_envs,), 0.5)
        self.cfg = types.SimpleNamespace(
            ranges=types.SimpleNamespace(
                lin_vel_x=(-0.4, 0.4), lin_vel_y=(-0.3, 0.3),
                ang_vel_z=(-1.0, 1.0), heading=(-3.14, 3.14)),
            rel_standing_envs=0.02, rel_heading_envs=1.0,
            rel_world_envs=0.1, rel_forward_envs=0.15,
            rel_turn_in_place_envs=0.15)


class FakePoseTerm:
    """A command term without vel_command_b (e.g. a pose target)."""


class FakeCommandManager:
    def __init__(self, terms):
        self._terms = dict(terms)

    @property
    def active_terms(self):
        return list(self._terms)

    def get_term(self, name):
        return self._terms[name]


def make_env(terms):
    return types.SimpleNamespace(command_manager=FakeCommandManager(terms))


class FakePlayer:
    def __init__(self):
        self.calls = []
        self.resets = 0

    def get_action(self, obs, is_deterministic=False):
        self.calls.append((obs, is_deterministic))
        return obs * 2.0

    def reset(self):
        self.resets += 1


class FakeController:
    def __init__(self):
        self.applies = 0

    def apply(self):
        self.applies += 1


# ------------------------------------------------- obs group detection

def test_pick_policy_group_actor():
    assert pick_policy_group({'actor': 1, 'critic': 2}) == 'actor'


def test_pick_policy_group_isaac_lab_style():
    assert pick_policy_group({'policy': 1, 'critic': 2}) == 'policy'


def test_pick_policy_group_prefers_actor():
    # mjlab-native naming wins when both are present
    assert pick_policy_group({'actor': 1, 'policy': 2}) == 'actor'


# ---------------------------------------------------- CommandController

def test_controller_finds_twist_term():
    twist = FakeVelocityTerm()
    ctrl = CommandController(make_env({'twist': twist}))
    assert ctrl.term is twist


def test_controller_falls_back_to_first_velocity_term():
    vel = FakeVelocityTerm()
    ctrl = CommandController(make_env({'pose': FakePoseTerm(), 'base_vel': vel}))
    assert ctrl.term is vel


def test_controller_raises_without_velocity_term():
    with pytest.raises(ValueError, match='no velocity command term'):
        CommandController(make_env({'pose': FakePoseTerm()}))


def test_controller_raises_without_command_manager():
    with pytest.raises(ValueError, match='command_manager'):
        CommandController(types.SimpleNamespace())


def test_set_velocity_writes_all_envs_and_disarms_revert_machinery():
    term = FakeVelocityTerm(num_envs=4)
    ctrl = CommandController(make_env({'twist': term}))
    ctrl.set_velocity(0.5, -0.1, 0.3)

    expected = torch.tensor([0.5, -0.1, 0.3]).expand(4, 3)
    assert torch.allclose(term.vel_command_b, expected)
    # standing/heading rewrites and the resample timer must be suppressed,
    # otherwise the term's _update_command silently reverts the override
    assert not term.is_standing_env.any()
    assert not term.is_heading_env.any()
    assert (term.time_left > 1e8).all()


def test_apply_reasserts_after_env_side_overwrite():
    # the core of the pattern: mjlab overwrites between steps; apply() every
    # step restores the override
    term = FakeVelocityTerm(num_envs=2)
    ctrl = CommandController(make_env({'twist': term}))
    ctrl.set_velocity(1.0, 0.0, 0.0)

    # simulate the env's _update_command + resample doing their worst
    term.vel_command_b.zero_()
    term.is_standing_env[:] = True
    term.is_heading_env[:] = True
    term.time_left[:] = 0.01

    ctrl.apply()
    assert torch.allclose(term.vel_command_b,
                          torch.tensor([1.0, 0.0, 0.0]).expand(2, 3))
    assert not term.is_standing_env.any()
    assert not term.is_heading_env.any()
    assert (term.time_left > 1e8).all()


def test_apply_guards_missing_flag_attributes():
    # flag names vary by term class -- a bare term with only vel_command_b
    # must still work
    term = types.SimpleNamespace(vel_command_b=torch.zeros(3, 3))
    ctrl = CommandController(make_env({'twist': term}))
    ctrl.set_velocity(0.2, 0.0, -0.4)
    assert torch.allclose(term.vel_command_b,
                          torch.tensor([0.2, 0.0, -0.4]).expand(3, 3))


def test_set_velocity_pins_resample_distribution():
    # a mid-episode reset resamples INSIDE env.step, after apply() ran: the
    # override only survives resets if the distribution itself is pinned
    term = FakeVelocityTerm()
    ctrl = CommandController(make_env({'twist': term}))
    ctrl.set_velocity(0.3, -0.1, 0.2)

    assert term.cfg.ranges.lin_vel_x == (0.3, 0.3)
    assert term.cfg.ranges.lin_vel_y == (-0.1, -0.1)
    assert term.cfg.ranges.ang_vel_z == (0.2, 0.2)
    for attr in ('rel_standing_envs', 'rel_heading_envs', 'rel_world_envs',
                 'rel_forward_envs', 'rel_turn_in_place_envs'):
        assert getattr(term.cfg, attr) == 0.0
    # heading range untouched: heading envs are disabled via the fraction
    assert term.cfg.ranges.heading == (-3.14, 3.14)


def test_constructor_pins_distribution_to_zero_command():
    # before any key press the robot should hold still -- resets must not
    # resample a random command underneath the zero override
    term = FakeVelocityTerm()
    CommandController(make_env({'twist': term}))
    assert term.cfg.ranges.lin_vel_x == (0.0, 0.0)
    assert term.cfg.rel_turn_in_place_envs == 0.0


def test_constructor_applies_zero_override_immediately():
    # the initial env.reset() runs before the controller exists: its random
    # command draws must be overwritten at attach time, not at the first
    # policy evaluation one viewer frame later
    term = FakeVelocityTerm()
    term.vel_command_b[:] = torch.tensor([0.3, -0.2, 0.5])
    CommandController(make_env({'twist': term}))
    assert torch.equal(term.vel_command_b, torch.zeros(4, 3))
    assert not term.is_standing_env.any()
    assert not term.is_heading_env.any()
    assert (term.time_left > 1e8).all()


def test_apply_repins_after_curriculum_rewrite():
    # the MicroDuck play cfg keeps its standing-envs curriculum active, which
    # rewrites rel_standing_envs at runtime -- apply() must re-pin
    term = FakeVelocityTerm()
    ctrl = CommandController(make_env({'twist': term}))
    ctrl.set_velocity(0.3, 0.0, 0.0)

    term.cfg.rel_standing_envs = 0.05
    term.cfg.ranges.lin_vel_x = (-0.4, 0.4)
    ctrl.apply()
    assert term.cfg.rel_standing_envs == 0.0
    assert term.cfg.ranges.lin_vel_x == (0.3, 0.3)


def test_apply_clears_world_envs_and_syncs_world_reference():
    # world-frame envs recompute vel_command_b from vel_command_w every step:
    # the flag must be cleared and the reference copy kept in sync
    term = FakeVelocityTerm()
    ctrl = CommandController(make_env({'twist': term}))
    ctrl.set_velocity(0.4, 0.0, 0.0)

    assert not term.is_world_env.any()
    assert not term.is_forward_env.any()
    assert torch.allclose(term.vel_command_w,
                          torch.tensor([0.4, 0.0, 0.0]).expand(4, 3))


def test_restore_distribution_returns_original_sampling():
    # mjlab's viser GUI builds sliders from cfg.ranges -- a detached
    # controller must be able to hand back the env with the original
    # distribution intact
    term = FakeVelocityTerm()
    ctrl = CommandController(make_env({'twist': term}))
    ctrl.set_velocity(0.3, 0.0, 0.0)

    ctrl.restore_distribution()
    assert term.cfg.ranges.lin_vel_x == (-0.4, 0.4)
    assert term.cfg.ranges.lin_vel_y == (-0.3, 0.3)
    assert term.cfg.ranges.ang_vel_z == (-1.0, 1.0)
    assert term.cfg.rel_standing_envs == 0.02
    assert term.cfg.rel_heading_envs == 1.0
    assert term.cfg.rel_world_envs == 0.1
    assert term.cfg.rel_forward_envs == 0.15
    assert term.cfg.rel_turn_in_place_envs == 0.15


def test_restore_distribution_guards_missing_cfg():
    term = types.SimpleNamespace(vel_command_b=torch.zeros(2, 3))
    ctrl = CommandController(make_env({'twist': term}))
    ctrl.restore_distribution()  # must not raise


def test_key_callback_steps_command_without_touching_env_state():
    # the hook runs on the viewer thread while the main loop may be inside
    # env.step: it may only rebind the command; apply() (main thread, from
    # PolicyAdapter) is what writes the tensor and the pinned ranges
    term = FakeVelocityTerm()
    ctrl = CommandController(make_env({'twist': term}))
    ranges = term.cfg.ranges
    pinned_zero = (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z)

    for _ in range(3):
        ctrl.key_callback(mjlab_play.KEY_KP_8)   # forward
    assert ctrl.command == pytest.approx((0.3, 0.0, 0.0))

    ctrl.key_callback(mjlab_play.KEY_KP_2)       # back
    ctrl.key_callback(mjlab_play.KEY_KP_4)       # yaw left
    ctrl.key_callback(mjlab_play.KEY_KP_6)       # yaw right
    ctrl.key_callback(mjlab_play.KEY_KP_7)       # strafe left
    assert ctrl.command == pytest.approx((0.2, 0.1, 0.0))
    assert isinstance(ctrl.command, tuple)
    # nothing env-side moved yet
    assert torch.equal(term.vel_command_b, torch.zeros(4, 3))
    assert (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z) == pinned_zero

    ctrl.apply()
    assert torch.allclose(term.vel_command_b,
                          torch.tensor([0.2, 0.1, 0.0]).expand(4, 3))
    assert ranges.lin_vel_x == pytest.approx((0.2, 0.2))
    assert ranges.lin_vel_y == pytest.approx((0.1, 0.1))
    assert ranges.ang_vel_z == pytest.approx((0.0, 0.0))

    ctrl.key_callback(mjlab_play.KEY_KP_0)       # zero
    assert ctrl.command == (0.0, 0.0, 0.0)


def test_key_callback_ignores_unknown_keys():
    # letters are the MuJoCo window's render-flag toggles (W wireframe, ...)
    # and must not double as command keys
    term = FakeVelocityTerm()
    ctrl = CommandController(make_env({'twist': term}))
    ctrl.set_velocity(0.5, 0.0, 0.0)
    before = term.vel_command_b.clone()
    for key in (ord('W'), ord('A'), ord('X'), ord('Z')):
        ctrl.key_callback(key)
    assert ctrl.command == pytest.approx((0.5, 0.0, 0.0))
    assert torch.equal(term.vel_command_b, before)


def test_apply_snapshots_command_once():
    # a viewer-thread rebind landing between the range pinning and the
    # tensor write must not tear: apply() reads self.command exactly once
    term = FakeVelocityTerm()
    ctrl = CommandController(make_env({'twist': term}))
    ctrl.command = (0.5, 0.0, 0.0)
    original_pin = ctrl._pin_resample_distribution

    def racing_pin(vx, vy, wz):
        original_pin(vx, vy, wz)
        ctrl.command = (0.9, 0.0, 0.0)  # key press lands mid-apply

    ctrl._pin_resample_distribution = racing_pin
    ctrl.apply()
    assert term.cfg.ranges.lin_vel_x == (0.5, 0.5)
    assert torch.allclose(term.vel_command_b,
                          torch.tensor([0.5, 0.0, 0.0]).expand(4, 3))


# ------------------------------------------------------ reserved keys

def _mjlab_reserved_keys():
    """Key codes the native viewer binds itself, read from its handler."""
    viewer_mod = pytest.importorskip('mjlab.viewer.native.viewer')
    from mjlab.viewer.native import keys
    src = inspect.getsource(viewer_mod.NativeMujocoViewer._safe_key_callback)
    names = set(re.findall(r'\bKEY_[A-Z0-9_]+\b', src))
    assert names, 'no KEY_ constants found in _safe_key_callback'
    return {getattr(keys, name) for name in names}


def test_command_keys_disjoint_from_mjlab_builtins():
    # the viewer runs its own binding and THEN forwards the key: a shared
    # key would also pause / step / toggle show-all-envs on every press
    reserved = _mjlab_reserved_keys()
    assert reserved.isdisjoint(mjlab_play.COMMAND_KEYS), reserved
    from mjlab.viewer.native import keys
    for name in ('KEY_KP_0', 'KEY_KP_2', 'KEY_KP_4', 'KEY_KP_6',
                 'KEY_KP_7', 'KEY_KP_8', 'KEY_KP_9'):
        assert getattr(mjlab_play, name) == getattr(keys, name)


# MuJoCo 3.10.0 mjVISSTRING / mjRNDSTRING shortcuts: the passive viewer
# toggles a visualization or render flag on every one of these -- all 26
# letters plus , ' ; ` \ /
MUJOCO_SHORTCUT_KEYS = (set(map(ord, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'))
                        | set(map(ord, ",';`\\/")))


def test_command_keys_disjoint_from_mujoco_shortcuts():
    assert MUJOCO_SHORTCUT_KEYS.isdisjoint(mjlab_play.COMMAND_KEYS)
    try:
        import mujoco
    except ImportError:
        return
    live = {ord(row[2][0])
            for table in (mujoco.mjVISSTRING, mujoco.mjRNDSTRING)
            for row in table if row[2]}
    assert live.isdisjoint(mjlab_play.COMMAND_KEYS), sorted(map(chr, live))
    # the hardcoded list above must not fall behind the installed tables
    assert live <= MUJOCO_SHORTCUT_KEYS, sorted(map(chr, live - MUJOCO_SHORTCUT_KEYS))


# ------------------------------------------------------- PolicyAdapter

def test_adapter_picks_group_and_passes_through():
    player = FakePlayer()
    obs = {'actor': torch.ones(4, 8), 'critic': torch.zeros(4, 12)}
    adapter = PolicyAdapter(player, 'actor', deterministic=True)

    actions = adapter(obs)
    sent_obs, det = player.calls[0]
    assert sent_obs is obs['actor']
    assert det is True
    assert torch.allclose(actions, obs['actor'] * 2.0)


def test_adapter_stochastic_flag_passthrough():
    player = FakePlayer()
    adapter = PolicyAdapter(player, 'policy', deterministic=False)
    adapter({'policy': torch.zeros(2, 3)})
    assert player.calls[0][1] is False


def test_adapter_reasserts_command_every_call():
    player = FakePlayer()
    controller = FakeController()
    adapter = PolicyAdapter(player, 'actor', controller=controller)
    obs = {'actor': torch.zeros(2, 3)}
    adapter(obs)
    adapter(obs)
    assert controller.applies == 2


def test_adapter_without_controller():
    adapter = PolicyAdapter(FakePlayer(), 'actor', controller=None)
    adapter({'actor': torch.zeros(1, 2)})  # must not raise


def test_adapter_reset_calls_player_reset():
    # PpoPlayerContinuous.reset is init_rnn: zero hidden states on reset
    player = FakePlayer()
    PolicyAdapter(player, 'actor').reset()
    assert player.resets == 1


def test_viewer_reset_reaches_player():
    # mjlab's BaseViewer.reset_environment calls policy.reset() when the
    # policy defines it: ENTER (native) and the Reset button (viser) must
    # zero the player's RNN state, not only the env
    base = pytest.importorskip('mjlab.viewer.base')

    class Viewer(base.BaseViewer):
        def setup(self):
            pass

        def sync_env_to_viewer(self):
            pass

        def sync_viewer_to_env(self):
            pass

        def close(self):
            pass

        def is_running(self):
            return False

    class FakeEnv:
        cfg = types.SimpleNamespace(viewer=None)
        num_envs = 2

        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1

    env, player = FakeEnv(), FakePlayer()
    Viewer(env, PolicyAdapter(player, 'actor')).reset_environment()
    assert env.resets == 1
    assert player.resets == 1


# ------------------------------------------------------ import hygiene

def test_module_imports_without_mjlab():
    # CI has no mjlab: the module (incl. the key-code table) must import with
    # mjlab unimportable, falling back to the GLFW key codes
    repo_root = os.path.realpath(
        os.path.join(os.path.dirname(mjlab_play.__file__), '..', '..'))
    code = (
        "import sys; sys.modules['mjlab'] = None\n"
        "import rl_games.envs.mjlab_play as m\n"
        "assert (m.KEY_KP_0, m.KEY_KP_2, m.KEY_KP_4, m.KEY_KP_6, m.KEY_KP_7,"
        " m.KEY_KP_8, m.KEY_KP_9) == (320, 322, 324, 326, 327, 328, 329)\n"
        "assert set(m.COMMAND_KEYS) == set(range(320, 330)) - {321, 323, 325}\n"
        "print('ok')\n"
    )
    env = dict(os.environ)
    env['PYTHONPATH'] = repo_root + os.pathsep + env.get('PYTHONPATH', '')
    res = subprocess.run([sys.executable, '-c', code],
                         capture_output=True, text=True, env=env)
    assert res.returncode == 0, res.stderr
    assert 'ok' in res.stdout


# ------------------------------------------------------- config sanity

CONFIG_PATH = os.path.realpath(
    os.path.join(os.path.dirname(mjlab_play.__file__),
                 '..', 'configs', 'mjlab', 'ppo_microduck_velocity.yaml'))


def test_microduck_config_parses_and_matches_recipe():
    with open(CONFIG_PATH) as f:
        params = yaml.safe_load(f)['params']

    config = params['config']
    assert config['env_name'] == 'mjlab_microduck_velocity'
    assert config['vecenv_type'] == 'MJLAB'
    assert config['env_config']['task_name'] == 'Mjlab-Velocity-Flat-MicroDuck'

    # batch geometry: 4096 envs x 24 steps in 4 minibatches x 5 mini-epochs
    assert config['num_actors'] == 4096
    assert config['horizon_length'] == 24
    assert config['minibatch_size'] == 24576
    assert config['num_actors'] * config['horizon_length'] == 4 * config['minibatch_size']
    assert config['mini_epochs'] == 5

    # the default MicroDuck config
    assert config['normalize_input'] is True
    assert config['normalize_value'] is False
    assert config['value_bootstrap'] is True    # 20 s truncation
    assert config['gamma'] == 0.99
    assert config['tau'] == 0.95
    assert config['learning_rate'] == 0.001
    assert config['lr_schedule'] == 'adaptive'
    assert config['kl_threshold'] == 0.01
    assert config['e_clip'] == 0.2
    # entropy MUST be 0 on the free per-dim logstd head: a positive bonus
    # puts a constant gradient on logstd and Adam grows sigma to collapse
    assert config['entropy_coef'] == 0.0
    # explicit adaptive-LR band; no [-1, 1] action pre-clamp (env clamps)
    assert config['min_lr'] == 1e-6
    assert config['max_lr'] == 1e-3
    assert config['clip_actions'] is False
    assert config['grad_norm'] == 1.0
    assert config['truncate_grads'] is True
    assert config['max_epochs'] == 4000

    # learned scalar std, init exp(0) = 1
    space = params['network']['space']['continuous']
    assert space['fixed_sigma'] is True
    assert space['sigma_init']['val'] == 0
    assert params['network']['mlp']['units'] == [512, 256, 128]
    assert params['network']['mlp']['activation'] == 'elu'

    # asymmetric central value: own [512,256,128] elu net, obs norm, lr 1e-3
    cv = config['central_value_config']
    assert cv['network']['mlp']['units'] == [512, 256, 128]
    assert cv['network']['mlp']['activation'] == 'elu'
    assert cv['normalize_input'] is True
    assert cv['learning_rate'] == 0.001
    assert cv['minibatch_size'] == 24576
    assert cv['mini_epochs'] == 5
