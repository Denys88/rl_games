"""CI-safe tests for the mjlab live-play module -- no mjlab required.

rl_games.envs.mjlab_play keeps all mjlab imports function-local or guarded,
so its pure logic (obs-group pick, command re-assert pattern, policy adapter)
is testable against stubs whether or not mjlab is installed.
"""

import os
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

    def get_action(self, obs, is_deterministic=False):
        self.calls.append((obs, is_deterministic))
        return obs * 2.0


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


def test_key_callback_steps_command():
    term = FakeVelocityTerm()
    ctrl = CommandController(make_env({'twist': term}))

    ctrl.key_callback(ord('W'))
    ctrl.key_callback(ord('W'))
    ctrl.key_callback(mjlab_play.KEY_UP)      # arrow == WASD
    assert ctrl.command == pytest.approx([0.3, 0.0, 0.0])

    ctrl.key_callback(ord('S'))
    ctrl.key_callback(mjlab_play.KEY_LEFT)    # yaw left
    ctrl.key_callback(ord('D'))               # yaw right
    ctrl.key_callback(ord('Q'))               # strafe left
    assert ctrl.command == pytest.approx([0.2, 0.1, 0.0])
    # every key press applies immediately
    assert torch.allclose(term.vel_command_b,
                          torch.tensor([0.2, 0.1, 0.0]).expand(4, 3))

    ctrl.key_callback(ord('X'))               # zero
    assert ctrl.command == pytest.approx([0.0, 0.0, 0.0])


def test_key_callback_ignores_unknown_keys():
    term = FakeVelocityTerm()
    ctrl = CommandController(make_env({'twist': term}))
    ctrl.set_velocity(0.5, 0.0, 0.0)
    before = term.vel_command_b.clone()
    ctrl.key_callback(ord('Z'))
    assert ctrl.command == pytest.approx([0.5, 0.0, 0.0])
    assert torch.equal(term.vel_command_b, before)


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


# ------------------------------------------------------ import hygiene

def test_module_imports_without_mjlab():
    # CI has no mjlab: the module (incl. the key-code table) must import with
    # mjlab unimportable, falling back to the GLFW key codes
    repo_root = os.path.realpath(
        os.path.join(os.path.dirname(mjlab_play.__file__), '..', '..'))
    code = (
        "import sys; sys.modules['mjlab'] = None\n"
        "import rl_games.envs.mjlab_play as m\n"
        "assert (m.KEY_UP, m.KEY_DOWN, m.KEY_LEFT, m.KEY_RIGHT) == (265, 264, 263, 262)\n"
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

    # the RSL-RL recipe translation
    assert config['normalize_input'] is True
    assert config['normalize_value'] is False
    assert config['value_bootstrap'] is True    # 20 s truncation
    assert config['gamma'] == 0.99
    assert config['tau'] == 0.95
    assert config['learning_rate'] == 0.001
    assert config['lr_schedule'] == 'adaptive'
    assert config['kl_threshold'] == 0.01
    assert config['e_clip'] == 0.2
    assert config['entropy_coef'] == 0.01
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
