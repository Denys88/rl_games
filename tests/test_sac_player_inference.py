"""SAC evaluation actions are graph-free and usable by vector environments."""

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl_games.algos_torch.players import SACPlayer
from rl_games.common.gymnasium_vecenv import GymnasiumVecEnv


class InferenceEnv:
    def __init__(self, actors, tensor):
        self.actors = actors
        self.tensor = tensor
        self.observation_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
        self.action_space = gym.spaces.Box(np.array([-2.0, 1.0], dtype=np.float32),
                                           np.array([1.0, 5.0], dtype=np.float32))
        self.seen = []

    def get_env_info(self):
        return {"observation_space": self.observation_space,
                "action_space": self.action_space, "agents": 1}

    def reset(self):
        obs = np.ones((self.actors, 4) if self.actors else (4,), dtype=np.float32)
        return torch.from_numpy(obs) if self.tensor else obs

    def step(self, actions):
        assert actions.shape == ((self.actors, 2) if self.actors else (2,))
        if self.tensor:
            assert not actions.requires_grad
            assert actions.grad_fn is None
            reward = torch.ones(self.actors or 1)
            done = torch.ones(self.actors or 1, dtype=torch.bool)
        else:
            assert isinstance(actions, np.ndarray)
            reward = np.ones(self.actors, dtype=np.float32) if self.actors else 1.0
            done = np.ones(self.actors, dtype=bool) if self.actors else True
        self.seen.append(actions)
        return self.reset(), reward, done, {}


def player_for(env, actors, deterministic, normalize):
    return SACPlayer({
        "model": {"name": "soft_actor_critic"},
        "network": {
            "name": "soft_actor_critic", "separate": True, "log_std_bounds": [-5, 1],
            "mlp": {"units": [8], "activation": "elu", "initializer": {"name": "default"}},
        },
        "config": {
            "env_name": "sac-evaluation-test", "env_info": env.get_env_info(), "vec_env": env,
            "device_name": "cpu", "normalize_input": normalize, "num_actors": actors or 1,
            "player": {"games_num": actors or 1, "max_steps": 2, "print_stats": False,
                       "deterministic": deterministic},
        },
    })


@pytest.mark.parametrize("actors", [0, 1, 3])
@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("deterministic", [False, True])
def test_sac_player_vector_evaluation_is_graph_free(actors, tensor, normalize, deterministic):
    env = InferenceEnv(actors, tensor)
    player = player_for(env, actors, deterministic, normalize)
    obs = player.env_reset(env)
    player.get_batch_size(obs, 1)
    with torch.no_grad():
        batch = obs if actors else obs.unsqueeze(0)
        torch.manual_seed(47)
        dist = player.model.actor(player.model.norm_obs(batch))
        reference = player.rescale_actions(dist.mean if deterministic else dist.sample())
        if not actors:
            reference = reference.squeeze()
    torch.manual_seed(47)
    with torch.enable_grad():
        actions = player.get_action(obs, deterministic)
        assert torch.is_grad_enabled()
    assert not actions.requires_grad
    assert actions.grad_fn is None
    torch.testing.assert_close(actions, reference)
    player.run()
    assert len(env.seen) == 1
    assert all(parameter.grad is None for parameter in player.model.parameters())


@pytest.mark.parametrize("deterministic", [False, True])
def test_sac_inference_does_not_save_backward_tensors(deterministic):
    env = InferenceEnv(3, False)
    player = player_for(env, 3, deterministic, False)
    obs = player.env_reset(env)
    player.get_batch_size(obs, 1)
    saved = []
    def pack(tensor):
        saved.append(tensor)
        return tensor
    with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(pack, lambda value: value):
        player.get_action(obs, deterministic)
    assert saved == []


@pytest.mark.parametrize("deterministic", [False, True])
def test_sac_player_runs_with_native_gymnasium_vector_env(deterministic):
    env = GymnasiumVecEnv("Pendulum-v1", 1, use_async=False, max_episode_steps=1)
    try:
        player_for(env, 1, deterministic, True).run()
    finally:
        env.env.close()
