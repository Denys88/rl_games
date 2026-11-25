"""Vector evaluation excludes NEXT_STEP reset rows and their recurrent state."""

import re

import gymnasium as gym
import numpy as np
import pytest
import torch

from rl_games.algos_torch.players import PpoPlayerDiscrete
from rl_games.common.gymnasium_vecenv import GymnasiumVecEnv


class EvaluationResetEnv:
    def __init__(self, mode, tensor):
        self.mode = mode
        self.tensor = tensor
        self.num_envs = 2
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.lengths = np.array([2, 3])
        self.completed = [[], []]
        self.steps = np.zeros(2, dtype=int)
        self.pending = np.zeros(2, dtype=bool)
        self.episodes = np.zeros(2, dtype=int)

    def get_env_info(self):
        return {"observation_space": self.observation_space, "action_space": self.action_space,
                "agents": 1, "autoreset_mode": self.mode}

    def obs(self):
        out = np.stack([np.arange(2), self.steps, self.episodes, np.ones(2)], axis=-1).astype(np.float32)
        return torch.from_numpy(out) if self.tensor else out

    def reset(self):
        self.steps[:] = 0
        self.pending[:] = False
        self.episodes[:] = 0
        self.completed = [[], []]
        return self.obs()

    def step(self, actions):
        assert actions.shape == (2,)
        rewards = np.ones(2, dtype=np.float32)
        dones = np.zeros(2, dtype=bool)
        for actor in range(2):
            if self.pending[actor]:
                self.steps[actor] = 0
                self.episodes[actor] += 1
                self.pending[actor] = False
                rewards[actor] = 0
                continue
            self.steps[actor] += 1
            if self.steps[actor] == self.lengths[actor]:
                dones[actor] = True
                self.completed[actor].append(int(self.steps[actor]))
                if self.mode == "next_step":
                    self.pending[actor] = True
                else:
                    self.steps[actor] = 0
                    self.episodes[actor] += 1
        if self.tensor:
            rewards, dones = torch.from_numpy(rewards), torch.from_numpy(dones)
        return self.obs(), rewards, dones, {}


def make_player(env, recurrent, balanced):
    network = {
        "name": "actor_critic", "separate": False, "space": {"discrete": {}},
        "mlp": {"units": [8], "activation": "elu", "initializer": {"name": "default"}},
    }
    if recurrent:
        network["rnn"] = {"name": recurrent, "units": 8, "layers": 1}
    return PpoPlayerDiscrete({
        "model": {"name": "discrete_a2c"}, "network": network,
        "config": {
            "env_name": "evaluation-reset-test", "env_info": env.get_env_info(), "vec_env": env,
            "device_name": "cpu", "normalize_input": False, "normalize_value": False,
            "num_actors": 2,
            "player": {"games_num": 4, "max_steps": 20, "print_stats": False,
                       "balance_env_rewards": balanced},
        },
    })


@pytest.mark.parametrize("mode", ["same_step", "next_step"])
@pytest.mark.parametrize("recurrent", [None, "gru", "lstm"])
@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("balanced", [False, True])
def test_vector_evaluation_counts_live_steps_and_resets_recurrent_state(
        mode, recurrent, tensor, balanced, capsys):
    torch.manual_seed(19)
    env = EvaluationResetEnv(mode, tensor)
    player = make_player(env, recurrent, balanced)
    starts = []
    if recurrent:
        def capture(module, inputs):
            data = inputs[0]
            fresh = data["obs"][:, 1] == 0
            for state in data["rnn_states"]:
                starts.append(state[:, fresh, :].clone())
        handle = player.model.register_forward_pre_hook(capture)
    player.run()
    if recurrent:
        handle.remove()
        assert starts
        assert all(torch.count_nonzero(state) == 0 for state in starts)
    summary = re.search(r"av reward: (\S+) av steps: (\S+)", capsys.readouterr().out)
    assert summary is not None
    if balanced:
        expected = np.mean([np.mean(lengths) for lengths in env.completed if lengths])
    else:
        expected = np.mean([length for lengths in env.completed for length in lengths])
    assert float(summary[1]) == pytest.approx(expected)
    assert float(summary[2]) == pytest.approx(expected)


@pytest.mark.parametrize("recurrent", [None, "gru", "lstm"])
def test_native_gymnasium_evaluation_does_not_count_reset_steps(recurrent, capsys):
    env = GymnasiumVecEnv("CartPole-v1", 2, use_async=False, max_episode_steps=2)
    try:
        make_player(env, recurrent, False).run()
        summary = re.search(r"av reward: (\S+) av steps: (\S+)", capsys.readouterr().out)
        assert summary is not None
        assert float(summary[1]) == pytest.approx(2.0)
        assert float(summary[2]) == pytest.approx(2.0)
    finally:
        env.env.close()
