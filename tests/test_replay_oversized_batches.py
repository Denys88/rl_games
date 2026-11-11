"""Ring-retention invariants for replay writes larger than the capacity."""

from collections import deque

import pytest
import torch

from rl_games.common.experience import VectorizedReplayBuffer


FIELDS = ("obses", "actions", "rewards", "next_obses", "dones", "truncated")


def transition_batch(start, count):
    identifiers = torch.arange(start, start + count, dtype=torch.float32)
    observations = torch.stack((identifiers, identifiers + 100), dim=1)
    actions = torch.stack((identifiers + 200, -identifiers), dim=1)
    rewards = (identifiers + 300).unsqueeze(1)
    next_observations = observations + 400
    dones = (identifiers.remainder(2) == 0).unsqueeze(1)
    truncated = (identifiers.remainder(3) == 0).unsqueeze(1)
    return observations, actions, rewards, next_observations, dones, truncated


def append_reference(history, batch):
    # A bounded chronological queue is independent of the production ring writes.
    for row in zip(*(field.tolist() for field in batch)):
        history.append(row)


def assert_retention(buffer, history, total):
    assert buffer.idx == total % buffer.capacity
    assert buffer.full is (total >= buffer.capacity)
    for column, name in enumerate(FIELDS):
        actual = getattr(buffer, name)
        if buffer.full:
            actual = torch.cat((actual[buffer.idx:], actual[:buffer.idx]))
        else:
            actual = actual[:buffer.idx]
        expected = torch.tensor(
            [row[column] for row in history], dtype=actual.dtype
        ).reshape(actual.shape)
        assert torch.equal(actual, expected), name


@pytest.mark.parametrize("capacity", [1, 2, 3, 5, 8])
def test_oversized_add_matches_chronological_queue(capacity):
    for initial_count in (0, 1, capacity - 1, capacity):
        for incoming_count in (capacity + 1, 2 * capacity, 2 * capacity + 3):
            buffer = VectorizedReplayBuffer((2,), (2,), capacity, "cpu")
            history = deque(maxlen=capacity)
            initial = transition_batch(0, initial_count)
            buffer.add(*initial)
            append_reference(history, initial)
            incoming = transition_batch(initial_count, incoming_count)
            buffer.add(*incoming)
            append_reference(history, incoming)
            assert_retention(buffer, history, initial_count + incoming_count)


def test_checkpoint_restore_and_further_writes_preserve_ring_position():
    capacity = 5
    original = VectorizedReplayBuffer((2,), (2,), capacity, "cpu")
    history = deque(maxlen=capacity)
    for start, count in ((0, 4), (4, 3)):
        batch = transition_batch(start, count)
        original.add(*batch)
        append_reference(history, batch)
    restored = VectorizedReplayBuffer((2,), (2,), capacity, "cpu")
    restored.load_state_dict(original.state_dict())
    total = 7
    for count in (12, 0, 2, 5, 16):
        batch = transition_batch(total, count)
        restored.add(*batch)
        append_reference(history, batch)
        total += count
        assert_retention(restored, history, total)


def test_sampling_returns_only_retained_and_aligned_transition_fields():
    buffer = VectorizedReplayBuffer((2,), (2,), 5, "cpu")
    buffer.add(*transition_batch(0, 19))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        observations, actions, rewards, next_observations, dones, truncated = buffer.sample(512)
    identifiers = observations[:, 0]
    assert ((identifiers >= 14) & (identifiers < 19)).all()
    assert torch.equal(observations[:, 1], identifiers + 100)
    assert torch.equal(actions[:, 0], identifiers + 200)
    assert torch.equal(actions[:, 1], -identifiers)
    assert torch.equal(rewards[:, 0], identifiers + 300)
    assert torch.equal(next_observations, observations + 400)
    assert torch.equal(dones[:, 0], identifiers.remainder(2) == 0)
    assert torch.equal(truncated[:, 0], identifiers.remainder(3) == 0)


def test_omitted_truncation_flags_remain_false_for_large_batch():
    buffer = VectorizedReplayBuffer((2,), (2,), 3, "cpu")
    buffer.add(*transition_batch(0, 2))
    batch = transition_batch(2, 11)
    buffer.add(*batch[:-1])
    history = deque(maxlen=3)
    batch = (*batch[:-1], torch.zeros_like(batch[-1]))
    append_reference(history, batch)
    assert_retention(buffer, history, 13)
    assert not buffer.truncated.any()


def test_empty_and_single_wrap_controls_keep_original_contract():
    buffer = VectorizedReplayBuffer((2,), (2,), 5, "cpu")
    history = deque(maxlen=5)
    total = 0
    for count in (0, 2, 1, 4, 0, 5, 1):
        batch = transition_batch(total, count)
        buffer.add(*batch)
        append_reference(history, batch)
        total += count
        assert_retention(buffer, history, total)


def test_public_sac_training_with_more_actors_than_replay_capacity(tmp_path):
    from tests.test_sac_correctness import make_fake_env_sac_agent

    agent, environment = make_fake_env_sac_agent(
        num_envs=11,
        ep_len=100,
        replay_buffer_size=3,
        batch_size=8,
        num_steps_per_episode=2,
        num_warmup_frames=0,
        max_epochs=1,
        train_dir=str(tmp_path),
        name="pytest_sac_oversized_replay",
    )
    parameters_before = [parameter.detach().clone() for parameter in agent.model.sac_network.actor.parameters()]
    agent.train()
    buffer = agent.replay_buffer
    assert buffer.full
    assert buffer.idx == (2 * environment.num_envs) % buffer.capacity
    chronological = torch.cat((buffer.obses[buffer.idx:], buffer.obses[:buffer.idx]))
    assert torch.equal(chronological[:, 0], torch.tensor([8.0, 9.0, 10.0]))
    assert torch.equal(chronological[:, 1], torch.ones(3))
    assert torch.equal(buffer.next_obses[:, 1], torch.full((3,), 2.0))
    parameters_after = list(agent.model.sac_network.actor.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in parameters_after)
    assert any(not torch.equal(before, after) for before, after in zip(parameters_before, parameters_after))
