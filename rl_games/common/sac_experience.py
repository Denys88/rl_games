import numpy as np
import random

from rl_games.common.segment_tree import SumSegmentTree, MinSegmentTree
import torch


class ReplayBuffer(object):
    def __init__(self, size, ob_space):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._obses = np.zeros((size,) + ob_space.shape, dtype=ob_space.dtype)
        self._next_obses = np.zeros((size,) + ob_space.shape, dtype=ob_space.dtype)
        self._rewards = np.zeros(size)
        self._actions = np.zeros(size, dtype=np.int32)
        self._dones = np.zeros(size, dtype=np.bool)

        self._maxsize = size
        self._next_idx = 0
        self._curr_size = 0

    def __len__(self):
        return self._curr_size

    def add(self, obs_t, action, reward, obs_tp1, done):

        self._curr_size = min(self._curr_size + 1, self._maxsize )

        self._obses[self._next_idx] = obs_t
        self._next_obses[self._next_idx] = obs_tp1
        self._rewards[self._next_idx] = reward
        self._actions[self._next_idx] = action
        self._dones[self._next_idx] = done

        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _get(self, idx):
        return self._obses[idx], self._actions[idx], self._rewards[idx], self._next_obses[idx], self._dones[idx]

    def _encode_sample(self, idxes):
        batch_size = len(idxes)
        obses_t, actions, rewards, obses_tp1, dones = [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size, [None] * batch_size
        it = 0
        for i in idxes:
            data = self._get(i)
            obs_t, action, reward, obs_tp1, done = data
            obses_t[it] = np.array(obs_t, copy=False)
            actions[it] = np.array(action, copy=False)
            rewards[it] = reward
            obses_tp1[it] = np.array(obs_tp1, copy=False)
            dones[it] = done
            it = it + 1
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, self._curr_size - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, ob_space):
        """Create Prioritized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)
        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size, ob_space)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, *args, **kwargs):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self._curr_size - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self._curr_size) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self._curr_size) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights)
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index idxes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < self._curr_size
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)


class VectorizedReplayBuffer:
    def __init__(self, obs_shape, action_shape, capacity, device):
        """Create Vectorized Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        See Also
        --------
        ReplayBuffer.__init__
        """

        self.device = device

        self.obses = torch.empty((capacity, *obs_shape), dtype=torch.float32, device=self.device)
        self.next_obses = torch.empty((capacity, *obs_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32, device=self.device)
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device=self.device)
        self.dones = torch.empty((capacity, 1), dtype=torch.bool, device=self.device)
        self.dones_no_max = torch.empty((capacity, 1), dtype=torch.bool, device=self.device)

        self.capacity = capacity
        self.idx = 0
        self.full = False
        

    def add(self, obs, action, reward, next_obs, done, done_no_max):

        num_observations = obs.shape[0]
        remaining_capacity = min(self.capacity - self.idx, num_observations)
        overflow = num_observations - remaining_capacity
        if remaining_capacity < num_observations:
            self.obses[0: overflow] = obs[-overflow:]
            self.actions[0: overflow] = action[-overflow:]
            self.rewards[0: overflow] = reward[-overflow:]
            self.next_obses[0: overflow] = next_obs[-overflow:]
            self.dones[0: overflow] = done[-overflow:]
            self.dones_no_max[0: overflow] = done_no_max[-overflow:]
            self.full = True
        self.obses[self.idx: self.idx + remaining_capacity] = obs[:remaining_capacity]
        self.actions[self.idx: self.idx + remaining_capacity] = action[:remaining_capacity]
        self.rewards[self.idx: self.idx + remaining_capacity] = reward[:remaining_capacity]
        self.next_obses[self.idx: self.idx + remaining_capacity] = next_obs[:remaining_capacity]
        self.dones[self.idx: self.idx + remaining_capacity] = done[:remaining_capacity]
        self.dones_no_max[self.idx: self.idx + remaining_capacity] = done_no_max[:remaining_capacity]

        self.idx = (self.idx + num_observations) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        obses: torch tensor
            batch of observations
        actions: torch tensor
            batch of actions executed given obs
        rewards: torch tensor
            rewards received as results of executing act_batch
        next_obses: torch tensor
            next set of observations seen after executing act_batch
        not_dones: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not
        not_dones_no_max: torch tensor
            inverse of whether the episode ended at this tuple of (observation, action) or not, specifically exlcuding maximum episode steps
        """

        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = self.obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_obses = self.next_obses[idxs]
        not_dones = ~self.dones[idxs]
        not_dones_no_max = ~self.dones_no_max[idxs]

        return obses, actions, rewards, next_obses, not_dones, not_dones_no_max




