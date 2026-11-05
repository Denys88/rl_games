import collections
import math
import weakref

import numpy as np
import random
import gymnasium as gym
import torch
from rl_games.common.segment_tree import SumSegmentTree, MinSegmentTree
from rl_games.algos_torch.torch_ext import numpy_to_torch_dtype_dict


class ReplayBuffer(object):
    """
    Replay buffer for storing and sampling transitions.
    """
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
        self._dones = np.zeros(size, dtype=bool)

        self._maxsize = size
        self._next_idx = 0
        self._curr_size = 0

    def __len__(self):
        return self._curr_size

    @torch.no_grad()
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
            obses_t[it] = np.asarray(obs_t)
            actions[it] = np.asarray(action)
            rewards[it] = reward
            obses_tp1[it] = np.asarray(obs_tp1)
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
    """
    Prioritized replay buffer that samples transitions based on priority.
    """
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
        """See ReplayBuffer.add"""
        idx = self._next_idx
        super().add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, self._curr_size)
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


def _cuda_unregister(ptr):
    try:
        torch.cuda.synchronize()  # no kernel may still read the pages
        torch.cuda.cudart().cudaHostUnregister(ptr)
    except Exception:  # interpreter shutdown
        pass


def _page_locked(shape, dtype):
    """Exact-size page-locked host tensor (pin_memory rounds big blocks up to a power of two)."""
    t = torch.empty(shape, dtype=dtype)
    nbytes = t.numel() * t.element_size()
    if nbytes and int(torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nbytes, 3)) == 0:  # portable | mapped
        weakref.finalize(t, _cuda_unregister, t.data_ptr())
        return t
    return torch.empty(shape, dtype=dtype, pin_memory=True)


class _DevicePointer:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {'shape': (nbytes,), 'typestr': '|u1', 'data': (ptr, False), 'version': 3}


def _mapped_view(host, device):
    """CUDA tensor aliasing page-locked ``host`` (UVA), checked by a write from each side; None if unusable."""
    nbytes = host.numel() * host.element_size()
    try:
        raw = torch.as_tensor(_DevicePointer(host.data_ptr(), nbytes), device=device)
        probe = torch.tensor([17, 34, 51], dtype=torch.uint8)
        host_bytes = host.view(-1).view(torch.uint8)
        host_bytes[:3] = probe
        torch.cuda.synchronize(device)
        ok = bool(torch.equal(raw[:3].cpu(), probe))
        raw[:3] = probe.flip(0).to(device)
        torch.cuda.synchronize(device)
        ok = ok and bool(torch.equal(host_bytes[:3], probe.flip(0)))
        return raw.view(host.dtype).view(host.shape) if ok else None
    except Exception:
        return None


class ReplayStorage:
    """Allocates replay columns on the device ('device'), or in page-locked host memory ('host').

    Host columns are read and written by GPU kernels through a mapped view of the same pages (UVA),
    with a CPU gather + copy fallback when the mapping is unavailable.
    """

    def __init__(self, device, storage='device'):
        if storage not in ('device', 'host'):
            raise ValueError(f"replay storage must be 'device' or 'host', got {storage!r}")
        self.device = torch.device(device)
        self.host_tensors = []  # page-locked tensors behind the mapped views, kept alive here
        self.mapped = (storage == 'host' and self.device.type == 'cuda'
                       and _mapped_view(_page_locked((64,), torch.uint8), self.device) is not None)
        # where the columns are indexed: the compute device, or the CPU for an unmapped host ring
        self.rows_device = torch.device('cpu') if storage == 'host' and not self.mapped else self.device
        self.pin = storage == 'host' and not self.mapped and torch.cuda.is_available()

    def alloc(self, shape, dtype, zero=True):
        if not self.mapped:
            make = torch.zeros if zero else torch.empty
            return make(shape, dtype=dtype, device=self.rows_device, pin_memory=self.pin)
        host = _page_locked(shape, dtype)
        if host.numel() == 0:
            return torch.empty(shape, dtype=dtype, device=self.device)
        if zero:
            host.zero_()
        view = _mapped_view(host, self.device)
        if view is None:
            raise RuntimeError(f'could not map {tuple(shape)} {dtype} host rows into {self.device}')
        self.host_tensors.append(host)
        return view

    def rows(self, x):
        # only host -> device copies may run asynchronously
        return x.to(self.rows_device, non_blocking=self.rows_device.type == 'cuda')


class NStepFolder:
    """Folds consecutive vec-env steps into n-step transitions, per env.

    push() takes one step for all envs: 'reward' (B, H), 'terminated' / 'truncated' (B,) bool, and
    'ends', tensors read at the end of the window (the next obs, a final-obs slot). With n steps
    pending it returns the transition anchored at the oldest: the discounted reward sum up to and
    including the first episode end in the window, that end's flags and 'ends', and 'steps' (B,), the
    number of rewards summed. Other keys come from the anchor step.
    """

    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.pending = collections.deque()

    def push(self, step):
        self.pending.append(step)
        if len(self.pending) < self.n_step:
            return None
        out = self._fold()
        self.pending.popleft()
        return out

    def _fold(self):
        last = self.pending[-1]
        ret, term, trunc = last['reward'], last['terminated'], last['truncated']
        ends = dict(last['ends'])
        steps = torch.full(term.shape, self.n_step, dtype=torch.int64, device=term.device)
        for k in range(self.n_step - 2, -1, -1):
            e = self.pending[k]
            done = e['terminated'] | e['truncated']
            ret = e['reward'] + self.gamma * ret * (~done).unsqueeze(-1)
            term = torch.where(done, e['terminated'], term)
            trunc = torch.where(done, e['truncated'], trunc)
            steps = torch.where(done, k + 1, steps)
            for key, value in e['ends'].items():
                ends[key] = torch.where(done.view(-1, *([1] * (value.dim() - 1))), value, ends[key])
        return dict(self.pending[0], reward=ret, terminated=term, truncated=trunc, ends=ends, steps=steps)


def _flat_flag(x):
    return x.bool().view(-1)


class VectorizedReplayBuffer:
    """
    Vectorized replay buffer implementation using Torch tensors.

    Rewards are (N, num_heads). With n_step > 1, add() takes one full vec-env step per call (every
    env, in a fixed order, with an optional ``live`` mask of the rows to keep) and stores the folded
    n-step transition: reward sum up to the first episode end, that end's next obs and flags, and the
    bootstrap discount gamma^k (returned as 'discount' by sample_batch).
    """
    def __init__(self, obs_shape, action_shape, capacity, device, num_heads=1, n_step=1, gamma=0.99,
                 obs_dtype=torch.float32, storage='device'):
        """Create Vectorized Replay buffer.
        Parameters
        ----------
        capacity: int
            Max number of transitions to store. Oldest entries are overwritten
            once the buffer is full.
        """

        self.device = device
        self.storage = ReplayStorage(device, storage)
        alloc = self.storage.alloc

        # Actions and rewards are float32; obs default to float32 (autocast handles training precision)
        self.obs_dtype = obs_dtype
        self.obses = alloc((capacity, *obs_shape), obs_dtype)
        self.next_obses = alloc((capacity, *obs_shape), obs_dtype)
        self.actions = alloc((capacity, *action_shape), torch.float32)
        self.rewards = alloc((capacity, num_heads), torch.float32)
        self.dones = alloc((capacity, 1), torch.bool)
        self.truncated = alloc((capacity, 1), torch.bool)

        self.n_step = n_step
        self.gamma = gamma
        self.folder = NStepFolder(n_step, gamma) if n_step > 1 else None
        self.discounts = alloc((capacity, 1), torch.float32) if n_step > 1 else None

        self.capacity = capacity
        self.idx = 0
        self.full = False

    @torch.no_grad()
    def add(self, obs, action, reward, next_obs, done, truncated=None, live=None):
        if truncated is None:
            truncated = torch.zeros_like(done)
        if self.folder is None:
            if live is not None:
                raise ValueError('live masks need n_step > 1; select the live rows before add()')
            self._write(obs, action, reward, next_obs, done, truncated)
            return
        done = _flat_flag(done)
        t = self.folder.push({
            'obs': obs, 'action': action, 'reward': reward.view(obs.shape[0], -1),
            'terminated': done, 'truncated': _flat_flag(truncated) & ~done, 'live': live,
            'ends': {'next_obs': next_obs},
        })
        if t is None:
            return
        discount = (self.gamma ** t['steps'].float()).unsqueeze(1)
        columns = [t['obs'], t['action'], t['reward'], t['ends']['next_obs'], t['terminated'].unsqueeze(1),
                   t['truncated'].unsqueeze(1), discount]
        if t['live'] is not None:
            rows = t['live'].nonzero(as_tuple=True)[0]
            columns = [c.index_select(0, rows) for c in columns]
        self._write(*columns)

    def _write(self, obs, action, reward, next_obs, done, truncated, discount=None):
        num_observations = obs.shape[0]
        if num_observations > self.capacity:
            # Earlier rows in this batch cannot survive a capacity-sized ring.
            # Advance over them to keep the same write position as sequential adds.
            skipped = num_observations - self.capacity
            obs = obs[skipped:]
            action = action[skipped:]
            reward = reward[skipped:]
            next_obs = next_obs[skipped:]
            done = done[skipped:]
            truncated = truncated[skipped:]
            if discount is not None:
                discount = discount[skipped:]
            self.idx = (self.idx + skipped) % self.capacity
            num_observations = self.capacity
            self.full = True
        remaining_capacity = min(self.capacity - self.idx, num_observations)
        overflow = num_observations - remaining_capacity
        if remaining_capacity < num_observations:
            self.obses[0:overflow].copy_(obs[-overflow:])
            self.actions[0:overflow].copy_(action[-overflow:])
            self.rewards[0:overflow].copy_(reward[-overflow:])
            self.next_obses[0:overflow].copy_(next_obs[-overflow:])
            self.dones[0:overflow].copy_(done[-overflow:])
            self.truncated[0:overflow].copy_(truncated[-overflow:])
            if discount is not None:
                self.discounts[0:overflow].copy_(discount[-overflow:])
            self.full = True

        self.obses[self.idx:self.idx + remaining_capacity].copy_(obs[:remaining_capacity])
        self.actions[self.idx:self.idx + remaining_capacity].copy_(action[:remaining_capacity])
        self.rewards[self.idx:self.idx + remaining_capacity].copy_(reward[:remaining_capacity])
        self.next_obses[self.idx:self.idx + remaining_capacity].copy_(next_obs[:remaining_capacity])
        self.dones[self.idx:self.idx + remaining_capacity].copy_(done[:remaining_capacity])
        self.truncated[self.idx:self.idx + remaining_capacity].copy_(truncated[:remaining_capacity])
        if discount is not None:
            self.discounts[self.idx:self.idx + remaining_capacity].copy_(discount[:remaining_capacity])

        self.idx = (self.idx + num_observations) % self.capacity
        # num_observations > 0 guard: a zero-row add on a fresh buffer leaves
        # idx == 0 and must NOT mark the (empty) buffer as full.
        self.full = self.full or (num_observations > 0 and self.idx == 0)

    def can_sample(self):
        return self.full or self.idx > 0

    def _out(self, x):
        return x.to(self.device, non_blocking=True)

    def sample_batch(self, batch_size):
        """Uniform batch: obs, action, reward (N, H), next_obs, done, truncated, discount (None for n_step 1)."""
        idxs = torch.randint(0,
                            self.capacity if self.full else self.idx,
                            (batch_size,), device=self.storage.rows_device)
        return {
            'obs': self._out(self.obses[idxs]).float(),
            'action': self._out(self.actions[idxs]),
            'reward': self._out(self.rewards[idxs]),
            'next_obs': self._out(self.next_obses[idxs]).float(),
            'done': self._out(self.dones[idxs]),
            'truncated': self._out(self.truncated[idxs]),
            'discount': self._out(self.discounts[idxs]) if self.discounts is not None else None,
        }

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
        dones: torch tensor
            whether the episode terminated at this tuple of (observation, action) or not
        truncated: torch tensor
            whether the episode was truncated (e.g. time limit) at this tuple of (observation, action) or not
        """
        b = self.sample_batch(batch_size)
        return b['obs'], b['action'], b['reward'], b['next_obs'], b['done'], b['truncated']

    def state_dict(self):
        """Return the replay buffer state for checkpointing."""
        state = {
            'obses': self.obses.cpu(),
            'next_obses': self.next_obses.cpu(),
            'actions': self.actions.cpu(),
            'rewards': self.rewards.cpu(),
            'dones': self.dones.cpu(),
            'truncated': self.truncated.cpu(),
            'idx': self.idx,
            'full': self.full,
        }
        if self.discounts is not None:
            state['discounts'] = self.discounts.cpu()
        return state

    def load_state_dict(self, state):
        """Restore the replay buffer state from a checkpoint."""
        self.obses.copy_(state['obses'].to(self.device))
        self.next_obses.copy_(state['next_obses'].to(self.device))
        self.actions.copy_(state['actions'].to(self.device))
        self.rewards.copy_(state['rewards'].to(self.device))
        self.dones.copy_(state['dones'].to(self.device))
        if 'truncated' in state:
            self.truncated.copy_(state['truncated'].to(self.device))
        else:
            self.truncated.zero_()
        if self.discounts is not None:
            self.discounts.copy_(state['discounts'].to(self.device))
        self.idx = state['idx']
        self.full = state['full']


class MemoryEfficientReplayBuffer:
    """Replay ring for a fixed set of vec envs that stores every obs once.

    num_envs rows per env step, so env e's row for step t + n sits n * num_envs rows after its row
    for step t. A row holds the n-step transition anchored at its step (obs, action, per-head reward
    sum, flags of the first episode end in the window, bootstrap discount for n > 1); its next obs
    is read from row i + n * num_envs at sample time, so the newest n steps are not sampled. A window
    that ends in a time-out bootstraps from the true final obs, kept in a side pool (one entry per
    time-out serves the up to n rows that end there); terminated windows keep nothing.

    add() takes one full vec-env step with same-step autoreset: next_obs of a time-out env must be
    its true final obs. Optional ``compactor`` drops reconstructible obs columns before storage and
    rebuilds them at sample time from the kept columns plus an int32 aux row per obs. It provides
    compact_dim, aux_dim, compact(obs (N, O)) -> (N, compact_dim) and expand(compact, aux) -> (N, O).
    """

    def __init__(self, obs_shape, action_shape, capacity, device, num_envs, num_heads=1, n_step=1, gamma=0.99,
                 obs_dtype=torch.float32, storage='device', compactor=None):
        if len(obs_shape) != 1:
            raise ValueError(f'the memory-efficient replay needs flat observations, got {tuple(obs_shape)}')
        if n_step * num_envs >= capacity:
            raise ValueError(f'replay capacity {capacity} must exceed n_step * num_envs = {n_step * num_envs}')
        self.device = torch.device(device)
        self.storage = ReplayStorage(device, storage)
        self.num_envs, self.capacity, self.n_step, self.gamma = num_envs, capacity, n_step, gamma
        self.obs_dtype = obs_dtype
        self.compactor = compactor
        self.stored_dim = compactor.compact_dim if compactor is not None else obs_shape[0]
        self.aux_dim = compactor.aux_dim if compactor is not None else 0

        alloc = lambda shape, dtype: self.storage.alloc(shape, dtype, zero=False)
        self.obs = alloc((capacity, self.stored_dim), obs_dtype)
        self.aux = alloc((capacity, self.aux_dim), torch.int32) if self.aux_dim else None
        self.action = alloc((capacity, *action_shape), torch.float32)
        self.reward = alloc((capacity, num_heads), torch.float32)
        self.terminated = alloc((capacity,), torch.bool)
        self.truncated = alloc((capacity,), torch.bool)
        self.discount = alloc((capacity,), torch.float32) if n_step > 1 else None
        # pool slot of the true final obs for time-out rows, -1 otherwise
        self.final_slot = alloc((capacity,), torch.int64)
        self.final_slot.fill_(-1)

        self._pool_obs = None
        self._pool_aux = None
        self._pool_birth = np.zeros(0, dtype=np.int64)  # env step of each slot's time-out
        self._pool_next = 0

        self.folder = NStepFolder(n_step, gamma)
        self.size = 0
        self.ptr = 0
        self.steps = 0  # env steps added

    def nbytes(self):
        tensors = [self.obs, self.aux, self.action, self.reward, self.terminated, self.truncated, self.discount,
                   self.final_slot, self._pool_obs, self._pool_aux]
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)

    def _stored(self, obs):
        if self.compactor is not None:
            obs = self.compactor.compact(obs)
        return obs.to(self.obs_dtype)

    def _ensure_pool(self, count, oldest_live_step):
        """Slots for ``count`` new final obs; grows the pool rather than overwrite a slot still referenced."""
        size = len(self._pool_birth)
        if size:
            slots = (self._pool_next + np.arange(count)) % size
            if count <= size and not (self._pool_birth[slots] >= oldest_live_step).any():
                self._pool_next = int((self._pool_next + count) % size)
                return slots
        new_size = max(2 * size, size + count, 8 * self.num_envs)
        obs = self.storage.alloc((new_size, self.stored_dim), self.obs_dtype, zero=False)
        aux = self.storage.alloc((new_size, self.aux_dim), torch.int32, zero=False) if self.aux_dim else None
        if size:
            obs[:size] = self._pool_obs
            if aux is not None:
                aux[:size] = self._pool_aux
        self._pool_obs, self._pool_aux = obs, aux
        birth = np.full(new_size, np.iinfo(np.int64).min, dtype=np.int64)
        birth[:size] = self._pool_birth
        self._pool_birth = birth
        slots = size + np.arange(count)
        self._pool_next = int((size + count) % new_size)
        return slots

    def _store_final(self, timeout, final_obs, final_aux):
        """Pool slot per env, -1 where the env did not time out at this step."""
        slot = torch.full((self.num_envs,), -1, dtype=torch.int64, device=self.device)
        envs = timeout.nonzero().squeeze(-1)
        if envs.numel() == 0:
            return slot
        # rows anchored at or after this step may still be alive (conservative by one partial batch)
        oldest_live = self.steps - self.n_step - math.ceil(self.capacity / self.num_envs) + 1
        slots = self._ensure_pool(int(envs.numel()), oldest_live)
        self._pool_birth[slots] = self.steps
        slots_t = torch.as_tensor(slots, device=self.device)
        rows = self.storage.rows(slots_t)
        self._pool_obs[rows] = self.storage.rows(self._stored(final_obs[envs]))
        if self._pool_aux is not None:
            self._pool_aux[rows] = self.storage.rows(final_aux[envs].to(torch.int32))
        slot[envs] = slots_t
        return slot

    @torch.no_grad()
    def add(self, obs, action, reward, next_obs, done, truncated=None, live=None, obs_aux=None, next_obs_aux=None):
        """One env step for all envs; done = terminated. A step both terminated and truncated counts as terminated."""
        if live is not None:
            raise ValueError('the memory-efficient replay needs same-step autoreset (every row live)')
        if obs.shape[0] != self.num_envs:
            raise ValueError(f'expected {self.num_envs} envs, got {obs.shape[0]}')
        if self.compactor is not None and (obs_aux is None or next_obs_aux is None):
            raise ValueError('a compacting replay needs obs_aux and next_obs_aux')
        terminated = _flat_flag(done.to(self.device))
        truncated = torch.zeros_like(terminated) if truncated is None else _flat_flag(truncated.to(self.device))
        truncated = truncated & ~terminated
        step = {
            'obs': self._stored(obs.to(self.device)),
            'action': action.to(self.device, torch.float32),
            'reward': reward.to(self.device, torch.float32).view(self.num_envs, -1),
            'terminated': terminated,
            'truncated': truncated,
            'ends': {'slot': self._store_final(truncated, next_obs.to(self.device), next_obs_aux)},
        }
        if obs_aux is not None:
            step['aux'] = obs_aux.to(self.device, torch.int32)
        self.steps += 1
        t = self.folder.push(step)
        if t is not None:
            self._write(t)

    def _write(self, t):
        b = self.num_envs
        if self.ptr + b <= self.capacity:
            rows = slice(self.ptr, self.ptr + b)
        else:
            rows = self.storage.rows((self.ptr + torch.arange(b, device=self.device)) % self.capacity)
        to_rows = self.storage.rows
        self.obs[rows] = to_rows(t['obs'])
        if self.aux is not None:
            self.aux[rows] = to_rows(t['aux'])
        self.action[rows] = to_rows(t['action'])
        self.reward[rows] = to_rows(t['reward'])
        self.terminated[rows] = to_rows(t['terminated'])
        self.truncated[rows] = to_rows(t['truncated'])
        if self.discount is not None:
            self.discount[rows] = to_rows(self.gamma ** t['steps'].float())
        self.final_slot[rows] = to_rows(t['ends']['slot'])
        self.ptr = (self.ptr + b) % self.capacity
        self.size = min(self.size + b, self.capacity)

    def can_sample(self):
        return self.size > self.n_step * self.num_envs

    def sample_batch(self, batch_size, rows=None):
        """Uniform batch on the compute device; obs and next_obs come back as float32."""
        lag = self.n_step * self.num_envs
        if rows is None:
            rows = torch.randint(0, self.size - lag, (batch_size,), device=self.storage.rows_device)
            if self.size == self.capacity:
                rows = (rows + self.ptr) % self.capacity  # logical 0 = oldest row
        else:
            rows = self.storage.rows(torch.as_tensor(rows, dtype=torch.long))
        both = torch.cat([rows, (rows + lag) % self.capacity])
        obs = self.obs[both]
        aux = self.aux[both] if self.aux is not None else None
        if self._pool_obs is not None:
            n = rows.shape[0]
            slot = self.final_slot[rows]
            hit = (slot >= 0).unsqueeze(-1)
            safe = slot.clamp(min=0)
            obs[n:] = torch.where(hit, self._pool_obs[safe], obs[n:])
            if aux is not None:
                aux[n:] = torch.where(hit, self._pool_aux[safe], aux[n:])
        obs = obs.to(self.device, non_blocking=True).float()
        if self.compactor is not None:
            obs = self.compactor.expand(obs, aux.to(self.device, non_blocking=True))
        obs, next_obs = obs.chunk(2)

        def out(x):
            return x.to(self.device, non_blocking=True)

        return {
            'obs': obs,
            'action': out(self.action[rows]),
            'reward': out(self.reward[rows]),
            'next_obs': next_obs,
            'done': out(self.terminated[rows]).unsqueeze(1),
            'truncated': out(self.truncated[rows]).unsqueeze(1),
            'discount': out(self.discount[rows]).unsqueeze(1) if self.discount is not None else None,
        }

    def sample(self, batch_size):
        b = self.sample_batch(batch_size)
        return b['obs'], b['action'], b['reward'], b['next_obs'], b['done'], b['truncated']

    def state_dict(self):
        raise NotImplementedError('saving the memory-efficient replay is not supported')


class ExperienceBuffer:
    """
    General experience buffer for on-policy algorithms.
    """
    def __init__(self, env_info, algo_info, device, aux_tensor_dict=None):
        self.env_info = env_info
        self.algo_info = algo_info
        self.device = device

        self.num_agents = env_info.get('agents', 1)
        self.action_space = env_info['action_space']

        self.num_actors = algo_info['num_actors']
        self.horizon_length = algo_info['horizon_length']
        self.has_central_value = algo_info['has_central_value']
        self.use_action_masks = algo_info.get('use_action_masks', False)
        batch_size = self.num_actors * self.num_agents
        self.is_discrete = False
        self.is_multi_discrete = False
        self.is_continuous = False
        self.obs_base_shape = (self.horizon_length, self.num_agents * self.num_actors)
        self.state_base_shape = (self.horizon_length, self.num_actors)

        # Check action space type by class name for gym/gymnasium compatibility
        action_space_type = type(self.action_space).__name__

        if action_space_type == 'Discrete':
            self.actions_shape = ()
            self.actions_num = self.action_space.n
            self.is_discrete = True
        elif action_space_type == 'Tuple':
            self.actions_shape = (len(self.action_space),)
            self.actions_num = [action.n for action in self.action_space]
            self.is_multi_discrete = True
        elif action_space_type == 'Box':
            self.actions_shape = (self.action_space.shape[0],)
            self.actions_num = self.action_space.shape[0]
            self.is_continuous = True
        self.tensor_dict = {}

        self._init_from_env_info(self.env_info)

        self.aux_tensor_dict = aux_tensor_dict
        if self.aux_tensor_dict is not None:
            self._init_from_aux_dict(self.aux_tensor_dict)

    def _init_from_env_info(self, env_info):
        obs_base_shape = self.obs_base_shape
        state_base_shape = self.state_base_shape

        self.tensor_dict['obses'] = self._create_tensor_from_space(env_info['observation_space'], obs_base_shape)
        if self.has_central_value:
            # Some envs omit state_space despite central_value being enabled.
            # Fall back to observation_space and raise a clear error if even that is missing.
            state_space = env_info.get('state_space', None)
            if state_space is None:
                state_space = env_info.get('observation_space', None)
            self.tensor_dict['states'] = self._create_tensor_from_space(state_space, state_base_shape)

        val_space = gym.spaces.Box(low=0, high=1, shape=(env_info.get('value_size', 1),))
        self.tensor_dict['rewards'] = self._create_tensor_from_space(val_space, obs_base_shape)
        self.tensor_dict['values'] = self._create_tensor_from_space(val_space, obs_base_shape)
        self.tensor_dict['neglogpacs'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1, shape=(), dtype=np.float32), obs_base_shape)
        self.tensor_dict['dones'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1, shape=(), dtype=np.uint8), obs_base_shape)

        if self.is_discrete or self.is_multi_discrete:
            self.tensor_dict['actions'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1, shape=self.actions_shape, dtype=int), obs_base_shape)
        if self.use_action_masks:
            self.tensor_dict['action_masks'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1, shape=(np.sum(self.actions_num),), dtype=bool), obs_base_shape)
        if self.is_continuous:
            self.tensor_dict['actions'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1, shape=self.actions_shape, dtype=np.float32), obs_base_shape)
            self.tensor_dict['mus'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1, shape=self.actions_shape, dtype=np.float32), obs_base_shape)
            self.tensor_dict['sigmas'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1, shape=self.actions_shape, dtype=np.float32), obs_base_shape)

    def _init_from_aux_dict(self, tensor_dict):
        obs_base_shape = self.obs_base_shape
        for k, v in tensor_dict.items():
            self.tensor_dict[k] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1, shape=(v), dtype=np.float32), obs_base_shape)

    def _create_tensor_from_space(self, space, base_shape):
        if space is None:
            # Make the failure explicit early instead of propagating None
            raise ValueError("Space is None while allocating tensors. Ensure env_info provides required spaces.")

        # Check space type by class name for gym/gymnasium compatibility
        space_type = type(space).__name__

        if space_type == 'Box':
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            return torch.zeros(base_shape + space.shape, dtype=dtype, device=self.device)
        elif space_type == 'Discrete':
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            return torch.zeros(base_shape, dtype=dtype, device=self.device)
        elif space_type == 'Tuple':
            # Assuming that tuple is only Discrete tuple
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            tuple_len = len(space)
            return torch.zeros(base_shape + (tuple_len,), dtype=dtype, device=self.device)
        elif space_type == 'Dict':
            t_dict = {}
            for k, v in space.spaces.items():
                t_dict[k] = self._create_tensor_from_space(v, base_shape)
            return t_dict
        else:
            # If we reach here, unsupported space type
            raise ValueError(f"Unsupported space type: {type(space).__name__} ({type(space)}). Expected Box, Discrete, Tuple, or Dict.")

    def update_data(self, name, index, val):
        """
        Update the tensor in the experience buffer for the given name at the specified index.

        Args:
            name (str): Key in the tensor dictionary.
            index: Index at which to update the tensor.
            val (dict or torch.Tensor): New value(s) to store.
        """
        stored_val = self.tensor_dict[name]
        if isinstance(stored_val, dict):
            # If the stored value is a dict (from Dict observation space)
            if isinstance(val, dict):
                # Update each tensor in the dict
                for k, v in val.items():
                    stored_val[k][index, :] = v
            else:
                raise ValueError(f"Expected dict value for '{name}' but got {type(val)}")
        else:
            # Regular tensor update
            if isinstance(val, dict):
                raise ValueError(f"Expected tensor value for '{name}' but got dict")
            else:
                stored_val[index, :] = val

    def update_data_rnn(self, name, indices, play_mask, val):
        """
        Update a tensor in the RNN experience buffer.

        Depending on the type of `val`, this method updates the data in `self.tensor_dict[name]` at the
        specified indices and masked by `play_mask`. If `val` is a dict, each key is updated individually.

        Args:
            name (str): Key in the tensor dictionary.
            indices: Indices at which to update the tensor.
            play_mask: Mask indicating positions to update.
            val (dict or torch.Tensor): Value(s) for the update.

        Raises:
            TypeError: If `val` is not a dict or torch.Tensor.
            KeyError: If a key in `val` is not found in `self.tensor_dict[name]`.
        """
        if not isinstance(val, (dict, torch.Tensor)):
            raise TypeError(f"Expected dict or tensor, got {type(val)}")

        stored_val = self.tensor_dict[name]
        if isinstance(stored_val, dict):
            # If the stored value is a dict (from Dict observation space)
            if isinstance(val, dict):
                for k, v in val.items():
                    if k not in stored_val:
                        raise KeyError(f"Key {k} not found in tensor_dict[{name}]")
                    stored_val[k][indices, play_mask] = v
            else:
                raise ValueError(f"Expected dict value for '{name}' but got {type(val)}")
        else:
            # Regular tensor update
            if isinstance(val, dict):
                raise ValueError(f"Expected tensor value for '{name}' but got dict")
            else:
                stored_val[indices, play_mask] = val

    def get_transformed(self, transform_op):
        res_dict = {}
        for k, v in self.tensor_dict.items():
            if isinstance(v, dict):
                transformed_dict = {}
                for kd, vd in v.items():
                    transformed_dict[kd] = transform_op(vd)
                res_dict[k] = transformed_dict
            else:
                res_dict[k] = transform_op(v)

        return res_dict

    def get_transformed_list(self, transform_op, tensor_list):
        res_dict = {}
        for k in tensor_list:
            v = self.tensor_dict.get(k)
            if v is None:
                continue
            if isinstance(v, dict):
                transformed_dict = {}
                for kd, vd in v.items():
                    transformed_dict[kd] = transform_op(vd)
                res_dict[k] = transformed_dict
            else:
                res_dict[k] = transform_op(v)

        return res_dict
