import numpy as np
import random
import gym
import torch
from rl_games.common.segment_tree import SumSegmentTree, MinSegmentTree
from rl_games.algos_torch.torch_ext import numpy_to_torch_dtype_dict

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



class ExperienceBuffer:
    '''
    More generalized than replay buffers.
    Implemented for on-policy algos
    '''
    def __init__(self, env_info, algo_info, device):
        self.env_info = env_info
        self.algo_info = algo_info
        self.device = device

        self.num_agents = env_info['agents']
        self.action_space = env_info['action_space']
        
        self.num_actors = algo_info['num_actors']
        self.steps_num = algo_info['steps_num']
        self.has_central_value = algo_info['has_central_value']
        self.use_action_masks = algo_info.get('use_action_masks', False)
        batch_size = self.num_actors * self.num_agents
        self.is_discrete = False
        self.is_multi_discrete = False
        self.is_continuous = False

        if type(self.action_space) is gym.spaces.Discrete:
            self.actions_shape = ()
            self.actions_num = self.action_space.n
            self.is_discrete = True
        if type(self.action_space) is gym.spaces.Tuple:
            self.actions_shape = (len(self.action_space),) 
            self.actions_num = [action.n for action in self.action_space]
            self.is_multi_discrete = True
        if type(self.action_space) is gym.spaces.Box:
            self.actions_shape = (self.action_space.shape[0],) 
            self.actions_num = self.action_space.shape[0]
            self.is_continuous = True
        self.tensor_dict = {}
        self._init_from_env_info(self.env_info)

    def _init_from_env_info(self, env_info):
        obs_base_shape = (self.steps_num, self.num_agents * self.num_actors)
        state_base_shape = (self.steps_num, self.num_actors)

        self.tensor_dict['obses'] = self._create_tensor_from_space(env_info['observation_space'], obs_base_shape)
        if self.has_central_value:
            self.tensor_dict['states'] = self._create_tensor_from_space(env_info['state_space'], state_base_shape)
        
        val_space = gym.spaces.Box(low=0, high=1,shape=(env_info.get('value_size',1),))
        self.tensor_dict['rewards'] = self._create_tensor_from_space(val_space, obs_base_shape)
        self.tensor_dict['values'] = self._create_tensor_from_space(val_space, obs_base_shape)
        self.tensor_dict['neglogpacs'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=(), dtype=np.float32), obs_base_shape)
        self.tensor_dict['dones'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=(), dtype=np.uint8), obs_base_shape)
        if self.is_discrete:
            self.tensor_dict['actions'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape, dtype=np.long), obs_base_shape)
        if self.use_action_masks:
            self.tensor_dict['action_masks'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape + (np.sum(self.actions_num),), dtype=np.bool), obs_base_shape)
        if self.is_continuous:
            self.tensor_dict['actions'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape, dtype=np.float32), obs_base_shape)
            self.tensor_dict['mus'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape, dtype=np.float32), obs_base_shape)
            self.tensor_dict['sigmas'] = self._create_tensor_from_space(gym.spaces.Box(low=0, high=1,shape=self.actions_shape, dtype=np.float32), obs_base_shape)

    def _create_tensor_from_space(self, space, base_shape):       
        if type(space) is gym.spaces.Box:
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            return torch.zeros(base_shape + space.shape, dtype= dtype, device = self.device)
        if type(space) is gym.spaces.Discrete:
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            return torch.zeros(base_shape, dtype= dtype, device = self.device)
        if type(space) is gym.spaces.Tuple:
            '''
            assuming that tuple is only Discrete tuple
            '''
            dtype = numpy_to_torch_dtype_dict[space.dtype]
            tuple_len = len(space)
            return torch.zeros(base_shape +(tuple_len,), dtype= dtype, device = self.device)
        if type(space) is gym.spaces.Dict:
            t_dict = {}
            for k,v in space.spaces.items():
                t_dict[k] = self._create_tensor_from_space(v, base_shape)
            return t_dict

    def update_data(self, name, index, val):
        if type(val) is dict:
            for k,v in val.items():
                self.tensor_dict[name][k][index,:] = v
        else:
            self.tensor_dict[name][index,:] = val


    def update_data_rnn(self, name, indices,play_mask, val):
        if type(val) is dict:
            for k,v in val:
                self.tensor_dict[name][k][indices,play_mask] = v
        else:
            self.tensor_dict[name][indices,play_mask] = val

    def get_transformed(self, transform_op):
        res_dict = {}
        for k, v in self.tensor_dict.items():
            if type(v) is dict:
                transformed_dict = {}
                for kd,vd in v.items():
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
            if type(v) is dict:
                transformed_dict = {}
                for kd,vd in v.items():
                    transformed_dict[kd] = transform_op(vd)
                res_dict[k] = transformed_dict
            else:
                res_dict[k] = transform_op(v)
        
        return res_dict