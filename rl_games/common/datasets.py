import torch
import copy
import torch
from torch.utils.data import Dataset
import random


class PPODataset(Dataset):
    def __init__(self, batch_size, minibatch_size, is_discrete, is_rnn, device, seq_length, permute=False):
        if batch_size % minibatch_size != 0:
            raise ValueError("Batch size must be divisible by minibatch size.")
        if batch_size % seq_length != 0:
            raise ValueError("Batch size must be divisible by sequence length.")

        self.is_rnn = is_rnn
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device
        self.length = self.batch_size // self.minibatch_size
        self.is_discrete = is_discrete
        self.is_continuous = not is_discrete
        self.num_games_batch = self.minibatch_size // self.seq_length

        self.special_names = ['rnn_states']
        self.permute = permute
        if self.permute:
            self.permutation_indices = torch.arange(self.batch_size, dtype=torch.long, device=self.device)

    def update_values_dict(self, values_dict):
        """Update the internal values dictionary."""
        self.values_dict = values_dict

    def update_mu_sigma(self, mu, sigma):
        """Update the mu and sigma values in the dataset."""
        start, end = self.last_range
        # Ensure the permutation does not break the logic for updating.
        # if self.permute:
        #    original_indices = self.permutation_indices[start:end]
        #    self.values_dict['mu'][original_indices] = mu
        #    self.values_dict['sigma'][original_indices] = sigma
        # else:
        self.values_dict['mu'][start:end] = mu
        self.values_dict['sigma'][start:end] = sigma

    def apply_permutation(self):
        """Permute the dataset indices if the permutation flag is enabled."""
        if self.permute and not self.is_rnn:
            self.permutation_indices = torch.randperm(self.batch_size, device=self.device, dtype=torch.long)
            for key, value in self.values_dict.items():
                if key not in self.special_names and value is not None:
                    if isinstance(value, dict):
                        for k, v in value.items():
                            self.values_dict[key][k] = v[self.permutation_indices]
                    else:
                        self.values_dict[key] = value[self.permutation_indices]

    def _slice_data(self, data, start, end):
        """Slice data from start to end, handling dictionaries."""
        if isinstance(data, dict):
            return {k: v[start:end] for k, v in data.items()}
        return data[start:end] if data is not None else None

    def _get_item_rnn(self, idx):
        """Retrieve a batch of data for RNN training."""
        gstart = idx * self.num_games_batch
        gend = (idx + 1) * self.num_games_batch
        start = gstart * self.seq_length
        end = gend * self.seq_length
        self.last_range = (start, end)

        input_dict = {k: self._slice_data(v, start, end) for k, v in self.values_dict.items() if k not in self.special_names}
        input_dict['rnn_states'] = [s[:, gstart:gend, :].contiguous() for s in self.values_dict['rnn_states']]
        return input_dict

    def _get_item(self, idx):
        """Retrieve a minibatch of data."""
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)

        input_dict = {k: self._slice_data(v, start, end) for k, v in self.values_dict.items() if k not in self.special_names and v is not None}
        return input_dict

    def __getitem__(self, idx):
        """Retrieve an item based on the dataset type (RNN or not)."""
        return self._get_item_rnn(idx) if self.is_rnn else self._get_item(idx)

    def __len__(self):
        """Return the number of minibatches."""
        return self.length

    def __iter__(self):
        """Iterate over all minibatches in order."""
        for idx in range(self.length):
            yield self[idx]


class DatasetList(Dataset):
    def __init__(self):
        self.dataset_list = []

    def __len__(self):
        return self.dataset_list[0].length * len(self.dataset_list)

    def add_dataset(self, dataset):
        self.dataset_list.append(copy.deepcopy(dataset))

    def clear(self):
        self.dataset_list = []

    def __getitem__(self, idx):
        ds_len = len(self.dataset_list)
        ds_idx = idx % ds_len
        in_idx = idx // ds_len
        return self.dataset_list[ds_idx].__getitem__(in_idx)
