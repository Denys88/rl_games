import torch
import copy
from torch.utils.data import Dataset

class PPODataset(Dataset):
    def __init__(self, batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len):
        self.is_rnn = is_rnn
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device
        self.length = self.batch_size // self.minibatch_size
        self.is_discrete = is_discrete
        self.is_continuous = not is_discrete
        total_games = self.batch_size // self.seq_len
        self.num_games_batch = self.minibatch_size // self.seq_len
        self.game_indexes = torch.arange(total_games, dtype=torch.long, device=self.device)
        self.flat_indexes = torch.arange(total_games * self.seq_len, dtype=torch.long, device=self.device).reshape(total_games, self.seq_len)

        self.special_names = ['rnn_states']

    def update_values_dict(self, values_dict):
        self.values_dict = values_dict     

    def update_mu_sigma(self, mu, sigma):	    
        start = self.last_range[0]	           
        end = self.last_range[1]	
        self.values_dict['mu'][start:end] = mu	
        self.values_dict['sigma'][start:end] = sigma 

    def __len__(self):
        return self.length

    def _get_item_rnn(self, idx):
        gstart = idx * self.num_games_batch
        gend = (idx + 1) * self.num_games_batch
        start = gstart * self.seq_len
        end = gend * self.seq_len
        self.last_range = (start, end)   
        input_dict = {}
        for k,v in self.values_dict.items():
            if k not in self.special_names:
                if isinstance(v, dict):
                    v_dict = {kd:vd[start:end] for kd, vd in v.items()}
                    input_dict[k] = v_dict
                else:
                    if v is not None:
                        input_dict[k] = v[start:end]
                    else:
                        input_dict[k] = None
        
        rnn_states = self.values_dict['rnn_states']
        input_dict['rnn_states'] = [s[:, gstart:gend, :].contiguous() for s in rnn_states]

        return input_dict

    def _get_item(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        input_dict = {}
        for k,v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                if type(v) is dict:
                    v_dict = { kd:vd[start:end] for kd, vd in v.items() }
                    input_dict[k] = v_dict
                else:
                    input_dict[k] = v[start:end]
                
        return input_dict

    def __getitem__(self, idx):
        if self.is_rnn:
            sample = self._get_item_rnn(idx)
        else:
            sample = self._get_item(idx)
        return sample



class DatasetList(Dataset):
    def __init__(self, datasets = []):
        self.dataset_list = []
        self.dataset_lens = []
        self.len = 0
        self.last_ds_idx = 0
        for d in datasets:
            self.add_dataset(d)
    def __len__(self):
        return self.len

    def deepcopy_dataset(self, dataset):
        self.add_dataset(copy.deepcopy(dataset))

    def add_dataset(self, dataset):
        self.len += len(dataset)
        self.dataset_lens.append(len(dataset))
        self.dataset_list.append(dataset)
    def clear(self):
        self.dataset_list = []
        self.dataset_lens = []
        self.len = 0
        self.last_ds_idx = 0

    def update_mu_sigma(self, mu, sigma):
        self.dataset_list[self.last_ds_idx].update_mu_sigma(mu, sigma)

    def __getitem__(self, idx):
        ds_idx = 0
        in_idx = idx
        while (in_idx >= self.dataset_lens[ds_idx]):
            in_idx -= self.dataset_lens[ds_idx]
            ds_idx += 1
        self.last_ds_idx = ds_idx
        return self.dataset_list[ds_idx].__getitem__(in_idx)


class ReplayBufferDataset(Dataset):
    def __init__(self, replay_buffer, start_idx, end_idx, batch_size):
        self.replay_buffer = replay_buffer
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.batch_size = batch_size
        self.len = (end_idx-start_idx) // batch_size

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        si = idx * self.batch_size
        ei = min((idx + 1) * self.batch_size, self.end_idx)
        return self.replay_buffer.obses[si:ei], self.replay_buffer.actions[si:ei], \
               self.replay_buffer.rewards[si:ei], self.replay_buffer.next_obses[si:ei], \
               self.replay_buffer.dones[si:ei]