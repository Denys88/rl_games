import torch
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

        self.special_names = ['rnn_states', 'learning_rate']
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
        #self.last_range = mbatch = self.flat_indexes[mb_indexes].flatten() 
        self.last_range = (start, end)   
        input_dict = {}
        for k,v in self.values_dict.items():
            if k not in self.special_names:
                input_dict[k] = v[start:end]
        
        rnn_states = self.values_dict['rnn_states']
        input_dict['rnn_states'] = [s[:,gstart:gend,:] for s in rnn_states]
        input_dict['learning_rate'] = self.values_dict.get('learning_rate')

        return input_dict

    def _get_item(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = (start, end)
        input_dict = {}
        for k,v in self.values_dict.items():
            if k not in self.special_names and v is not None:
                input_dict[k] = v[start:end]
                
        input_dict['learning_rate'] = self.values_dict.get('learning_rate')

        return input_dict

    def __getitem__(self, idx):
        if self.is_rnn:
            sample = self._get_item_rnn(idx)
        else:
            sample = self._get_item(idx)
        return sample