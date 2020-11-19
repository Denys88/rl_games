import torch

class PPODataset(torch.Dataset):

    def __init__(self, batch_size, minibatch_size, is_discrete, is_rnn, device, seq_len):

        self.values_dict = values_dict
        self.is_rnn = is_rnn
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device
        self.length = self.batch_size // self.minibatch_size
        self.is_discrete = is_discrete
        self.is_continuous = not is_discrete
        self.total_games = self.batch_size // self.seq_len
        self.num_games_batch = self.minibatch_size // self.seq_len
        self.game_indexes = torch.arange(total_games, dtype=torch.long, device=self.device)
        self.flat_indexes = torch.arange(total_games * self.seq_len, dtype=torch.long, device=self.device).reshape(total_games, self.seq_len)

    def update_values_dict(self, values_dict):
        self.values_dict = values_dict

    def update_mu_sigma(self, mu, sigma):
        self.self.values_dict['mu'][self.last_range] = mu
        self.self.values_dict['mu'][self.last_range] = sigma
        
    def __len__(self):
        return self.length

    def _get_item_rnn(self, idx):
        batch = torch.range(idx * num_games_batch, (idx + 1) * num_games_batch - 1, dtype=torch.long, device=self.device)
        mb_indexes = self.game_indexes[batch]
        self.last_range = mbatch = self.flat_indexes[mb_indexes].flatten()    
       
        old_values = self.values_dict['old_values']
        old_logp_actions = self.values_dict['old_logp_actions']
        advantages = self.values_dict['advantages']
        returns = self.values_dict['returns']
        actions = self.values_dict['actions']
        obs = self.values_dict['obs']
        rnn_states = self.values_dict['rnn_states']
        rnn_masks = self.values_dict['rnn_masks']

        input_dict['old_values'] = old_values[mbatch]
        input_dict['old_logp_actions'] = old_logp_actions[mbatch]
        input_dict['advantages'] = advantages[mbatch]
        input_dict['returns'] = returns[mbatch]
        input_dict['actions'] = actions[mbatch]
        input_dict['obs'] = obs[mbatch]
        input_dict['rnn_states'] = [s[:,mb_indexes,:] for s in rnn_states]
        input_dict['rnn_masks'] = rnn_masks[mbatch]
        input_dict['learning_rate'] = self.values_dict['learning_rate']

        if self.is_continuous:
            mus = self.values_dict['mu']
            sigmas = self.values_dict['sigma']
            input_dict['mu'] = mus[mbatch]
            input_dict['sigma'] = sigmas[mbatch]
        return input_dict

    def _get_item(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        self.last_range = range(start, end)
        old_values = self.values_dict['old_values']
        old_logp_actions = self.values_dict['old_logp_actions']
        advantages = self.values_dict['advantages']
        returns = self.values_dict['returns']
        actions = self.values_dict['actions']
        obs = self.values_dict['obs']

        input_dict['old_values'] = old_values[start:end]
        input_dict['old_logp_actions'] = old_logp_actions[start:end]
        input_dict['advantages'] = advantages[start:end]
        input_dict['returns'] = returns[start:end]
        input_dict['actions'] = actions[start:end]
        input_dict['obs'] = obs[start:end]
        input_dict['learning_rate'] = self.values_dict['learning_rate']

        if self.is_continuous:
            mus = self.values_dict['mu']
            sigmas = self.values_dict['sigmas']
            input_dict['mu'] = mus[start:end]
            input_dict['sigma'] = sigmas[start:end]

        return input_dict

    def __getitem__(self, idx):
        if self.is_rnn:
            sample = self._get_item_rnn(idx)
        else:
            sample = self._get_item(idx)


        return sample