import torch
import torch.nn as nn
import torch.nn.functional as F
from rl_games.common import networks
from rl_games.common import layers

class MoENet(networks.NetworkBuilder.BaseNetwork):
    def __init__(self, params, **kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        num_inputs = 0

        self.central_value = params.get('central_value', False)
        self.value_size = kwargs.pop('value_size', 1)

        # Parameters from params
        num_experts = params.get('num_experts', 3)
        hidden_size = params.get('hidden_size', 128)
        gating_hidden_size = params.get('gating_hidden_size', 64)
        self.use_sparse_gating = params.get('use_sparse_gating', False)
        self.top_k = params.get('top_k', 1)
        self.lambda_entropy = params.get('lambda_entropy', 0.01)
        self.lambda_diversity = params.get('lambda_diversity', 0.01)

        # Input processing
        assert isinstance(input_shape, dict), "Input shape must be a dict"
        for k, v in input_shape.items():
            num_inputs += v[0]

        # Gating Network
        self.gating_fc1 = nn.Linear(num_inputs, gating_hidden_size)
        self.gating_fc2 = nn.Linear(gating_hidden_size, num_experts)

        # Expert Networks
        self.expert_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            ) for _ in range(num_experts)
        ])

        # Output layers
        self.mean_linear = nn.Linear(hidden_size, actions_num)
        self.value = nn.Linear(hidden_size, self.value_size)

        # Auxiliary loss map
        self.aux_loss_map = {
            'entropy_loss': None,
            'diversity_loss': None,
        }

    def is_rnn(self):
        return False

    def get_aux_loss(self):
        return self.aux_loss_map

    def forward(self, obs_dict):
        # Combine observations
        obs = []
        for k in obs_dict:
            obs.append(obs_dict[k])
        obs = torch.cat(obs, dim=-1)

        # Gating Network Forward Pass
        gating_x = F.relu(self.gating_fc1(obs))
        gating_logits = self.gating_fc2(gating_x)  # Shape: [batch_size, num_experts]
        gating_weights = F.softmax(gating_logits, dim=1)

        # Apply Sparse Gating if enabled
        if self.use_sparse_gating:
            topk_values, topk_indices = torch.topk(gating_weights, self.top_k, dim=1)
            sparse_mask = torch.zeros_like(gating_weights)
            sparse_mask.scatter_(1, topk_indices, topk_values)
            gating_weights = sparse_mask / sparse_mask.sum(dim=1, keepdim=True)  # Re-normalize

        # Compute Entropy Loss for Gating Weights
        entropy = -torch.sum(gating_weights * torch.log(gating_weights + 1e-8), dim=1)
        entropy_loss = torch.mean(entropy)
        self.aux_loss_map['entropy_loss'] = self.lambda_entropy * entropy_loss

        # Expert Networks Forward Pass
        expert_outputs = []
        for expert in self.expert_networks:
            expert_outputs.append(expert(obs))  # Each output shape: [batch_size, hidden_size]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # Shape: [batch_size, num_experts, hidden_size]

        # Compute Diversity Loss
        diversity_loss = 0.0
        num_experts = len(self.expert_networks)
        for i in range(num_experts):
            for j in range(i + 1, num_experts):
                similarity = F.cosine_similarity(expert_outputs[:, i, :], expert_outputs[:, j, :], dim=-1)
                diversity_loss += torch.mean(similarity)
        num_pairs = num_experts * (num_experts - 1) / 2
        diversity_loss = diversity_loss / num_pairs
        self.aux_loss_map['diversity_loss'] = self.lambda_diversity * diversity_loss

        # Aggregate Expert Outputs
        gating_weights = gating_weights.unsqueeze(-1)  # Shape: [batch_size, num_experts, 1]
        aggregated_output = torch.sum(gating_weights * expert_outputs, dim=1)  # Shape: [batch_size, hidden_size]

        out = aggregated_output
        value = self.value_act(self.value(out))
        states = None
        if self.central_value:
            return value, states

        if self.is_discrete:
            logits = self.logits(out)
            return logits, value, states
        if self.is_multi_discrete:
            logits = [logit(out) for logit in self.logits]
            return logits, value, states
        if self.is_continuous:
            mu = self.mu_act(self.mu(out))
            if self.fixed_sigma:
                sigma = self.sigma_act(self.sigma)
            else:
                sigma = self.sigma_act(self.sigma(out))
                return mu, mu*0 + sigma, value, states


from rl_games.algos_torch.network_builder import NetworkBuilder

class MoENetBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        return MoENet(self.params, **kwargs)

    def __call__(self, name, **kwargs):
        return self.build(name, **kwargs)