import torch
import torch.nn as nn
import torch.nn.functional as F


class SwitchFeedForward(nn.Module):

    def __init__(self, 
                 model_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 is_scale_prob: bool,
                 num_experts: int,
                 activation: nn.Module = nn.ReLU

    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model_dim = model_dim
        self.out_dim = out_dim
        self.is_scale_prob = is_scale_prob
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_dim, out_dim),
                activation(),
                #nn.Linear(model_dim, hidden_dim),
                #activation(),
                #nn.Linear(hidden_dim, out_dim),
                #activation(),
            )
            for _ in range(num_experts)
        ])
        # Routing layer and softmax
        self.switch = nn.Linear(model_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x: torch.Tensor):
        route_prob = self.softmax(self.switch(x))
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.num_experts)]
        
        final_output = torch.zeros((x.size(0), self.out_dim), device=x.device)                               
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.num_experts)])

        # Get outputs of the expert FFNs
        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.num_experts)]
        # Assign to final output
        for i in range(self.num_experts):
            final_output[indexes_list[i], :] = expert_output[i]
        
        if self.is_scale_prob:
            # Multiply by the expert outputs by the probabilities $y = p_i(x) E_i(x)$
            final_output = final_output * route_prob_max.view(-1, 1)
        else:
            # not sure if this is correct
            final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)


        return final_output, counts, route_prob.sum(0), route_prob_max



class MoEFF(nn.Module):
    def __init__(self, 
                 model_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 num_experts: int,
                 activation: nn.Module = nn.ReLU,
                 **kwargs
    ):
        super().__init__()

        # Parameters from params
        self.model_dim = model_dim
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.gating_hidden_size = kwargs.get('gating_hidden_size', 64)
        self.use_sparse_gating = kwargs.get('use_sparse_gating', True)
        self.use_entropy_loss = kwargs.get('use_entropy_loss', True)
        self.use_diversity_loss = kwargs.get('use_diversity_loss', True)
        self.top_k = kwargs.get('top_k', 2)
        self.lambda_entropy = kwargs.get('lambda_entropy', 0.01)
        self.lambda_diversity = kwargs.get('lambda_diversity', 0.00)


        # Gating Network
        self.gating_fc1 = nn.Linear(self.model_dim, self.gating_hidden_size)
        self.gating_fc2 = nn.Linear(self.gating_hidden_size, num_experts)

        # Expert Networks
        self.expert_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.model_dim, out_dim),
                activation(),
            ) for _ in range(num_experts)
        ])


        # Auxiliary loss map
        self.aux_loss_map = {
        }
        if self.use_diversity_loss:
            self.aux_loss_map['moe_diversity_loss'] = 0.0
        if self.use_entropy_loss:
            self.aux_loss_map['moe_entropy_loss'] = 0.0

    def get_aux_loss(self):
        return self.aux_loss_map
    
    def forward(self, x):

        # Gating Network Forward Pass
        gating_x = F.relu(self.gating_fc1(x))
        gating_logits = self.gating_fc2(gating_x)  # Shape: [batch_size, num_experts]
        orig_gating_weights = F.softmax(gating_logits, dim=1)
        gating_weights = orig_gating_weights
        # Apply Sparse Gating if enabled
        if self.use_sparse_gating:
            topk_values, topk_indices = torch.topk(gating_weights, self.top_k, dim=1)
            sparse_mask = torch.zeros_like(gating_weights)
            sparse_mask.scatter_(1, topk_indices, topk_values)
            # probably better go with masked softmax
            gating_weights = sparse_mask / sparse_mask.sum(dim=1, keepdim=True)  

        if self.use_entropy_loss:
        # Compute Entropy Loss for Gating Weights
            entropy = -torch.sum(gating_weights * torch.log(gating_weights + 1e-8), dim=1)
            entropy_loss = torch.mean(entropy)
            self.aux_loss_map['moe_entropy_loss'] = -self.lambda_entropy * entropy_loss

        # Expert Networks Forward Pass
        expert_outputs = []
        for expert in self.expert_networks:
            expert_outputs.append(expert(x))  # Each output shape: [batch_size, hidden_size]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # Shape: [batch_size, num_experts, hidden_size]

        # Compute Diversity Loss
        if self.use_diversity_loss:
            diversity_loss = 0.0
            num_experts = len(self.expert_networks)
            for i in range(num_experts):
                for j in range(i + 1, num_experts):
                    similarity = F.cosine_similarity(expert_outputs[:, i, :], expert_outputs[:, j, :], dim=-1)
                    diversity_loss += torch.mean(similarity)
            num_pairs = num_experts * (num_experts - 1) / 2
            diversity_loss = diversity_loss / num_pairs
            self.aux_loss_map['moe_diversity_loss'] = self.lambda_diversity * diversity_loss

        # Aggregate Expert Outputs
        gating_weights = gating_weights.unsqueeze(-1)  # Shape: [batch_size, num_experts, 1]
        aggregated_output = torch.sum(gating_weights * expert_outputs, dim=1)  # Shape: [batch_size, hidden_size]
        out = aggregated_output
        return out


class MoEBlock(nn.Module):
    def __init__(self,
                 input_size: int,
                 model_units: list[int],
                 expert_units: list[int],
                 num_experts: int,
    ):
        super().__init__()
        self.num_experts = num_experts
        in_size = input_size
        layers =[]
        for u, m in zip(expert_units, model_units):
            layers.append(MoEFF(in_size, m, u, num_experts))
            in_size = u
        self.layers = nn.ModuleList(layers)
        self.load_balancing_loss = None

    def get_aux_loss(self):
        return {
            "moe_load_balancing_loss": self.load_balancing_loss
        }

    def forward(self, x: torch.Tensor):
        moe_diversity_loss, moe_entropy_loss = 0, 0
        for layer in self.layers:
            x = layer(x)
            moe_diversity_loss = moe_diversity_loss + layer.get_aux_loss()['moe_diversity_loss']
            moe_entropy_loss = moe_diversity_loss + layer.get_aux_loss()['moe_entropy_loss']
  
        self.load_balancing_loss = moe_diversity_loss / len(self.layers) + moe_entropy_loss / len(self.layers)
        return x

'''
    def forward(self, x: torch.Tensor):
        counts, route_prob_sums, route_prob_maxs = [], [], []
        for layer in self.layers:
            x, count, route_prob_sum, route_prob_max = layer(x)
            counts.append(count)
            route_prob_sums.append(route_prob_sum)
            route_prob_maxs.append(route_prob_max)
            
        counts = torch.stack(counts)
        route_prob_sums = torch.stack(route_prob_sums) 
        route_prob_maxs = torch.stack(route_prob_maxs)

        total = counts.sum(dim=-1, keepdims=True)
        route_frac = counts / total
        route_prob = route_prob_sums / total
  
        self.load_balancing_loss = self.num_experts * (route_frac * route_prob).sum()
        return x
'''