import torch
import numpy as np


class PolicyPlayerWrapper:
    def __init__(self, policy):
        self.policy = policy
        self.previous_action_distribution = None

    def _batch_obses(self, obses):
        return {
            k: torch.tensor([obs[k] for obs in obses]).cuda() for k in obses[0].keys()
        }

    def __call__(self, obs):
        with torch.no_grad():
            obs = self._batch_obses([obs])
            action_distribution = self.policy(obs)[0]
            action_distribution = torch.nn.functional.softmax(
                action_distribution, dim=-1
            )
            self.previous_action_distribution = action_distribution.cpu().numpy()
            action = torch.multinomial(action_distribution, 1).item()
            return action
