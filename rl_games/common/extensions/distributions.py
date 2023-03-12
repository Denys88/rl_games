import torch
import torch.distributions as distr
import torch.functional as F

class CategoricalMaskedNaive(torch.distributions.Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=None):
        self.masks = masks
        if self.masks is None:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            inf_mask = torch.log(masks.float())
            logits = logits + inf_mask
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        if self.masks is None:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p[p_log_p != p_log_p] = 0
        return -p_log_p.sum(-1)


class CategoricalMasked(torch.distributions.Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=None):
        self.masks = masks
        if masks is None:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.device = self.masks.device
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(self.device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def rsample(self):
        u = torch.distributions.Uniform(low=torch.zeros_like(self.logits, device = self.logits.device), high=torch.ones_like(self.logits, device = self.logits.device)).sample()
        #print(u.size(), self.logits.size())
        rand_logits = self.logits -(-u.log()).log()
        return torch.max(rand_logits, axis=-1)[1]

    def entropy(self):
        if self.masks is None:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(self.device))
        return -p_log_p.sum(-1)


class OneHotDist(distr.one_hot_categorical.OneHotCategoricalStraightThrough):

  def __init__(self, logits=None, probs=None, unimix_ratio=0.0):
    if logits is not None and probs is None and unimix_ratio > 0.0:
      probs = F.softmax(logits, dim=-1)
      probs = probs * (1.0-unimix_ratio) + unimix_ratio / probs.shape[-1]
      logits = None
    super().__init__(logits=logits, probs=probs)

  def mode(self):
    _mode = F.one_hot(torch.argmax(super().logits, axis=-1), super().logits.shape[-1])
    return _mode.detach() + super().logits - super().logits.detach()



class TwoHotDist(distr.one_hot_categorical.OneHotCategorical):

  def __init__(self, logits=None, probs=None, unimix_ratio=0.0, device='cuda'):
    if logits is not None and probs is None and unimix_ratio > 0.0:
      probs = F.softmax(logits, dim=-1)
      probs = probs * (1.0-unimix_ratio) + unimix_ratio / probs.shape[-1]
      logits = None
    super().__init__(logits=logits, probs=probs)

    self.buckets = torch.linspace(-20.0, 20.0, steps=255).to(device)
    self.width = (self.buckets[-1] - self.buckets[0]) / 255

  def mode(self):
    _mode = super().probs * self.buckets
    return torch.sum(_mode, dim=-1, keepdim=True)

  # Inside OneHotCategorical, log_prob is calculated using only max element in targets
  def log_prob(self, x):
    # x(time, batch, 1)
    x = (x - self.buckets[0]) / self.width
    lower_indices = (x).to(torch.int64)
    # lower_indices is idnside 0 ~ len(buckets)-2
    lower_indices = torch.clip(lower_indices, max=len(self.buckets)-2)
    # upper_indices is inside 1 ~ len(buckets)-1
    upper_indices = lower_indices + 1
    lower_weight = torch.abs(x - upper_indices).squeeze(-1)
    upper_weight = torch.abs(x - lower_indices).squeeze(-1)
    # (time, batch, 1) -> (time, batch, bucket_class)
    lower_log_prob = super().log_prob(F.one_hot(lower_indices.squeeze(-1), num_classes=len(self.buckets)))
    upper_log_prob = super().log_prob(F.one_hot(upper_indices.squeeze(-1), num_classes=len(self.buckets)))

    # label = lower_log_prob * lower_weight + upper_log_prob * upper_weight
    # # (time, batch, bucket_class) -> (time, batch)
    # cross_entropy = torch.sum(torch.log(super().probs) * label, axis=-1)

    return lower_weight * lower_log_prob + upper_weight * upper_log_prob
  


