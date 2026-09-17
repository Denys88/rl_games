import torch
import rl_games.algos_torch.torch_ext as torch_ext


class DefaultDiagnostics(object):
    def __init__(self):
        pass
    def send_info(self, writter):
        pass    
    def epoch(self, agent, current_epoch):
        pass
    def mini_epoch(self, agent, miniepoch):
        pass
    def mini_batch(self, agent, batch, e_clip, minibatch):
        pass


class PpoDiagnostics(DefaultDiagnostics):
    def __init__(self):
        self.diag_dict = {}
        self.clip_fracs = []
        self.exp_vars = []
        self.policy_stats = []
        self.current_epoch = 0

    def send_info(self, writter):
        if writter is None:
            return
        for k, v in self.diag_dict.items():
            writter.add_scalar(k, v.cpu().numpy(), self.current_epoch)

    def epoch(self, agent, current_epoch):
        self.current_epoch = current_epoch
        if agent.normalize_rms_advantage:
            # effective (clamped + eps) normalization stats, not raw EMA moments
            adv_mean, adv_std = agent.advantage_mean_std.get_mean_std()
            self.diag_dict['diagnostics/rms_advantage/mean'] = adv_mean.detach()
            self.diag_dict['diagnostics/rms_advantage/var'] = (adv_std * adv_std).detach()
        if agent.normalize_value:
            self.diag_dict['diagnostics/rms_value/mean'] = agent.value_mean_std.running_mean
            self.diag_dict['diagnostics/rms_value/var'] = agent.value_mean_std.running_var

        exp_var = torch.stack(self.exp_vars, axis=0).mean()
        self.exp_vars = []
        self.diag_dict['diagnostics/exp_var'] = exp_var

    def mini_epoch(self, agent, miniepoch):
        clip_frac = torch.stack(self.clip_fracs, axis=0).mean()
        self.clip_fracs = []
        self.diag_dict['diagnostics/clip_frac/{0}'.format(miniepoch)] = clip_frac
        for name, reduction in (
            ('sigma_min', 'min'), ('sigma_max', 'max'),
            ('mu_abs_max', 'max'), ('advantage_abs_max', 'max'),
            ('logratio_abs_max', 'max'),
        ):
            key = f'diagnostics/policy/{name}/{miniepoch}'
            self.diag_dict.pop(key, None)
            values = [stats[name] for stats in self.policy_stats if name in stats]
            if values:
                self.diag_dict[key] = getattr(torch.stack(values), reduction)()
        key = f'diagnostics/policy/sigma_mean/{miniepoch}'
        self.diag_dict.pop(key, None)
        sigmas = [stats for stats in self.policy_stats if 'sigma_sum' in stats]
        if sigmas:
            self.diag_dict[key] = (
                torch.stack([stats['sigma_sum'] for stats in sigmas]).sum()
                / sum(stats['sigma_count'] for stats in sigmas))
        self.policy_stats = []

    def _record_policy_stats(self, batch, new_neglogp, old_neglogp, masks):
        # Called only by opt-in PpoDiagnostics, under no_grad. Keep scalar
        # snapshots only; never retain minibatch tensors or their graphs.
        valid = masks.reshape(-1).bool() if masks is not None else None

        def rows(tensor):
            tensor = tensor.detach().reshape(new_neglogp.numel(), -1)
            return tensor[valid] if valid is not None else tensor

        stats = {}
        # These log probabilities always reference the collection policy,
        # even when the KL scheduler's mean/std reference is refreshed.
        logratio = rows(old_neglogp).float() - rows(new_neglogp).float()
        if logratio.numel() == 0:
            return  # no valid rows: omit metrics rather than report zeros
        stats['logratio_abs_max'] = logratio.abs().max().cpu()
        if 'sigma' in batch:
            sigma = rows(batch['sigma']).float()
            stats['sigma_min'] = sigma.min().cpu()
            stats['sigma_max'] = sigma.max().cpu()
            stats['sigma_sum'] = sigma.sum().cpu()
            stats['sigma_count'] = sigma.numel()
        for field, name in (('mu', 'mu_abs_max'), ('advantages', 'advantage_abs_max')):
            if field in batch:
                stats[name] = rows(batch[field]).float().abs().max().cpu()
        self.policy_stats.append(stats)

    def mini_batch(self, agent, batch, e_clip, minibatch):
        with torch.no_grad():
            values = batch['values'].detach()
            returns = batch['returns'].detach()
            new_neglogp = batch['new_neglogp'].detach()
            old_neglogp = batch['old_neglogp'].detach()
            masks = batch['masks']
            if masks is not None:
                # production masks are per-row [N]; accept [N, 1] too
                masks = masks.reshape(-1)
            exp_var = torch_ext.explained_variance(values, returns, masks)

            clip_frac = torch_ext.policy_clip_fraction(new_neglogp, old_neglogp, e_clip, masks)
            self.exp_vars.append(exp_var.detach().cpu())
            self.clip_fracs.append(clip_frac.detach().cpu())
            self._record_policy_stats(batch, new_neglogp, old_neglogp, masks)
