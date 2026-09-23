"""Frozen-teacher distillation inside PPO (DAgger as a loss term).

The teacher is a frozen rl_games policy that consumes privileged `states`;
the student consumes `obs`. Three piecewise-linear schedules over epochs:

  coef      weight of the distillation loss on student-visited states
  ppo_coef  weight of the PPO loss (0 -> pure DAgger phase)
  beta      P(env action taken from the teacher) — classic DAgger mixing;
            the student's neglogp is recomputed for substituted rows so PPO
            importance ratios stay valid.

Config (algo `config.distillation`):

    distillation:
      teacher_config: <yaml with the teacher's params>
      teacher_checkpoint: <rl_games .pth>
      loss: kl                 # discrete: KL(teacher||student); continuous: kl | mse
      coef: [[0, 1.0], [400, 1.0], [800, 0.0]]
      ppo_coef: [[0, 0.0], [400, 0.0], [401, 1.0]]
      beta: [[0, 0.5], [200, 0.0]]
      warm_start_value: True   # copy matching teacher tensors into the central value net
      teacher_mu_clip: 1.0     # continuous only: clamp teacher means to the action range

Because PPO's on-policy buffer is rebuilt from the student's own rollouts
every epoch, the distillation term IS DAgger: supervised on the states the
student visits. With ppo_coef 0 it is pure DAgger; ramping ppo_coef up and
coef down gives DAgger -> PPO fine-tune; both on gives teacher-guided RL.
"""

import numpy as np
import torch
import yaml


def piecewise_linear(spec, epoch):
    """[[epoch, value], ...] -> value at `epoch` (linear between knots,
    constant outside). A scalar spec is a constant; None is 0."""
    if spec is None:
        return 0.0
    if isinstance(spec, (int, float)):
        return float(spec)
    knots = sorted((float(e), float(v)) for e, v in spec)
    if epoch <= knots[0][0]:
        return knots[0][1]
    for (e0, v0), (e1, v1) in zip(knots[:-1], knots[1:]):
        if e0 <= epoch <= e1:
            if e1 == e0:
                return v1
            return v0 + (v1 - v0) * (epoch - e0) / (e1 - e0)
    return knots[-1][1]


def _strip_prefix(sd, prefix='_orig_mod.'):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}


class TeacherDistillation:
    def __init__(self, config, teacher_model, is_discrete, device):
        self.config = config
        self.teacher = teacher_model.to(device).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.is_discrete = is_discrete
        self.device = device
        self.loss_type = config.get('loss', 'kl')
        if self.loss_type not in ('kl', 'mse'):
            raise ValueError("distillation loss must be 'kl' or 'mse'")
        self._coef = config.get('coef', 1.0)
        self._ppo_coef = config.get('ppo_coef', 1.0)
        self._beta = config.get('beta', 0.0)
        self.warm_start_value = bool(config.get('warm_start_value', True))
        # continuous: clamp the teacher's means to +-clip before the loss (the env
        # clips actions anyway; unbounded teacher means are unreachable targets)
        self.teacher_mu_clip = config.get('teacher_mu_clip', None)
        self.last_loss = None

    # ----------------------------------------------------------- schedules

    def coef(self, epoch):
        return piecewise_linear(self._coef, epoch)

    def ppo_coef(self, epoch):
        return piecewise_linear(self._ppo_coef, epoch)

    def beta(self, epoch):
        return piecewise_linear(self._beta, epoch)

    # ------------------------------------------------------------- teacher

    @torch.no_grad()
    def teacher_forward(self, states):
        return self.teacher({'obs': states, 'is_train': False})

    @classmethod
    def from_config(cls, config, state_shape, actions_num, device):
        """Build the frozen teacher from its own yaml + checkpoint."""
        from rl_games.algos_torch.model_builder import ModelBuilder
        params = yaml.safe_load(open(config['teacher_config']))['params']
        net = ModelBuilder().load(params)
        tcfg = params['config']
        model = net.build({'actions_num': actions_num, 'input_shape': tuple(state_shape), 'num_seqs': 1,
                           'value_size': tcfg.get('value_size', 1),
                           'normalize_value': tcfg.get('normalize_value', False),
                           'normalize_input': tcfg.get('normalize_input', False)})
        ckpt = torch.load(config['teacher_checkpoint'], map_location=device, weights_only=False)
        model.load_state_dict(_strip_prefix(ckpt['model']))
        is_discrete = params['algo']['name'] == 'a2c_discrete'
        print(f"[Distillation] teacher {config['teacher_checkpoint']} (epoch {ckpt.get('epoch', '?')}), "
              f"{'discrete' if is_discrete else 'continuous'}, coef {config.get('coef')}, "
              f"ppo_coef {config.get('ppo_coef')}, beta {config.get('beta')}")
        return cls(config, model, is_discrete, device)

    # ---------------------------------------------------------------- loss

    def loss(self, res_dict, states, rnn_masks=None):
        """Mean distillation loss over rows (masked when rnn_masks given)."""
        t = self.teacher_forward(states)
        if self.is_discrete:
            logp_t = torch.log_softmax(t['logits'].float(), dim=-1)
            logp_s = torch.log_softmax(res_dict['logits'].float(), dim=-1)
            per_row = (logp_t.exp() * (logp_t - logp_s)).sum(dim=-1)
        else:
            mu_s, sig_s = res_dict['mus'].float(), res_dict['sigmas'].float()
            mu_t, sig_t = t['mus'].float(), t['sigmas'].float()
            if self.teacher_mu_clip is not None:
                c = float(self.teacher_mu_clip)
                mu_t = mu_t.clamp(-c, c)
            if self.loss_type == 'mse':
                per_row = ((mu_t - mu_s) ** 2).sum(dim=-1)
            else:
                per_row = (torch.log(sig_s / sig_t)
                           + (sig_t ** 2 + (mu_t - mu_s) ** 2) / (2 * sig_s ** 2) - 0.5).sum(dim=-1)
        if rnn_masks is not None:
            m = rnn_masks.reshape(-1).float()
            out = (per_row * m).sum() / m.sum().clamp(min=1.0)
        else:
            out = per_row.mean()
        self.last_loss = out.detach()
        return out

    # ------------------------------------------------------------- mixing

    def mix_actions(self, res_dict, states, epoch):
        """DAgger beta-mixing on the rollout batch: substitute the teacher's
        sampled action for a Bernoulli(beta) subset of rows and recompute the
        student's neglogp for those rows."""
        beta = self.beta(epoch)
        if beta <= 0.0:
            return res_dict
        n = states.shape[0]
        sel = torch.rand(n, device=states.device) < beta
        if not bool(sel.any()):
            return res_dict
        t = self.teacher_forward(states[sel])
        actions = res_dict['actions'].clone()
        if self.is_discrete:
            ta = torch.distributions.Categorical(logits=t['logits'].float()).sample()
            actions[sel] = ta.to(actions.dtype)
            neglogp = -torch.log_softmax(res_dict['logits'].float(), dim=-1).gather(
                1, actions.view(-1, 1).long()).squeeze(1)
        else:
            ta = torch.distributions.Normal(t['mus'].float(), t['sigmas'].float()).sample()
            actions[sel] = ta.to(actions.dtype)
            mus, sig = res_dict['mus'].float(), res_dict['sigmas'].float()
            neglogp = 0.5 * (((actions.float() - mus) / sig) ** 2).sum(-1) \
                + 0.5 * np.log(2.0 * np.pi) * actions.shape[-1] + torch.log(sig).sum(-1)
        res_dict['actions'] = actions
        res_dict['neglogpacs'] = neglogp.to(res_dict['neglogpacs'].dtype)
        return res_dict

    # ---------------------------------------------------------- warm start

    def warm_start_central_value(self, cv_model):
        """Copy every teacher tensor whose name and shape match into the
        central-value model (trunk, value head, input/value normalisers)."""
        tsd = _strip_prefix(self.teacher.state_dict())
        csd = cv_model.state_dict()
        new, copied, skipped = {}, 0, 0
        for k, v in csd.items():
            if k in tsd and tuple(tsd[k].shape) == tuple(v.shape):
                new[k] = tsd[k].to(v.device, v.dtype)
                copied += 1
            else:
                skipped += 1
        cv_model.load_state_dict(new, strict=False)
        print(f'[Distillation] warm-started central value: {copied} tensors copied, {skipped} skipped')
        return copied, skipped

    # ------------------------------------------------------------- logging

    def log(self, writer, epoch, frame):
        if writer is None:
            return
        writer.add_scalar('info/distill_coef', self.coef(epoch), frame)
        writer.add_scalar('info/ppo_coef', self.ppo_coef(epoch), frame)
        writer.add_scalar('info/teacher_beta', self.beta(epoch), frame)
