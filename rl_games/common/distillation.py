"""Frozen-teacher distillation inside PPO (DAgger as a loss term).

The teacher is a frozen rl_games policy that consumes privileged `states`;
the student consumes `obs`. Three piecewise-linear schedules over epochs:

  coef      weight of the distillation loss on student-visited states
  ppo_coef  weight of the PPO loss (0 -> pure DAgger phase)
  beta      P(env action taken from the teacher) — classic DAgger mixing.
            Substituted rows are flagged `from_teacher` and carry no actor
            loss: those actions were not sampled from the student, so a PPO
            term on them would be advantage-weighted behaviour cloning, not
            a policy gradient. They still train the critic and the
            distillation term.

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

ppo_coef scales the policy part of the PPO loss only (actor, entropy, bounds);
the critic keeps fitting the student's returns during the DAgger phase, so it
is not cold when PPO switches on. With a central value net this is moot: the
central value trains on its own and `warm_start_value` seeds it from the
teacher.

The teacher runs once per epoch over the rollout batch (`targets`); the
minibatches carry its logits / means, so the update never calls it.

Continuous `kl` also pulls the student's sigma to the teacher's (usually
annealed small) and scales the mean gradient by 1 / sigma_s^2; `mse` matches
the means only and leaves sigma to PPO.

Limits: the teacher is feedforward (no rnn states are passed to it), the
student's action space must be the teacher's (discrete <-> discrete or
continuous <-> continuous), multi-discrete, multi-agent envs and Dict state
spaces are not supported.
"""

import os

import numpy as np
import torch
import yaml

# the fill CategoricalMasked uses for invalid actions (finite, so a row with
# no valid action stays a proper distribution)
MASKED_LOGIT = -1e8
TARGET_KEYS = ('teacher_logits', 'teacher_mus', 'teacher_sigmas')


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


def _schedule(config, name, default):
    """Read and validate a schedule: a scalar or [[epoch, value], ...]. A key
    that is present but empty (yaml `ppo_coef:`) is an error, not 0."""
    if name not in config:
        return default
    spec = config[name]
    if isinstance(spec, bool) or spec is None:
        raise ValueError(f"distillation.{name} must be a number or [[epoch, value], ...], got {spec!r}")
    if isinstance(spec, (int, float)):
        return float(spec)
    try:
        knots = [(float(e), float(v)) for e, v in spec]
    except (TypeError, ValueError):
        knots = []
    if not knots:
        raise ValueError(f"distillation.{name} must be a number or [[epoch, value], ...], got {spec!r}")
    return knots


def _last_knot(spec):
    return 0 if isinstance(spec, (int, float)) else int(max(e for e, _ in spec))


def _resolve_path(path):
    """A relative path that does not exist from the working directory is
    tried against the repository root (configs name `rl_games/configs/...`)."""
    path = os.path.expanduser(str(path))
    if os.path.isabs(path) or os.path.exists(path):
        return path
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate = os.path.join(root, path)
    return candidate if os.path.exists(candidate) else path


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
        self._coef = _schedule(config, 'coef', 1.0)
        self._ppo_coef = _schedule(config, 'ppo_coef', 1.0)
        self._beta = _schedule(config, 'beta', 0.0)
        self.warm_start_value = bool(config.get('warm_start_value', True))
        # continuous: clamp the teacher's means to +-clip before the loss (the env
        # clips actions anyway; unbounded teacher means are unreachable targets)
        self.teacher_mu_clip = config.get('teacher_mu_clip', None)
        # rows per teacher forward when the whole rollout batch is labelled
        self.target_chunk = int(config.get('target_chunk', 16384))

    # ----------------------------------------------------------- schedules

    def coef(self, epoch):
        return piecewise_linear(self._coef, epoch)

    def ppo_coef(self, epoch):
        return piecewise_linear(self._ppo_coef, epoch)

    def beta(self, epoch):
        return piecewise_linear(self._beta, epoch)

    def has_pure_distillation_phase(self):
        """True if some epoch has the distillation term on and PPO off."""
        last = max(_last_knot(s) for s in (self._coef, self._ppo_coef))
        return any(self.coef(e) > 0 and self.ppo_coef(e) <= 0 for e in range(last + 2))

    # ------------------------------------------------------------- teacher

    @torch.no_grad()
    def teacher_forward(self, states):
        return self.teacher({'obs': states, 'is_train': False})

    def _clip_mu(self, mu):
        if self.teacher_mu_clip is None:
            return mu
        c = float(self.teacher_mu_clip)
        return mu.clamp(-c, c)

    @torch.no_grad()
    def targets(self, states):
        """The teacher's outputs for `states` as fp32 tensors keyed like the
        minibatch entries: `teacher_logits`, or `teacher_mus` (clipped to
        `teacher_mu_clip`) and `teacher_sigmas`."""
        out = {}
        for start in range(0, states.shape[0], self.target_chunk):
            t = self.teacher_forward(states[start:start + self.target_chunk])
            if self.is_discrete:
                part = {'teacher_logits': self._logits(t)}
            else:
                part = {'teacher_mus': self._clip_mu(t['mus'].float()), 'teacher_sigmas': t['sigmas'].float()}
            for k, v in part.items():
                out.setdefault(k, []).append(v)
        return {k: torch.cat(v, dim=0) for k, v in out.items()}

    @classmethod
    def from_config(cls, config, state_shape, actions_num, device, is_discrete=None):
        """Build the frozen teacher from its own yaml + checkpoint. `is_discrete`
        is the student's action space; it must match the teacher's."""
        from rl_games.algos_torch.model_builder import ModelBuilder
        with open(_resolve_path(config['teacher_config'])) as f:
            params = yaml.safe_load(f)['params']
        teacher_algo = params['algo']['name']
        if teacher_algo not in ('a2c_discrete', 'a2c_continuous'):
            raise ValueError(f"distillation teacher must be a2c_discrete or a2c_continuous, got '{teacher_algo}'")
        teacher_discrete = teacher_algo == 'a2c_discrete'
        if is_discrete is not None and bool(is_discrete) != teacher_discrete:
            raise ValueError("distillation teacher and student must share the action space type "
                             f"(teacher {teacher_algo}, student {'discrete' if is_discrete else 'continuous'})")
        net = ModelBuilder().load(params)
        tcfg = params['config']
        model = net.build({'actions_num': actions_num, 'input_shape': tuple(state_shape), 'num_seqs': 1,
                           'value_size': tcfg.get('value_size', 1),
                           'normalize_value': tcfg.get('normalize_value', False),
                           'normalize_input': tcfg.get('normalize_input', False)})
        if model.is_rnn():
            raise NotImplementedError('distillation teacher must be feedforward: the teacher gets no rnn states')
        ckpt = torch.load(_resolve_path(config['teacher_checkpoint']), map_location=device, weights_only=False)
        model.load_state_dict(_strip_prefix(ckpt['model']))
        print(f"[Distillation] teacher {config['teacher_checkpoint']} (epoch {ckpt.get('epoch', '?')}), "
              f"{'discrete' if teacher_discrete else 'continuous'}, coef {config.get('coef')}, "
              f"ppo_coef {config.get('ppo_coef')}, beta {config.get('beta')}")
        return cls(config, model, teacher_discrete, device)

    # ---------------------------------------------------------------- loss

    @staticmethod
    def _logits(d, action_masks=None):
        logits = d['logits']
        if isinstance(logits, (list, tuple)):
            raise NotImplementedError('distillation does not support multi-discrete action spaces')
        logits = logits.float()
        if action_masks is not None:
            # invalid actions get CategoricalMasked's fill, as for the student;
            # unmasked, KL(teacher || student) blows up wherever the student
            # has no mass
            logits = logits.masked_fill(~action_masks.bool(), MASKED_LOGIT)
        return logits

    def loss(self, res_dict, targets, rnn_masks=None, action_masks=None):
        """Mean distillation loss over rows (masked when rnn_masks given).
        `targets` is a dict holding the teacher's outputs (a minibatch, see
        `targets()`), or the states tensor to label on the spot.
        `action_masks` (discrete, bool [rows, actions]) restrict both policies
        to the valid actions."""
        if torch.is_tensor(targets):
            targets = self.targets(targets)
        if self.is_discrete:
            logp_t = torch.log_softmax(self._logits({'logits': targets['teacher_logits']}, action_masks), dim=-1)
            logp_s = torch.log_softmax(self._logits(res_dict, action_masks), dim=-1)
            p_t = logp_t.exp()
            # masked actions have p_t == 0 and contribute nothing
            per_row = torch.where(p_t > 0, p_t * (logp_t - logp_s), torch.zeros_like(p_t)).sum(dim=-1)
        else:
            mu_s, sig_s = res_dict['mus'].float(), res_dict['sigmas'].float()
            mu_t, sig_t = targets['teacher_mus'].float(), targets['teacher_sigmas'].float()
            if self.loss_type == 'mse':
                per_row = ((mu_t - mu_s) ** 2).sum(dim=-1)
            else:
                per_row = (torch.log(sig_s / sig_t)
                           + (sig_t ** 2 + (mu_t - mu_s) ** 2) / (2 * sig_s ** 2) - 0.5).sum(dim=-1)
        if rnn_masks is not None:
            m = rnn_masks.reshape(-1).float()
            return (per_row * m).sum() / m.sum().clamp(min=1.0)
        return per_row.mean()

    def combine(self, ppo_loss, value_loss, d_loss, epoch):
        """Total minibatch loss. ppo_coef scales the policy part of the PPO
        loss only; the critic term (`value_loss`, already weighted, part of
        `ppo_loss`) stays on so the value head keeps learning when ppo_coef
        is 0."""
        total = self.ppo_coef(epoch) * (ppo_loss - value_loss) + value_loss
        coef = self.coef(epoch)
        # at coef 0 the term is monitoring only: keep it out of the graph, so
        # a non-finite distillation loss cannot reach the update
        return total + coef * d_loss if coef > 0 else total

    @staticmethod
    def student_rows(input_dict):
        """1 for rows whose action the student sampled, 0 for teacher rows."""
        return 1.0 - input_dict['from_teacher'].float()

    # ------------------------------------------------------------- mixing

    def mix_actions(self, res_dict, states, epoch, action_masks=None):
        """DAgger beta-mixing on the rollout batch: substitute the teacher's
        sampled action for a Bernoulli(beta) subset of rows. Those rows are
        flagged in `res_dict['from_teacher']` (always set) and excluded from
        the actor loss; the student's neglogp is still recomputed for them so
        the stored value is the student's for the action taken."""
        n = states.shape[0]
        res_dict['from_teacher'] = torch.zeros(n, dtype=torch.float32, device=states.device)
        beta = self.beta(epoch)
        if beta <= 0.0:
            return res_dict
        sel = torch.rand(n, device=states.device) < beta
        if not bool(sel.any()):
            return res_dict
        t = self.teacher_forward(states[sel])
        actions = res_dict['actions'].clone()
        if self.is_discrete:
            t_masks = action_masks[sel] if action_masks is not None else None
            ta = torch.distributions.Categorical(logits=self._logits(t, t_masks)).sample()
            actions[sel] = ta.to(actions.dtype)
            neglogp = -torch.log_softmax(self._logits(res_dict, action_masks), dim=-1).gather(
                1, actions.view(-1, 1).long()).squeeze(1)
        else:
            ta = torch.distributions.Normal(self._clip_mu(t['mus'].float()), t['sigmas'].float()).sample()
            actions[sel] = ta.to(actions.dtype)
            mus, sig = res_dict['mus'].float(), res_dict['sigmas'].float()
            neglogp = 0.5 * (((actions.float() - mus) / sig) ** 2).sum(-1) \
                + 0.5 * np.log(2.0 * np.pi) * actions.shape[-1] + torch.log(sig).sum(-1)
        res_dict['actions'] = actions
        res_dict['neglogpacs'] = neglogp.to(res_dict['neglogpacs'].dtype)
        res_dict['from_teacher'] = sel.float()
        return res_dict

    # ---------------------------------------------------------- warm start

    def warm_start_central_value(self, cv_model):
        """Copy every teacher tensor whose name and shape match into the
        central-value model (trunk, value head, input/value normalisers).

        A `separate: True` teacher keeps its critic in `critic_mlp` /
        `critic_cnn` while a central-value net has one trunk named
        `actor_*`: the critic trunk is copied there. Matching by name alone
        would put the teacher's value head on its actor trunk."""
        tsd = _strip_prefix(self.teacher.state_dict())
        csd = _strip_prefix(cv_model.state_dict())
        critic_parts = ('.critic_mlp.', '.critic_cnn.')
        actor_parts = ('.actor_mlp.', '.actor_cnn.')
        if any(p in k for k in tsd for p in critic_parts) and not any(p in k for k in csd for p in critic_parts):
            tsd = {k: v for k, v in tsd.items() if not any(p in k for p in actor_parts)}
            for k in list(tsd):
                for c, a in zip(critic_parts, actor_parts):
                    if c in k:
                        tsd[k.replace(c, a)] = tsd.pop(k)
        new, skipped = {}, []
        for k, v in csd.items():
            if k in tsd and tuple(tsd[k].shape) == tuple(v.shape):
                new[k] = tsd[k].to(v.device, v.dtype)
            else:
                skipped.append(k)
        # load through the (possibly compiled) module's own key names
        prefix = '_orig_mod.' if any(k.startswith('_orig_mod.') for k in cv_model.state_dict()) else ''
        cv_model.load_state_dict({prefix + k: v for k, v in new.items()}, strict=False)
        print(f'[Distillation] warm-started central value: {len(new)} tensors copied, {len(skipped)} skipped'
              + (f' ({", ".join(skipped)})' if skipped else ''))
        return len(new), len(skipped)

    # ------------------------------------------------------------- logging

    def log(self, writer, epoch, frame):
        if writer is None:
            return
        writer.add_scalar('info/distill_coef', self.coef(epoch), frame)
        writer.add_scalar('info/ppo_coef', self.ppo_coef(epoch), frame)
        writer.add_scalar('info/teacher_beta', self.beta(epoch), frame)
