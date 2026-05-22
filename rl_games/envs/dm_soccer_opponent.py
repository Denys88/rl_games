"""Frozen-policy opponent for self-play in :mod:`rl_games.envs.dm_soccer`.

Each Ray-worker copy of :class:`DMSoccerAntEnv` holds one of these. When
:class:`rl_games.algos_torch.self_play_manager.SelfPlayManager` decides the
learner is good enough to add to the pool, it calls ``set_weights`` on a
strided subset of envs. That call lands in the env's ``update_weights``,
which delegates here. We append the new state_dict to a bounded ring
buffer; at every episode reset we re-sample an opponent from the buffer
with a recency-weighted distribution (linear weights favour newer
checkpoints). This is the same pattern that worked for the pong self-play
training described in the project memory.

Why a separate class instead of inlining: the network has to be built
*inside* the Ray worker process (torch state + GPU/CPU device live there),
and we need to mirror the learner's exact architecture so ``load_state_dict``
matches key-for-key. Keeping the construction logic here makes the env
wrapper testable without torch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _strip_compile_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the ``_orig_mod.`` prefix that torch.compile prepends."""
    out: Dict[str, Any] = {}
    for k, v in state_dict.items():
        if k.startswith('_orig_mod.'):
            out[k[len('_orig_mod.'):]] = v
        else:
            out[k] = v
    return out


def _to_cpu(obj):
    """Recursively move tensors in a nested dict/list to CPU.

    SelfPlayManager pushes weights through Ray; CUDA tensors don't survive
    cross-process serialization without a host copy first (see project
    memory: pong self-play needed the same fix).
    """
    import torch  # local import — keeps env worker startup lazy

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v) for v in obj)
    return obj


class FrozenA2COpponent:
    """Pool-based frozen actor for self-play.

    Holds up to ``pool_size`` learner snapshots. ``act`` runs the currently
    loaded snapshot in eval mode; ``resample`` re-draws from the pool with
    linear recency weighting. Designed to be called from inside a Ray env
    worker (where the env wrapper actually sees obs and emits actions).

    Args:
        network_params: ``params['network']`` block from the learner YAML.
        model_name: ``params['model']['name']`` (e.g. ``continuous_a2c_logstd``).
        obs_shape: Tuple, the policy obs shape — must match what the
            learner sees. For dm_soccer ant: ``(296,)``.
        actions_num: 8 for the ant.
        normalize_input: Mirror the learner's ``normalize_input`` flag so the
            running_mean_std buffer is part of the model's state_dict.
        normalize_value: Mirror the learner's ``normalize_value`` flag.
        pool_size: Max snapshots kept; oldest is evicted when full.
        deterministic: If True, return the policy mean; else sample.
        device: ``'cpu'`` is fine for opponent inference — it keeps GPU
            free for the learner and avoids cross-device copies for Ray
            workers.
    """

    def __init__(
        self,
        network_params: Dict[str, Any],
        model_name: str,
        obs_shape,
        actions_num: int,
        normalize_input: bool = True,
        normalize_value: bool = False,
        pool_size: int = 8,
        deterministic: bool = True,
        device: str = 'cpu',
    ):
        # Defer torch / rl_games imports until first use — env workers that
        # never receive policy weights (random/noop modes) shouldn't pay
        # the torch import tax.
        import torch  # noqa: WPS433
        from rl_games.algos_torch.model_builder import ModelBuilder  # noqa: WPS433

        self._torch = torch
        self.device = device
        self.deterministic = deterministic
        self.pool_size = int(pool_size)
        self.pool: List[Dict[str, Any]] = []  # ring buffer of state_dicts

        # Build the same model the learner uses. The actual learner does:
        #   builder = ModelBuilder()
        #   self.config['network'] = builder.load(params)        # → BaseModel
        #   self.model = self.network.build(build_config)         # → nn.Module
        # i.e. BaseModel.build composes the network builder + input/value
        # normalizers. We mirror that exact path so state_dict keys line up
        # 1:1 with the learner's checkpoint.
        builder = ModelBuilder()
        self.model = builder.load({'network': network_params, 'model': {'name': model_name}})
        build_config = {
            'actions_num': int(actions_num),
            'input_shape': tuple(obs_shape),
            'num_seqs': 1,
            'value_size': 1,
            'normalize_value': bool(normalize_value),
            'normalize_input': bool(normalize_input),
        }
        self.module = self.model.build(build_config).to(self.device).eval()

    # --- pool management -------------------------------------------------- #

    def add_snapshot(self, state_dict: Dict[str, Any]) -> int:
        """Push a fresh checkpoint into the pool, evicting the oldest if full.

        Returns the new pool size (useful for logging).
        """
        sd = _strip_compile_prefix(_to_cpu(state_dict))
        self.pool.append(sd)
        if len(self.pool) > self.pool_size:
            self.pool.pop(0)
        # Default-load the freshest one so 'act' works immediately even if
        # `resample` isn't called between weight pushes.
        self._load(sd)
        return len(self.pool)

    def load_weights(self, weights: Dict[str, Any]) -> int:
        """SelfPlayManager hands us its full weights dict; we pull 'model'."""
        sd = weights.get('model', weights) if isinstance(weights, dict) else weights
        return self.add_snapshot(sd)

    def resample(self, rng: Optional[np.random.RandomState] = None) -> int:
        """Pick a pool index with linear recency weighting; load that snapshot.

        Returns the chosen index (0 = oldest, len(pool)-1 = newest).
        Linear weights mean a 4-snapshot pool sees probabilities
        [0.1, 0.2, 0.3, 0.4] — newest is 4× as likely as oldest, but the
        oldest is never zeroed out, which keeps the learner from cycling.
        """
        if not self.pool:
            return -1
        rng = rng or np.random
        n = len(self.pool)
        weights = np.arange(1, n + 1, dtype=np.float64)
        weights /= weights.sum()
        idx = int(rng.choice(n, p=weights))
        self._load(self.pool[idx])
        return idx

    def _load(self, state_dict: Dict[str, Any]) -> None:
        # strict=True would catch arch drift between learner/opponent — but
        # the running_mean_std buffer count differs in some rl_games builds;
        # we use strict=False and surface any unexpected keys at debug time.
        missing, unexpected = self.module.load_state_dict(state_dict, strict=False)
        if unexpected:
            # Don't spam every step — print once per push.
            print(f'[FrozenA2COpponent] unexpected keys when loading: {unexpected[:3]}...')

    # --- inference -------------------------------------------------------- #

    def act(self, obs_np: np.ndarray) -> np.ndarray:
        """Map a batch of obs ``(N, obs_dim)`` to a list of N actions.

        Returns a list (not a stacked array) because :class:`DMSoccerAntEnv`
        immediately splays them into the dm_env action list.
        """
        torch = self._torch
        with torch.no_grad():
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
            input_dict = {'obs': obs, 'is_train': False}
            out = self.module(input_dict)
            actions = out['mus'] if self.deterministic else out['actions']
            return list(actions.detach().cpu().numpy())

    @property
    def is_ready(self) -> bool:
        return len(self.pool) > 0


def make_opponent_factory(
    network_params: Dict[str, Any],
    model_name: str,
    obs_shape,
    actions_num: int,
    normalize_input: bool = True,
    normalize_value: bool = False,
    pool_size: int = 8,
    deterministic: bool = True,
    device: str = 'cpu',
):
    """Return a 0-arg factory the env worker can call to instantiate the opponent.

    The env wrapper stores a *factory*, not an instance, because instantiating
    the torch model in the main process and shipping it through Ray would
    duplicate parameters across every worker. By instead sending a
    closure-of-config, each worker builds its own copy on first use.
    """
    def _factory():
        return FrozenA2COpponent(
            network_params=network_params,
            model_name=model_name,
            obs_shape=obs_shape,
            actions_num=actions_num,
            normalize_input=normalize_input,
            normalize_value=normalize_value,
            pool_size=pool_size,
            deterministic=deterministic,
            device=device,
        )
    return _factory
