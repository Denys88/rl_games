from copy import deepcopy

import torch
import numpy as np
from tqdm import tqdm

import ray
from torch.utils.tensorboard import SummaryWriter

from model import BaseModel
from bounded_storage import GPUBoundedStorage, convert_storage
from player_wrapper import PolicyPlayerWrapper
from pokerenv_cfr import Action, HeadsUpPoker, ObsProcessor
from cfr_env_wrapper import CFREnvWrapper
from eval_policy import EvalPolicyPlayer

from time_tools import Timers

NUM_WORKERS = 64
BATCH_SIZE = 16384

BOUNDED_STORAGE_MAX_SIZE = 40_000_000


def eval_policy(env, policy, logger, games_to_play=50000):
    player = PolicyPlayerWrapper(policy)

    eval_policy_player = EvalPolicyPlayer(env)
    simple_player_scores = eval_policy_player.eval(player, games_to_play)

    for opponent_name, score in simple_player_scores.items():
        logger.add_scalar(f"policy_evaluation/{opponent_name}/mean_reward", score)


class BatchSampler:
    def __init__(self, bounded_storage):
        self.dicts, self.ts, self.values = bounded_storage.get_storage()

    def __len__(self):
        return len(self.ts)

    def __call__(self, batch_size):
        indices = torch.randint(0, len(self), (batch_size,), device="cuda")

        obs = {k: v[indices] for k, v in self.dicts.items()}
        ts = self.ts[indices]
        values = self.values[indices]

        return obs, ts, values


class MultiBatchSampler(BatchSampler):
    def __call__(self, mini_batches, batch_size):
        indices = torch.randint(
            0,
            len(self),
            (
                mini_batches,
                batch_size,
            ),
            device="cuda",
        )
        obs = {k: v[indices] for k, v in self.dicts.items()}
        ts = self.ts[indices]
        values = self.values[indices]

        for i in range(mini_batches):
            yield {k: v[i] for k, v in obs.items()}, ts[i], values[i]


def train_values(player, samples):
    mini_batches = 4000
    optimizer = torch.optim.Adam(player.parameters(), lr=1e-3)
    for obses, ts, values in MultiBatchSampler(samples)(mini_batches, BATCH_SIZE):
        optimizer.zero_grad()
        value_per_action = player(obses)
        loss = (ts * (value_per_action - values).pow(2)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(player.parameters(), max_norm=1.0)
        optimizer.step()


def train_policy(policy, policy_storage, logger):
    epochs = 50
    learning_rate = 1e-3
    mini_batches = epochs * len(policy_storage) // BATCH_SIZE
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=mini_batches // epochs, gamma=0.9
    )
    multi_batch_sampler = MultiBatchSampler(policy_storage)
    for obses, ts, distributions in multi_batch_sampler(mini_batches, BATCH_SIZE):
        optimizer.zero_grad()
        action_distribution = policy(obses)
        action_distribution = torch.nn.functional.softmax(action_distribution, dim=-1)
        loss = (ts * (action_distribution - distributions).pow(2)).mean()
        logger.add_scalar("policy_training/loss", loss.item(), iter)
        loss.backward()
        optimizer.step()
        scheduler.step()


def regret_matching(values, eps: float = 1e-6):
    values = torch.clamp(values, min=0)
    total = torch.sum(values)
    if total <= eps:
        return torch.ones_like(values) / values.shape[-1]
    return values / total


def _batch_obs(obs):
    batched_obs = {
        k: torch.tensor(obs[k], dtype=torch.int8).unsqueeze(0)
        for k in ["board_and_hand", "stage", "first_to_act_next_stage"]
    }
    batched_obs["bets_and_stacks"] = torch.tensor(
        obs["bets_and_stacks"], dtype=torch.float32
    ).unsqueeze(0)
    return batched_obs


def traverse_cfr(env, player_idx, players, samples_storage, policy_storage, cfr_iter):
    if env.done:
        return env.reward[player_idx]

    obs = env.obs
    batched_obs = _batch_obs(obs)
    if player_idx == obs["player_idx"]:
        values = players[player_idx](batched_obs)[0]
        distribution = regret_matching(values).numpy()
        va = np.zeros(len(Action), dtype=np.float32)
        for action_idx, action in enumerate(Action):
            # avoid a copy of env for the last action
            cfr_env = deepcopy(env) if action_idx + 1 < len(Action) else env
            cfr_env.step(action)
            va[action_idx] = traverse_cfr(
                cfr_env, player_idx, players, samples_storage, policy_storage, cfr_iter
            )
        mean_value_action = np.dot(distribution, va)
        va -= mean_value_action
        samples_storage[player_idx].append((obs, cfr_iter, va))
        return mean_value_action
    else:
        values = players[1 - player_idx](batched_obs)[0]
        distribution = regret_matching(values)
        sampled_action = torch.multinomial(distribution, 1).item()
        policy_storage.append((obs, cfr_iter, distribution.numpy()))
        env.step(sampled_action)
        return traverse_cfr(
            env, player_idx, players, samples_storage, policy_storage, cfr_iter
        )


@ray.remote
def traverses_run(cfr_iter, player_idx, traverses):
    torch.set_num_threads(1)
    value_storage = [[], []]
    policy_storage = []

    players = [BaseModel() for _ in range(2)]
    for idx in range(2):
        players[idx].load_state_dict(
            torch.load(f"/tmp/player_{idx}.pth", weights_only=True, map_location="cpu")
        )
        players[idx].eval()

    env = CFREnvWrapper(HeadsUpPoker(ObsProcessor()))
    with torch.no_grad():
        for _ in range(traverses):
            env.reset()
            traverse_cfr(
                env,
                player_idx,
                players,
                value_storage,
                policy_storage,
                cfr_iter,
            )
    return convert_storage(value_storage[player_idx]), convert_storage(policy_storage)


def save_players(players):
    for idx in range(2):
        torch.save(players[idx].state_dict(), f"/tmp/player_{idx}.pth")


def perform_cfr_iteration(
    cfr_iter,
    num_players,
    traverses_per_iteration,
    timers,
    players,
    samples_storage,
    policy_storage,
    logger,
):
    for player_idx in range(num_players):
        iteration = cfr_iter * num_players + player_idx
        save_players(players)

        timers.start("traverse")
        traverses_per_run = (traverses_per_iteration + NUM_WORKERS - 1) // NUM_WORKERS
        future_results = [
            traverses_run.remote(cfr_iter + 1, player_idx, traverses_per_run)
            for _ in range(NUM_WORKERS)
        ]
        results = ray.get(future_results)
        traverse_time = timers.stop("traverse")
        logger.add_scalar("traverse_time", traverse_time, iteration)

        for value, pol in results:
            samples_storage[player_idx].add_all(value)
            policy_storage.add_all(pol)

        players[player_idx] = BaseModel().cuda()
        timers.start("train values model")
        train_values(players[player_idx], samples_storage[player_idx])
        train_values_model_time = timers.stop("train values model")
        logger.add_scalar("train_values_model_time", train_values_model_time, iteration)

        logger.add_scalar(
            f"samples_storage_size/player_{player_idx}",
            len(samples_storage[player_idx]),
            iteration,
        )


def train_and_eval_policy(env, policy_storage, logger, timers):
    policy = BaseModel().cuda()
    timers.start("train policy")
    train_policy(policy, policy_storage, logger)
    train_policy_time = timers.stop("train policy")
    logger.add_scalar("train_policy_time", train_policy_time, 0)
    torch.save(policy.state_dict(), "policy.pth")

    eval_games = 50000
    eval_policy(env, policy, logger, eval_games)


def deepcfr(cfr_iterations, traverses_per_iteration):
    num_players = 2
    assert num_players == 2

    samples_storage = [
        GPUBoundedStorage(BOUNDED_STORAGE_MAX_SIZE) for _ in range(num_players)
    ]
    policy_storage = GPUBoundedStorage(BOUNDED_STORAGE_MAX_SIZE)

    timers = Timers()
    logger = SummaryWriter()
    players = [BaseModel() for _ in range(num_players)]
    for cfr_iter in tqdm(range(cfr_iterations)):
        perform_cfr_iteration(
            cfr_iter,
            num_players,
            traverses_per_iteration,
            timers,
            players,
            samples_storage,
            policy_storage,
            logger,
        )
        logger.add_scalar("policy_storage_size", len(policy_storage), cfr_iter)

    env = HeadsUpPoker(ObsProcessor())
    policy_storage.save("policy_storage.pt")
    train_and_eval_policy(env, policy_storage, logger, timers)


def policy_training_only():
    timers = Timers()
    logger = SummaryWriter()
    env = HeadsUpPoker(ObsProcessor())
    policy_storage = GPUBoundedStorage(BOUNDED_STORAGE_MAX_SIZE)
    policy_storage.load("policy_storage.pt")
    train_and_eval_policy(env, policy_storage, logger, timers)


if __name__ == "__main__":
    ray.init()

    cfr_iterations = 300
    traverses_per_iteration = 10000
    deepcfr(cfr_iterations, traverses_per_iteration)

    ray.shutdown()
