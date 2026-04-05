# DeepMind Control Suite

RL-Games supports [DeepMind Control Suite](https://github.com/google-deepmind/dm_control) environments via [Gymnasium](https://gymnasium.farama.org/) using [Shimmy](https://shimmy.farama.org/environments/dm_control/) compatibility wrappers.

## Installation

```bash
pip install shimmy[dm-control]
```

This installs `dm_control`, `shimmy`, and registers all DM Control environments under the `dm_control/` Gymnasium namespace.

## Single-Agent Environments

Shimmy registers 85 DM Control environments with Gymnasium. Environment IDs follow the pattern `dm_control/{domain}-{task}-v0`:

```python
import gymnasium as gym

env = gym.make("dm_control/cartpole-balance-v0", render_mode="human")
observation, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
env.close()
```

Common environments:
- `dm_control/cartpole-balance-v0`
- `dm_control/cheetah-run-v0`
- `dm_control/walker-walk-v0`
- `dm_control/humanoid-walk-v0`
- `dm_control/humanoid-run-v0`
- `dm_control/acrobot-swingup-v0`
- `dm_control/hopper-hop-v0`
- `dm_control/fish-swim-v0`
- `dm_control/pendulum-swingup-v0`

To list all available environments:
```python
from gymnasium.envs.registration import registry
dm_envs = [env_id for env_id in registry if env_id.startswith("dm_control")]
```

## Multi-Agent: Soccer

DM Control includes a multi-agent soccer environment where teams of agents compete. This uses [PettingZoo](https://pettingzoo.farama.org/) for the multi-agent API via Shimmy.

```bash
pip install shimmy[dm-control-multi-agent]
```

```python
from shimmy import DmControlMultiAgentCompatibilityV0

env = DmControlMultiAgentCompatibilityV0(team_size=2, render_mode="human")
observations = env.reset()
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
env.close()
```

Walker types: `BOXHEAD`, `ANT`, `HUMANOID`.

## Training with RL-Games

TODO: DM Control configs need to be migrated from the legacy envpool format to Gymnasium. New training results and plots will be added.

## Previous Results

See [EnvPool legacy results](ENVPOOL_LEGACY.md) for training results from rl_games <= 1.6.5.
