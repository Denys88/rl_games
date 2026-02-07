# MuJoCo Training with EnvPool

RL-Games supports high-performance vectorized MuJoCo environments via [EnvPool](https://github.com/sail-sg/envpool), providing 3-4x faster environment stepping compared to standard Gym vectorization.

## Installation

Install EnvPool dependency:

**Using Poetry:**
```bash
poetry install -E envpool
```

**Using pip:**
```bash
pip install envpool
```

## Quick Start

Train any MuJoCo environment using EnvPool configs:

```bash
# Humanoid-v4
python runner.py --train --file rl_games/configs/mujoco/humanoid_envpool.yaml

# HalfCheetah-v4
python runner.py --train --file rl_games/configs/mujoco/halfcheetah_envpool.yaml

# Other available configs: hopper, walker2d, ant
```

## Known Issues

### NumPy 2+ Incompatibility (CRITICAL)

**Issue:** NumPy 2.0 and later versions are **NOT compatible** with EnvPool ([see issue](https://github.com/sail-sg/envpool/issues/312)). EnvPool will fail to work correctly with NumPy 2+.

**Required Solution:** You **must** downgrade to NumPy 1.26.4 when using EnvPool:

```bash
pip uninstall numpy
pip install numpy==1.26.4
```

**Symptoms:** Zero observations, environments not training, or unexpected behavior.

**Note:** This is not optional - EnvPool requires NumPy 1.x and will not work with NumPy 2+.

## Training Results

Below are learning curves for standard MuJoCo continuous control benchmarks trained with PPO and EnvPool vectorization:

### HalfCheetah-v4
![HalfCheetah](pictures/mujoco/mujoco_halfcheetah_envpool.png)

### Hopper-v4
![Hopper](pictures/mujoco/mujoco_hopper_envpool.png)

### Walker2d-v4
![Walker2d](pictures/mujoco/mujoco_walker2d_envpool.png)

### Ant-v4
![Ant](pictures/mujoco/mujoco_ant_envpool.png)

### Humanoid-v4
![Humanoid](pictures/mujoco/mujoco_humanoid_envpool.png)

## Performance Notes

- **EnvPool provides 3-4x faster environment stepping** compared to standard Gym vectorization
- Training to convergence typically takes 5-30 minutes on a single GPU (RTX 3090 or equivalent), depending on CPU
- All environments use the same PPO hyperparameters from the config files
