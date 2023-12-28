"""
Compatibility layer for Gym -> Gymnasium transition.
Adapted from Stable Baselines3, Tianshou, and Shimmy https://github.com/Farama-Foundation/Shimmy
Thanks to @alex-petrenko
"""

import warnings
from inspect import signature
from typing import Union

import gymnasium

try:
    import gym  # pytype: disable=import-error

    gym_installed = True
except ImportError:
    gym_installed = False


def patch_non_gymnasium_env(env: Union["gym.Env", gymnasium.Env]) -> gymnasium.Env:
    env = _patch_env(env)

    try:
        # patching spaces
        if not isinstance(env.observation_space, gymnasium.Space):
            env.observation_space = convert_space(env.observation_space)
        if not isinstance(env.action_space, gymnasium.Space):
            env.action_space = convert_space(env.action_space)
    except AttributeError:
        # gym.Env does not have observation_space and action_space or they're defined as properties
        # in this case... God bless us all
        log.warning("Could not patch spaces for the environment. Consider switching to Gymnasium API.")

    return env


def _patch_env(env: Union["gym.Env", gymnasium.Env]) -> gymnasium.Env:
    """
    Adapted from https://github.com/thu-ml/tianshou.

    Takes an environment and patches it to return Gymnasium env.
    This function takes the environment object and returns a patched
    env, using shimmy wrapper to convert it to Gymnasium,
    if necessary.

    :param env: A gym/gymnasium env
    :return: Patched env (gymnasium env)
    """

    # Gymnasium env, no patching to be done
    if isinstance(env, gymnasium.Env):
        return env

    if not gym_installed or not isinstance(env, gym.Env):
        raise ValueError(
            f"The environment is of type {type(env)}, not a Gymnasium "
            f"environment. In this case, we expect OpenAI Gym to be "
            f"installed and the environment to be an OpenAI Gym environment."
        )

    try:
        import shimmy
    except ImportError as e:
        raise ImportError(
            "Missing shimmy installation. You are using an OpenAI Gym environment. "
            "Sample Factory has transitioned to using Gymnasium internally. "
            "In order to use OpenAI Gym environments with SF, you need to "
            "install shimmy (`pip install 'shimmy>=0.2.1'`)."
        ) from e

    warnings.warn(
        "You provided an OpenAI Gym environment. "
        "We strongly recommend transitioning to Gymnasium environments. "
        "Sample Factory is automatically wrapping your environments in a compatibility "
        "layer, which could potentially cause issues."
    )

    if "seed" in signature(env.unwrapped.reset).parameters:
        # Gym 0.26+ env
        gymnasium_env = shimmy.GymV26CompatibilityV0(env=env)
    else:
        # Gym 0.21 env
        gymnasium_env = shimmy.GymV21CompatibilityV0(env=env)

    # preserving potential multi-agent env attributes
    if hasattr(env, "num_agents"):
        gymnasium_env.num_agents = env.num_agents
    if hasattr(env, "is_multiagent"):
        gymnasium_env.is_multiagent = env.is_multiagent

    return gymnasium_env


def convert_space(space: Union["gym.Space", gymnasium.Space]) -> gymnasium.Space:  # pragma: no cover
    """
    Takes a space and patches it to return Gymnasium Space.
    This function takes the space object and returns a patched
    space, using shimmy wrapper to convert it to Gymnasium,
    if necessary.

    :param space: A gym/gymnasium Space
    :return: Patched space (gymnasium Space)
    """

    # Gymnasium space, no convertion to be done
    if isinstance(space, gymnasium.Space):
        return space

    if not gym_installed or not isinstance(space, gym.Space):
        raise ValueError(
            f"The space is of type {type(space)}, not a Gymnasium "
            f"space. In this case, we expect OpenAI Gym to be "
            f"installed and the space to be an OpenAI Gym space."
        )

    try:
        import shimmy  # pytype: disable=import-error
    except ImportError as e:
        raise ImportError(
            "Missing shimmy installation. You provided an OpenAI Gym space. "
            "Sample Factory has transitioned to using Gymnasium internally. "
            "In order to use OpenAI Gym space with Sample Factory, you need to "
            "install shimmy (`pip install 'shimmy>=0.2.1'`)."
        ) from e

    return shimmy.openai_gym_compatibility._convert_space(space)


    