import numpy as np
class LinearValueProcessor:
    def __init__(self, start_eps, end_eps, end_eps_frames):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.end_eps_frames = end_eps_frames
    
    def __call__(self, frame):
        if frame >= self.end_eps_frames:
            return self.end_eps
        df = frame / self.end_eps_frames
        return df * self.end_eps + (1.0 - df) * self.start_eps


class DefaultRewardsShaper:
    def __init__(self, scale_value = 1, shift_value = 0, min_val=-np.inf, max_val=np.inf):
        self.scale_value = scale_value
        self.shift_value = shift_value
        self.min_val = min_val
        self.max_val = max_val
    def __call__(self, reward):
        reward = reward + self.shift_value
        reward = reward * self.scale_value

        reward = np.clip(reward, self.min_val, self.max_val)
        return reward

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def compute_gae(rewards, dones, values, gamma, tau):
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * tau * (1 -dones[step]) * gae
        returns.append(gae + values[step])
    return returns[::-1]


def flatten_first_two_dims(arr):
    if arr.ndim > 2:
        return arr.reshape(-1, *arr.shape[-(arr.ndim-2):])
    else:
        return arr.reshape(-1)


def get_or_default(config, name, def_val):
    if name in config:
        return config[name]
    else:
        return def_val


def free_mem():
    import ctypes
    ctypes.CDLL('libc.so.6').malloc_trim(0) 