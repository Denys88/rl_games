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
    def __init__(self, clip_value = 1, scale_value = 1, shift_value = 0):
        self.clip_value = clip_value
        self.scale_value = scale_value
        self.shift_value = shift_value

    def __call__(self, reward):
        reward = reward + self.shift_value
        reward = reward * self.scale_value
        if (np.abs(reward)) > self.clip_value:
            reward = np.sign(reward) * self.clip_value
        return reward