
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



'''
class StepProcessor:
    def __init__(self, env, n_step):

'''   