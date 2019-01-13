import cv2
import gym
import gym.spaces
import numpy as np
import collections
from skimage import color
from scipy.misc import imresize

class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        """For environments where the user need to press FIRE for the game to start."""
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class AtariResetLive(gym.Wrapper):
    """
    Wraps an Atari environment to end an episode when a life is lost.
    """
    def __init__(self, env=None):
        super(AtariResetLive, self).__init__(env)
        self.step_info = None

    def lives(self):
        if self.step_info is None:
            return 0
        else:
            return self.step_info['ale.lives']

    def step(self, action):
        lifes_before = self.lives()
        next_state, reward, done, self.step_info = self.env.step(action)
        lifes_after = self.lives()
        if lifes_before > lifes_after:
            done = True
        return next_state, reward, done, self.step_info 


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        #img = img[:, :, 0] * 0.333 + img[:, :, 1] * 0.333 + img[:, :, 2] * 0.333
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)

class ProcessFrame64(gym.ObservationWrapper):
    def __init__(self, env):
        """A gym wrapper that crops, scales image into the desired shapes and optionally grayscales it."""
        gym.ObservationWrapper.__init__(self,env)
        
        self.img_size = (64, 64)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)

    def observation(self, img):
        """what happens to each observation"""
        
        # Here's what you need to do:
        #  * crop image, remove irrelevant parts
        #  * resize image to self.img_size 
        #     (use imresize imported above or any library you want,
        #      e.g. opencv, skimage, PIL, keras)
        #  * cast image to grayscale
        #  * convert image pixels to (0,1) range, float32 type
        cropped = img[35:195, 0 : 160, :]
        cropped = imresize(cropped, self.img_size)
        cropped = color.rgb2gray(cropped).astype('float32')
        
        return np.expand_dims(cropped, axis=2) 

class ImageToTF(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToTF, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(old_space.low.repeat(n_steps, axis=0),
                                                old_space.high.repeat(n_steps, axis=0), dtype=dtype)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env(env_name):
    env = gym.make(env_name)
    env = FireResetEnv(env)
    env = AtariResetLive(env)
    #env = MaxAndSkipEnv(env)
    
    env = ProcessFrame84(env)
    env = ImageToTF(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)

def make_env_with_monitor(env_name, folder):
    
    env = gym.make(env_name)
    env = FireResetEnv(env)
    env = AtariResetLive(env)
    env = gym.wrappers.Monitor(env,directory=folder,force=True)
    #env = MaxAndSkipEnv(env)
    
    env = ProcessFrame84(env)
    env = ImageToTF(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)
