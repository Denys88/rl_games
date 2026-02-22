import numpy as np
from numpy.random import randint

import os
os.environ.setdefault('PATH', '')
from collections import deque

import gymnasium as gym
from gymnasium import spaces
from copy import copy


def _parse_reset_result(result):
    """Extract obs from gymnasium reset() return of (obs, info)."""
    return result[0]


def _parse_step_result(result):
    """Convert gymnasium 5-tuple step() return to (obs, reward, done, info)."""
    obs, reward, terminated, truncated, info = result
    done = terminated or truncated
    if 'time_outs' not in info:
        info['time_outs'] = truncated
    return obs, reward, done, info


class InfoWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.reward = 0

    def reset(self, **kwargs):
        self.reward = 0
        result = self.env.reset(**kwargs)
        return _parse_reset_result(result)

    def step(self, action):
        observation, reward, done, info = _parse_step_result(self.env.step(action))
        self.reward += reward
        if done:
            info['scores'] = self.reward
        return observation, reward, done, info


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        _parse_reset_result(self.env.reset(**kwargs))
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = _parse_step_result(self.env.step(self.noop_action))
            if done:
                obs = _parse_reset_result(self.env.reset(**kwargs))
        return obs

    def step(self, ac):
        return _parse_step_result(self.env.step(ac))


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        _parse_reset_result(self.env.reset(**kwargs))
        obs, _, done, _ = _parse_step_result(self.env.step(1))
        if done:
            _parse_reset_result(self.env.reset(**kwargs))
        obs, _, done, _ = _parse_step_result(self.env.step(2))
        if done:
            _parse_reset_result(self.env.reset(**kwargs))
        return obs

    def step(self, ac):
        return _parse_step_result(self.env.step(ac))


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on True game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = _parse_step_result(self.env.step(action))
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = _parse_reset_result(self.env.reset(**kwargs))
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = _parse_step_result(self.env.step(0))
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class EpisodeStackedEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.max_stacked_steps = 1000
        self.current_steps = 0

    def step(self, action):
        obs, reward, done, info = _parse_step_result(self.env.step(action))
        if reward == 0:
            self.current_steps += 1
        else:
            self.current_steps = 0
        if self.current_steps == self.max_stacked_steps:
            self.current_steps = 0
            print('max_stacked_steps!')
            done = True
            reward = -1
            obs = _parse_reset_result(self.env.reset())
        return obs, reward, done, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4, use_max=True):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        self.use_max = use_max
        # most recent raw observations (for max pooling across time steps)
        if self.use_max:
            self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.uint8)
        else:
            self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=np.float32)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = _parse_step_result(self.env.step(action))
            if self.use_max:
                if i == self._skip - 2: self._obs_buffer[0] = obs
                if i == self._skip - 1: self._obs_buffer[1] = obs
            else:
                self._obs_buffer[0] = obs

            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        if self.use_max:
            max_frame = self._obs_buffer.max(axis=0)
        else:
            max_frame = self._obs_buffer[0]

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return _parse_reset_result(self.env.reset(**kwargs))


class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, width=84, height=84, grayscale=True):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        if self.grayscale:
            self.observation_space = spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 1), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255,
                shape=(self.height, self.width, 3), dtype=np.uint8)

    def observation(self, frame):
        import cv2
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.grayscale:
            frame = np.expand_dims(frame, -1)
        return frame


class FrameStack(gym.Wrapper):
    def __init__(self, env, k, flat=False):
        """
        Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.flat = flat
        self.frames = deque([], maxlen=k)
        observation_space = env.observation_space
        self.shp = shp = observation_space.shape
        #TODO: remove consts -1 and 1
        if flat:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(shp[:-1] + (shp[-1] * k,)), dtype=observation_space.dtype)
        else:
            if len(shp) == 1:
                self.observation_space = spaces.Box(low=-1, high=1, shape=(k, shp[0]), dtype=observation_space.dtype)
            else:
                self.observation_space = spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=observation_space.dtype)

    def reset(self):
        ob = _parse_reset_result(self.env.reset())
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = _parse_step_result(self.env.step(action))
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        if self.flat:
            return np.squeeze(self.frames).flatten()
        else:
            if len(self.shp) == 1:
                res = np.concatenate([f[..., np.newaxis] for f in self.frames], axis=-1)
                #print('shape:', np.shape(res))
                #print('shape:', np.shape(np.transpose(res)))
                return np.transpose(res)
            else:
                return np.concatenate(self.frames, axis=-1)
        #return LazyFrames(list(self.frames))


class BatchedFrameStack(gym.Wrapper):
    def __init__(self, env, k, transpose=False, flatten=False):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        self.shp = shp = env.observation_space.shape
        self.transpose = transpose
        self.flatten = flatten
        if transpose:
            assert(not flatten)
            self.observation_space = spaces.Box(low=0, high=1, shape=(shp[0], k), dtype=env.observation_space.dtype)
        else:
            if flatten:
                self.observation_space = spaces.Box(low=0, high=1, shape=(k * shp[0],), dtype=env.observation_space.dtype)
            else:
                self.observation_space = spaces.Box(low=0, high=1, shape=(k, shp[0]), dtype=env.observation_space.dtype)

    def reset(self):
        ob = _parse_reset_result(self.env.reset())
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = _parse_step_result(self.env.step(action))
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        if self.transpose:
            frames = np.transpose(self.frames, (1, 2, 0))
        else:
            if self.flatten:
                frames = np.array(self.frames)
                shape = np.shape(frames)
                frames = np.transpose(self.frames, (1, 0, 2))
                frames = np.reshape(self.frames, (shape[1], shape[0] * shape[2]))
            else:
                frames = np.transpose(self.frames, (1, 0, 2))
        return frames


class BatchedFrameStackWithStates(gym.Wrapper):
    def __init__(self, env, k, transpose=False, flatten=False):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.obses = deque([], maxlen=k)
        self.states = deque([], maxlen=k)
        self.shp = shp = env.observation_space.shape
        self.state_shp = state_shp = env.state_space.shape
        self.transpose = transpose
        self.flatten = flatten
        if transpose:
            assert(not flatten)
            self.observation_space = spaces.Box(low=0, high=1, shape=(shp[0], k), dtype=env.observation_space.dtype)
            self.state_space = spaces.Box(low=0, high=1, shape=(state_shp[0], k), dtype=env.observation_space.dtype)
        else:
            if flatten:
                self.observation_space = spaces.Box(low=0, high=1, shape=(k*shp[0],), dtype=env.observation_space.dtype)
                self.state_space = spaces.Box(low=0, high=1, shape=(k*state_shp[0],), dtype=env.observation_space.dtype)
            else:
                self.observation_space = spaces.Box(low=0, high=1, shape=(k, shp[0]), dtype=env.observation_space.dtype)
                self.state_space = spaces.Box(low=0, high=1, shape=(k, state_shp[0]), dtype=env.observation_space.dtype)

    def reset(self):
        obs_dict = _parse_reset_result(self.env.reset())
        ob = obs_dict["obs"]
        state = obs_dict["state"]
        for _ in range(self.k):
            self.obses.append(ob)
            self.states.append(state)
        return self._get_ob()

    def step(self, action):
        obs_dict, reward, done, info = _parse_step_result(self.env.step(action))
        ob = obs_dict["obs"]
        state = obs_dict["state"]
        self.obses.append(ob)
        self.states.append(state)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.obses) == self.k
        obses = self.process_data(self.obses)
        states = self.process_data(self.states)
        return {"obs": obses, "state": states}

    def process_data(self, data):
        if len(np.shape(data)) < 3:
            return np.array(data)
        if self.transpose:
            obses = np.transpose(data, (1, 2, 0))
        else:
            if self.flatten:
                obses = np.array(data)
                shape = np.shape(obses)
                obses = np.transpose(data, (1, 0, 2))
                obses = np.reshape(data, (shape[1], shape[0] * shape[2]))
            else:
                obses = np.transpose(data, (1, 0, 2))
        return obses



class ScaledFloatFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        # careful! This undoes the memory optimization, use
        # with smaller replay buffers only.
        return np.array(observation).astype(np.float32) / 255.0


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class ReallyDoneWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Make it work with video monitor to record whole game video instead of one life
        """
        self.old_env = env
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        old_lives = self.env.unwrapped.ale.lives()
        obs, reward, done, info = _parse_step_result(self.env.step(action))
        lives = self.env.unwrapped.ale.lives()
        if done:
            return obs, reward, done, info
        if old_lives > lives:
            print('lives:', lives)
            obs, _, done, _ = _parse_step_result(self.env.step(1))
        done = lives == 0
        return obs, reward, done, info


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs):
        self._cur_x = 0
        self._max_x = 0
        return _parse_reset_result(self.env.reset(**kwargs))

    def step(self, action):
        obs, rew, done, info = _parse_step_result(self.env.step(action))
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def reset(self):
        self.last_action = 0
        return _parse_reset_result(self.env.reset())

    def step(self, action):
        if self.unwrapped.np_random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = _parse_step_result(self.env.step(action))
        return obs, reward, done, info


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = _parse_step_result(self.env.step(action))
        self.visited_rooms.add(self.get_current_room())
        if done:
            if 'scores' not in info:
                info['scores'] = {}
            info['scores'].update(visited_rooms=copy(self.visited_rooms))
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return _parse_reset_result(self.env.reset())


class TimeLimit(gym.Wrapper):
    """
    A little bit changed original openai's TimeLimit env.
    Main difference is that we always send true or false in infos['time_outs']
    """
    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        self.concat_infos = True
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = _parse_step_result(self.env.step(action))
        self._elapsed_steps += 1
        info['time_outs'] = False
        if self._elapsed_steps >= self._max_episode_steps:
            info['time_outs'] = True
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return _parse_reset_result(self.env.reset(**kwargs))


class ImpalaEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        self.observation_space = spaces.Dict({
            'observation': self.env.observation_space,
            'reward': spaces.Box(low=0, high=1, shape=(), dtype=np.float32),
            'last_action': spaces.Box(low=0, high=self.env.action_space.n, shape=(), dtype=int)
        })

    def step(self, action):
        if not np.isscalar(action):
            action = action.item()
        obs, reward, done, info = _parse_step_result(self.env.step(action))
        obs = {
            'observation': obs,
            'reward': np.clip(reward, -1, 1),
            'last_action': action
        }
        return obs, reward, done, info

    def reset(self):
        obs = _parse_reset_result(self.env.reset())
        obs = {
            'observation': obs,
            'reward': 0.0,
            'last_action': 0
        }
        return obs


class MaskVelocityWrapper(gym.ObservationWrapper):
    """
    Gym environment observation wrapper used to mask velocity terms in
    observations. The intention is the make the MDP partially observatiable.
    """
    def __init__(self, env, name):
        super(MaskVelocityWrapper, self).__init__(env)
        if name == "CartPole-v1":
            self.mask = np.array([1., 0., 1., 0.])
        elif name == "Pendulum-v0":
            self.mask = np.array([1., 1., 0.])
        elif name == "LunarLander-v2":
            self.mask = np.array([1., 1., 0., 0., 1., 0., 1., 1,])
        elif name == "LunarLanderContinuous-v2":
            self.mask = np.array([1., 1., 0., 0., 1., 0., 1., 1,])
        else:
            raise NotImplementedError

    def observation(self, observation):
        return observation * self.mask


class OldGymWrapper(gym.Env):
    """Wrapper to convert gymnasium env to old gym-style 4-tuple API.

    This is useful for environments that use the new gymnasium API
    but need to interface with code expecting the old gym API.
    """
    def __init__(self, env):
        self.env = env
        self.observation_space = self.convert_space(env.observation_space)
        self.action_space = self.convert_space(env.action_space)

    @staticmethod
    def convert_space(space):
        """Recursively convert/copy gymnasium spaces.

        Since we now use gymnasium as gym, this mostly just recreates
        the spaces to ensure they're proper gymnasium spaces.
        """
        if isinstance(space, spaces.Box):
            return spaces.Box(
                low=space.low,
                high=space.high,
                shape=space.shape,
                dtype=space.dtype
            )
        elif isinstance(space, spaces.Discrete):
            return spaces.Discrete(n=space.n)
        elif isinstance(space, spaces.MultiDiscrete):
            return spaces.MultiDiscrete(nvec=space.nvec)
        elif isinstance(space, spaces.MultiBinary):
            return spaces.MultiBinary(n=space.n)
        elif isinstance(space, spaces.Tuple):
            return spaces.Tuple([OldGymWrapper.convert_space(s) for s in space.spaces])
        elif isinstance(space, spaces.Dict):
            return spaces.Dict({k: OldGymWrapper.convert_space(s) for k, s in space.spaces.items()})
        else:
            # Return space as-is if unknown type
            return space

    def reset(self):
        observation = _parse_reset_result(self.env.reset())
        # Flatten the observation if needed
        observation = spaces.flatten(self.observation_space, observation)
        return observation

    def step(self, action):
        # Unflatten the action
        action = spaces.unflatten(self.action_space, action)
        observation, reward, done, info = _parse_step_result(self.env.step(action))
        # Flatten the observation
        observation = spaces.flatten(self.observation_space, observation)
        return observation, reward, done, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


def make_atari(env_id, timelimit=True, noop_max=0, skip=4, sticky=False, directory=None, **kwargs):
    env = gym.make(env_id, **kwargs)
    if 'Montezuma' in env_id:
        env = MontezumaInfoWrapper(env, room_address=3 if 'Montezuma' in env_id else 1)
        env = StickyActionEnv(env)
    env = InfoWrapper(env)
    if directory is not None:
        from gymnasium.wrappers import RecordVideo
        env = RecordVideo(env, video_folder=directory)
    if sticky:
        env = StickyActionEnv(env)
    if not timelimit:
        env = env.env
    if noop_max > 0:
        env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=skip)
    return env


def wrap_deepmind(env, episode_life=False, clip_rewards=True, frame_stack=True, scale =False, wrap_impala=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    if wrap_impala:
        env = ImpalaEnvWrapper(env)
    return env


def wrap_carracing(env, clip_rewards=True, frame_stack=True, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


def make_car_racing(env_id, skip=4):
    env = make_atari(env_id, noop_max=0, skip=skip)
    return wrap_carracing(env, clip_rewards=False)


def make_atari_deepmind(env_id, noop_max=30, skip=4, sticky=False, episode_life=True, wrap_impala=False, **kwargs):
    env = make_atari(env_id, noop_max=noop_max, skip=skip, sticky=sticky, **kwargs)
    return wrap_deepmind(env, episode_life=episode_life, clip_rewards=False, wrap_impala=wrap_impala)
