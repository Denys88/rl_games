from rl_games.envs.poker.poker_env import HeadsUpPoker, Action
from rl_games.envs.poker.deepcfr.obs_processor import ObsProcessor

import torch
import numpy as np
from gym import spaces
import os

class RandomPlayer:
    def __call__(self, _):
        return np.random.choice(
            [Action.FOLD, Action.CHECK_CALL, Action.RAISE, Action.ALL_IN]
        )


class RLGamesObsProcessor(ObsProcessor):
    def __call__(self, obs):
        board = self._process_board(obs["board"])
        player_hand = self._process_hand(obs["player_hand"])
        stage = self._process_stage(obs["stage"])
        first_to_act_next_stage = self._process_first_to_act_next_stage(
            obs["first_to_act_next_stage"]
        )
        bets_and_stacks = self._process_bets_and_stacks(obs)
        return np.array(
            player_hand + board + [stage, first_to_act_next_stage] + bets_and_stacks,
            dtype=np.float32,
        )


class PolicyPlayerWrapper:
    def __init__(self, policy):
        self.policy = policy

    def _batch_obses(self, obses):
        return {k: torch.tensor([obs[k] for obs in obses]) for k in obses[0].keys()}

    def __call__(self, obs):
        with torch.no_grad():
            obs_dict = {
                "board_and_hand": [int(x) for x in obs[:21]],
                "stage": int(obs[21]),
                "first_to_act_next_stage": int(obs[22]),
                "bets_and_stacks": list(obs[23:]),
            }

            obs = self._batch_obses([obs_dict])
            action_distribution = self.policy(obs)[0]
            action_distribution = torch.nn.functional.softmax(
                action_distribution, dim=-1
            )
            action = torch.multinomial(action_distribution, 1).item()
            return action



class RLGAgentWrapper:
    def __init__(self, agent, process_obs=True, is_deterministic=False):
        self.agent = agent
        self.is_deterministic = is_deterministic
        self.process_obs = process_obs
        if process_obs:
            self.obs_processor = RLGamesObsProcessor()
    def __call__(self, obs):
        if self.process_obs:
            obs = self.obs_processor(obs)
        obs = self.agent.obs_to_torch(obs)
        action = self.agent.get_action(obs, self.is_deterministic).item()
        return action

class HeadsUpPokerRLGames(HeadsUpPoker):
    def __init__(self):
        from deepcfr.model import BaseModel

        obs_processor = RLGamesObsProcessor()
        policy = BaseModel()
        policy.load_state_dict(
            torch.load(
                "deepcfr/policy.pth",
                weights_only=True,
                map_location="cpu",
            )
        )
        model = PolicyPlayerWrapper(policy)
        # model = RandomPlayer()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32
        )

        super(HeadsUpPokerRLGames, self).__init__(obs_processor, model)


class HeadsUpPokerRLGamesSelfplay(HeadsUpPoker):
    def __init__(self):
        
        obs_processor = RLGamesObsProcessor()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32
        )

        super(HeadsUpPokerRLGamesSelfplay, self).__init__(obs_processor, None)
        self.agent = self._create_agent()
        model = RLGAgentWrapper(self.agent, process_obs=False)
        self.env_player = model

    def update_weights(self, weigths):
        self.agent.set_weights(weigths)


    def _create_agent(self, config='rl_games/configs/ma/poker_sp_env.yaml'):
        import yaml
        from rl_games.torch_runner import Runner
        with open(config, 'r') as stream:
            config = yaml.safe_load(stream)
            runner = Runner()
            from rl_games.common.env_configurations import get_env_info
            config['params']['config']['env_info'] = get_env_info(self)
            runner.load(config)
        

        # 'RAYLIB has bug here, CUDA_VISIBLE_DEVICES become unset'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

        return runner.create_player()


if __name__ == "__main__":
    env = HeadsUpPokerRLGames()
    observation = env.reset()
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            observation = env.reset()
    env.close()
