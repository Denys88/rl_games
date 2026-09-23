import gym
import numpy as np
from gym import spaces
from treys import Card, Deck, Evaluator
from rl_games.envs.poker.deepcfr.enums import Action, Stage
from rl_games.envs.poker.deepcfr.obs_processor import ObsProcessor


class RandomPlayer:
    def __call__(self, obs):
        return np.random.choice(
            [Action.FOLD, Action.CHECK_CALL, Action.RAISE, Action.ALL_IN]
        )


class AlwaysCallPlayer:
    def __call__(self, obs):
        return Action.CHECK_CALL


def _convert_list_of_cards_to_str(cards):
    return [Card.int_to_str(card) for card in cards]


class HeadsUpPoker(gym.Env):
    def __init__(self, obs_processor, model):
        super(HeadsUpPoker, self).__init__()

        # env player
        self.env_player = model
        self.obs_processor = obs_processor

        # define action space
        self.action_space = spaces.Discrete(len(Action))

        # poker hand evaluator
        self.evaluator = Evaluator()

        # config
        self.big_blind = 2
        self.small_blind = 1
        self.num_players = 2
        self.stack_size = 100

        assert self.big_blind < self.stack_size
        assert self.small_blind < self.big_blind

        # env variables
        self.deck = None
        self.board = None
        self.player_hand = None
        self.stack_sizes = None
        self.dealer_idx = None
        self.active_players = None
        self.players_acted_this_stage = None
        self.bets = None
        self.pot_size = None
        self.bets_this_stage = None
        self.current_idx = None
        self.stage = None
        self.game_counter = 0

    def _initialize_stack_sizes(self):
        return [self.stack_size, self.stack_size]

    def _next_player(self, idx):
        idx = (idx + 1) % self.num_players
        while idx not in self.active_players:
            idx = (idx + 1) % self.num_players
        return idx

    def _stage_over(self):
        everyone_acted = set(self.active_players) == set(self.players_acted_this_stage)
        if not everyone_acted:
            return False
        max_bet_this_stage = max(self.bets_this_stage)
        for player_idx in self.active_players:
            if (
                self.bets_this_stage[player_idx] < max_bet_this_stage
                and self.stack_sizes[player_idx] != 0
            ):
                return False
        return True

    def _move_to_next_player(self):
        self.current_idx = self._next_player(self.current_idx)
        return self.current_idx

    def reset(self):
        self.game_counter += 1

        self.deck = Deck()
        self.deck.shuffle()

        self.board = self.deck.draw(5)
        self.player_hand = [self.deck.draw(2), self.deck.draw(2)]
        self.dealer_idx = 0
        self.stage = Stage.PREFLOP
        self.active_players = [0, 1]
        self.players_acted_this_stage = set()
        self.pot_size = 0
        self.bets = [0, 0]
        self.bets_this_stage = [0, 0]
        self.stack_sizes = self._initialize_stack_sizes()
        self.current_idx = self.dealer_idx
        self.is_player_dealer = np.random.uniform() < 0.5
        self._apply_blinds()

        if not self.is_player_dealer:
            self._env_player_acts()

        return self._get_obs()

    def _apply_blinds(self):
        self.bets = [self.small_blind, self.big_blind]
        self.stack_sizes[0] -= self.bets[0]
        self.stack_sizes[1] -= self.bets[1]
        self.pot_size += sum(self.bets)
        self.bets_this_stage = [self.small_blind, self.big_blind]

    def _env_player_acts(self):
        action = self.env_player(self._get_obs())
        self._player_acts(action)

    def _game_over(self):
        assert len(self.active_players) > 0
        return len(self.active_players) == 1

    def _everyone_all_in(self):
        return len(self.active_players) == 2 and all(
            self.stack_sizes[player_idx] == 0 for player_idx in self.active_players
        )

    def _evaluate(self):
        player_0 = self.evaluator.evaluate(self.board, self.player_hand[0])
        player_1 = self.evaluator.evaluate(self.board, self.player_hand[1])
        if player_0 == player_1:
            return 0

        if self.is_player_dealer:
            mult = 1 if player_0 < player_1 else -1
        else:
            mult = 1 if player_0 > player_1 else -1
        return mult * min(self.bets[0], self.bets[1])

    def _player_acts(self, action):
        if type(action) in [np.int64, int]:
            action = Action(action)

        if action == Action.FOLD:
            self.active_players.remove(self.current_idx)
        elif action == Action.CHECK_CALL:
            max_bet_this_stage = max(self.bets_this_stage)
            bet_update = max_bet_this_stage - self.bets_this_stage[self.current_idx]
            self.bets[self.current_idx] += bet_update
            self.bets_this_stage[self.current_idx] += bet_update
            self.stack_sizes[self.current_idx] -= bet_update
            self.pot_size += bet_update
        elif action == Action.RAISE:
            max_bet_this_stage = max(self.bets_this_stage)
            bet_update = (
                max_bet_this_stage
                - self.bets_this_stage[self.current_idx]
                + self.big_blind
            )
            if self.stack_sizes[self.current_idx] < bet_update:
                bet_update = self.stack_sizes[self.current_idx]
            self.bets[self.current_idx] += bet_update
            self.bets_this_stage[self.current_idx] += bet_update
            self.stack_sizes[self.current_idx] -= bet_update
            self.pot_size += bet_update
        elif action == Action.ALL_IN:
            bet_update = self.stack_sizes[self.current_idx]
            self.bets[self.current_idx] += bet_update
            self.bets_this_stage[self.current_idx] += bet_update
            self.stack_sizes[self.current_idx] = 0
            self.pot_size += bet_update
        else:
            raise ValueError("Invalid action")

        self.players_acted_this_stage.add(self.current_idx)

        self._move_to_next_player()

    def _env_player_move(self):
        if self.is_player_dealer:
            return self.current_idx == 1
        return self.current_idx == 0

    def step(self, action):
        action = Action(action)

        # env player folded
        if self._game_over():
            return None, self.pot_size - self.bets[self.active_players[0]], True, {}

        self._player_acts(action)

        # player folded
        if self._game_over():
            return None, -(self.pot_size - self.bets[self.active_players[0]]), True, {}

        # move to next stage if needed
        if self._stage_over():
            self._next_stage()

        # if evaluation phase
        if self.stage == Stage.END or self._everyone_all_in():
            return None, self._evaluate(), True, {}

        while self._env_player_move():
            self._env_player_acts()
            # env player folded
            if self._game_over():
                return None, self.pot_size - self.bets[self.active_players[0]], True, {}

            # move to next stage if needed
            if self._stage_over():
                self._next_stage()

            # if evaluation phase
            if self.stage == Stage.END or self._everyone_all_in():
                return None, self._evaluate(), True, {}

        # if evaluation phase
        if self.stage == Stage.END or self._everyone_all_in():
            return None, self._evaluate(), True, {}

        return self._get_obs(), 0, False, {}

    def _board(self):
        if self.stage == Stage.PREFLOP:
            return []
        if self.stage == Stage.FLOP:
            return self.board[:3]
        if self.stage == Stage.TURN:
            return self.board[:4]
        return self.board

    def _get_obs(self):
        next_player = self._next_player(self.current_idx)
        return self.obs_processor(
            {
                "board": self._board(),
                "player_hand": self.player_hand[self.current_idx],
                "stack_size": self.stack_sizes[self.current_idx],
                "pot_size": self.pot_size,
                "stage": self.stage,
                "player_total_bet": self.bets[self.current_idx],
                "opponent_total_bet": self.bets[next_player],
                "player_this_stage_bet": self.bets_this_stage[self.current_idx],
                "opponent_this_stage_bet": self.bets_this_stage[next_player],
                "first_to_act_next_stage": self.current_idx != self.dealer_idx,
            }
        )

    def render(self):
        print("*" * 50)
        print(f"Game id: {self.game_counter}")
        print(f"board: {_convert_list_of_cards_to_str(self._board())}")
        print(
            f"player_hand: {_convert_list_of_cards_to_str(self.player_hand[self.current_idx])}"
        )
        print(f"stack_size: {self.stack_sizes[self.current_idx]}")
        print(f"pot_size: {self.pot_size}")
        print(f"player_total_bet: {self.bets[self.current_idx]}")
        print(f"opponent_total_bet: {self.bets[self._next_player(self.current_idx)]}")
        print(f"player_this_stage_bet: {self.bets_this_stage[self.current_idx]}")
        print(
            f"opponent_this_stage_bet: {self.bets_this_stage[self._next_player(self.current_idx)]}"
        )
        print(f"first_to_act_next_stage: {self.current_idx != self.dealer_idx}")
        print(f"stage: {self.stage.name}")
        print("*" * 50)

    def _next_stage(self):
        self.players_acted_this_stage = set()
        self.bets_this_stage = [0, 0]
        assert self.stage != Stage.END
        self.stage = Stage(self.stage.value + 1)
        self.current_idx = self.dealer_idx
        self._move_to_next_player()


def debug_env():
    MAX_ITER = 100
    all_rewards = []
    obs_processor = ObsProcessor()
    env = HeadsUpPoker(obs_processor, AlwaysCallPlayer())
    observation = env.reset()
    for _ in range(MAX_ITER):
        env.render()
        action = int(input("Enter action: "))
        observation, reward, done, info = env.step(action)
        if done:
            board = _convert_list_of_cards_to_str(env.board)
            player_0 = _convert_list_of_cards_to_str(env.player_hand[0])
            player_1 = _convert_list_of_cards_to_str(env.player_hand[1])
            print("reward: ", reward)
            print("board:", board)
            print("player_0:", player_0)
            print("player_1:", player_1)
            all_rewards.append(reward)
            observation = env.reset()
    env.close()

    print("Number of hands played:", len(all_rewards))
    print("Average rewards:", sum(all_rewards) / len(all_rewards))


if __name__ == "__main__":
    debug_env()
