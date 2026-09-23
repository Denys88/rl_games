import gym
import numpy as np
from gym import spaces
from treys import Card, Deck, Evaluator

from enums import Action, Stage
from obs_processor import ObsProcessor


def _convert_list_of_cards_to_str(cards):
    return [Card.int_to_str(card) for card in cards]


_evaluator = Evaluator()


class HeadsUpPoker(gym.Env):
    def __init__(self, obs_processor):
        super(HeadsUpPoker, self).__init__()

        # env player
        self.obs_processor = obs_processor

        # define action space
        self.action_space = spaces.Discrete(len(Action))

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
        self.raises_on_this_stage = 0

    def _initialize_stack_sizes(self):
        return [self.stack_size, self.stack_size]

    def _next_player(self, idx):
        idx = (idx + 1) % self.num_players
        while idx not in self.active_players:
            idx = (idx + 1) % self.num_players
        return idx

    def _stage_over(self):
        everyone_acted = all(
            player_idx in self.players_acted_this_stage
            for player_idx in self.active_players
        )
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

        self.board = []
        self.raises_on_this_stage = 0
        self.player_hand = [self.deck.draw(2), self.deck.draw(2)]
        self.dealer_idx = 0
        self.stage = Stage.PREFLOP
        self.active_players = [0, 1]
        self.players_acted_this_stage = []
        self.pot_size = 0
        self.bets = [0, 0]
        self.bets_this_stage = [0, 0]
        self.stack_sizes = self._initialize_stack_sizes()
        self.current_idx = self.dealer_idx
        self._apply_blinds()

        return self._get_obs()

    def _apply_blinds(self):
        self.bets = [self.small_blind, self.big_blind]
        self.stack_sizes[0] -= self.bets[0]
        self.stack_sizes[1] -= self.bets[1]
        self.pot_size += sum(self.bets)
        self.bets_this_stage = [self.small_blind, self.big_blind]

    def _game_over(self):
        assert len(self.active_players) > 0
        return len(self.active_players) == 1

    def _everyone_all_in(self):
        return len(self.active_players) == 2 and all(
            self.stack_sizes[player_idx] == 0 for player_idx in self.active_players
        )

    def _evaluate(self):
        # draw remaining cards
        if len(self.board) < 5:
            self.board += self.deck.draw(5 - len(self.board))

        player_0 = _evaluator.evaluate(self.board, self.player_hand[0])
        player_1 = _evaluator.evaluate(self.board, self.player_hand[1])
        if player_0 == player_1:
            return [0, 0]

        player_0_mult = 1 if player_0 < player_1 else -1
        player_1_mult = 1 if player_0 > player_1 else -1
        pot_value = min(self.bets[0], self.bets[1])
        return [player_0_mult * pot_value, player_1_mult * pot_value]

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

        self.players_acted_this_stage.append(self.current_idx)

    def step(self, action):
        action = Action(action)

        # if there are 3 raises in a row, the last raise is an all-in
        if action == Action.RAISE:
            self.raises_on_this_stage += 1
            if self.raises_on_this_stage == 3:
                action = Action.ALL_IN
        else:
            self.raises_on_this_stage = 0
        self._player_acts(action)

        # player folded
        if self._game_over():
            winner_idx = self.active_players[0]
            assert winner_idx != self.current_idx

            loser_idx = self.current_idx
            rewards = [0, 0]
            rewards[winner_idx] = self.pot_size - self.bets[winner_idx]
            rewards[loser_idx] = -self.bets[loser_idx]

            return None, rewards, True, {}

        self._move_to_next_player()
        # move to next stage if needed
        if self._stage_over():
            self._next_stage()

        # if evaluation phase
        if self.stage == Stage.END or self._everyone_all_in():
            return None, self._evaluate(), True, {}

        return self._get_obs(), None, False, {}

    def _get_obs(self):
        next_player = self._next_player(self.current_idx)
        return self.obs_processor(
            {
                "board": self.board,
                "player_idx": self.current_idx,
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
        print(f"board: {_convert_list_of_cards_to_str(self.board)}")
        print(
            f"player_hand: {_convert_list_of_cards_to_str(self.player_hand[self.current_idx])}"
        )
        print(f"stack_size: {self.stack_sizes[self.current_idx]}")
        print(f"pot_size: {self.pot_size}")
        print(f"player idx: {self.current_idx}")
        print(f"player_total_bet: {self.bets[self.current_idx]}")
        print(f"opponent_total_bet: {self.bets[self._next_player(self.current_idx)]}")
        print(f"player_this_stage_bet: {self.bets_this_stage[self.current_idx]}")
        print(
            f"opponent_this_stage_bet: {self.bets_this_stage[self._next_player(self.current_idx)]}"
        )
        print(f"first_to_act_next_stage: {self.current_idx != self.dealer_idx}")
        print(f"stage: {self.stage.name}")
        print("*" * 50)

    def _draw_cards(self):
        if self.stage == Stage.PREFLOP:
            return

        if self.stage == Stage.FLOP:
            self.board += self.deck.draw(3)
        elif self.stage == Stage.TURN:
            self.board += self.deck.draw(1)
        elif self.stage == Stage.RIVER:
            self.board += self.deck.draw(1)

    def _next_stage(self):
        self.players_acted_this_stage = []
        self.bets_this_stage = [0, 0]
        assert self.stage != Stage.END
        self.stage = Stage(self.stage.value + 1)
        self.current_idx = self.dealer_idx
        self._move_to_next_player()
        self._draw_cards()


def debug_env():
    MAX_ITER = 10000
    all_rewards = []
    obs_processor = ObsProcessor()
    env = HeadsUpPoker(obs_processor)
    observation = env.reset()

    class AlwaysCallPlayer:
        def __call__(self, _):
            return Action.CHECK_CALL

    players = [AlwaysCallPlayer(), AlwaysCallPlayer()]

    for _ in range(MAX_ITER):
        env.render()
        player_idx = observation["player_idx"]
        action = players[player_idx](observation)
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
    player_0_rewards = sum(reward[0] for reward in all_rewards) / len(all_rewards)
    player_1_rewards = sum(reward[1] for reward in all_rewards) / len(all_rewards)
    print("Average rewards:", player_0_rewards, player_1_rewards)


if __name__ == "__main__":
    debug_env()
