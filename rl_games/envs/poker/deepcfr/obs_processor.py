from treys import Card


class ObsProcessor:
    def _get_suit_int(self, card):
        suit_int = Card.get_suit_int(card)
        if suit_int == 1:
            return 0
        elif suit_int == 2:
            return 1
        elif suit_int == 4:
            return 2
        elif suit_int == 8:
            return 3
        raise ValueError("Invalid suit")

    def _process_card(self, card):
        card_rank = Card.get_rank_int(card)
        card_suit = self._get_suit_int(card)
        card_index = card_rank + card_suit * 13
        return [card_rank + 1, card_suit + 1, card_index + 1]

    def _process_board(self, board):
        result = []
        for i in range(5):
            if i >= len(board):
                result += [0, 0, 0]
            else:
                result += self._process_card(board[i])
        return result

    def _process_hand(self, hand):
        result = []
        for card in hand:
            result += self._process_card(card)
        return result

    def _process_stage(self, stage):
        return stage.value

    def _process_first_to_act_next_stage(self, first_to_act_next_stage):
        return int(first_to_act_next_stage)

    def _process_bets_and_stacks(self, obs):
        stack_size = obs["stack_size"]
        pot_size = obs["pot_size"]
        player_total_bet = obs["player_total_bet"]
        opponent_total_bet = obs["opponent_total_bet"]
        player_this_stage_bet = obs["player_this_stage_bet"]
        opponent_this_stage_bet = obs["opponent_this_stage_bet"]

        # return normalized values
        return [
            (opponent_this_stage_bet - player_this_stage_bet) / pot_size,
            player_total_bet / pot_size,
            opponent_total_bet / pot_size,
            player_this_stage_bet / pot_size,
            opponent_this_stage_bet / pot_size,
            stack_size / pot_size,
            pot_size / 1000,
            (
                (opponent_this_stage_bet - player_this_stage_bet) / stack_size
                if stack_size > 0
                else 0
            ),
        ]

    def __call__(self, obs):
        board = self._process_board(obs["board"])
        player_hand = self._process_hand(obs["player_hand"])
        stage = self._process_stage(obs["stage"])
        first_to_act_next_stage = self._process_first_to_act_next_stage(
            obs["first_to_act_next_stage"]
        )
        bets_and_stacks = self._process_bets_and_stacks(obs)
        processed_obs = {
            "board_and_hand": player_hand + board,  # 6 + 15
            "stage": stage,  # 1
            "first_to_act_next_stage": first_to_act_next_stage,  # 1
            "bets_and_stacks": bets_and_stacks,  # 8
        }

        if "player_idx" in obs:
            processed_obs["player_idx"] = obs["player_idx"]

        return processed_obs
