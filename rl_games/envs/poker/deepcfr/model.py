import torch
import torch.nn as nn

SUITS = 4
RANKS = 13
EMBEDDING_DIM = 64

# Number of actions: fold, check/call, raise, all-in
NUM_ACTIONS = 4


class CardEmbedding(nn.Module):
    def __init__(self, dim):
        super(CardEmbedding, self).__init__()
        self.rank_embedding = nn.Embedding(RANKS + 1, dim)
        self.suit_embedding = nn.Embedding(SUITS + 1, dim)
        self.card_embedding = nn.Embedding(RANKS * SUITS + 1, dim)

    def forward(self, x):
        ranks = x[:, :, 0].long()
        suits = x[:, :, 1].long()
        card_indices = x[:, :, 2].long()

        ranks_emb = self.rank_embedding(ranks)
        suits_emb = self.suit_embedding(suits)
        card_indices_emb = self.card_embedding(card_indices)

        embeddings = ranks_emb + suits_emb + card_indices_emb
        hand_embedding = embeddings[:, :2, :].sum(dim=1)
        flop = embeddings[:, 2:5, :].sum(dim=1)
        turn = embeddings[:, 5:6, :].sum(dim=1)
        river = embeddings[:, 6:7, :].sum(dim=1)

        return torch.cat([hand_embedding, flop, turn, river], dim=1)


class CardModel(nn.Module):
    def __init__(self):
        super(CardModel, self).__init__()
        self.cards_embeddings = CardEmbedding(EMBEDDING_DIM)
        self.fc1 = nn.Linear(EMBEDDING_DIM * 4, EMBEDDING_DIM)
        self.fc2 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.fc3 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.cards_embeddings(x)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return x


class StageAndOrderModel(nn.Module):
    def __init__(self):
        super(StageAndOrderModel, self).__init__()
        self.stage_embedding = nn.Embedding(4, EMBEDDING_DIM)
        self.first_to_act_embedding = nn.Embedding(2, EMBEDDING_DIM)
        self.fc1 = nn.Linear(2 * EMBEDDING_DIM, EMBEDDING_DIM)
        self.fc2 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.act = nn.ReLU()

    def forward(self, stage, first_to_act_next_stage):
        stage_emb = self.stage_embedding(stage)
        first_to_act_next_stage_emb = self.first_to_act_embedding(
            first_to_act_next_stage
        )
        x = torch.cat([stage_emb, first_to_act_next_stage_emb], dim=1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x


class BetsModel(nn.Module):
    def __init__(self):
        super(BetsModel, self).__init__()
        self.fc1 = nn.Linear(8, EMBEDDING_DIM)
        self.fc2 = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x) + x)
        return x


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.num_actions = NUM_ACTIONS
        self.card_model = CardModel()
        self.stage_and_order_model = StageAndOrderModel()
        self.bets_model = BetsModel()

        self.act = torch.nn.ReLU()
        self.comb1 = torch.nn.Linear(3 * EMBEDDING_DIM, EMBEDDING_DIM)
        self.comb2 = torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.comb3 = torch.nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)

        self.action_head = torch.nn.Linear(EMBEDDING_DIM, NUM_ACTIONS)

        self.action_head.weight.data.fill_(0)
        self.action_head.bias.data.fill_(0)

    def normalize(self, z):
        return (z - z.mean(dim=1, keepdim=True)) / (z.std(dim=1, keepdim=True) + 1e-6)

    def forward(self, x):
        stage = x["stage"].long()
        board_and_hand = x["board_and_hand"]
        first_to_act_next_stage = x["first_to_act_next_stage"].long()

        board_and_hand = board_and_hand.view(-1, 7, 3)
        board_and_hand_emb = self.card_model(board_and_hand)

        stage_and_order_emb = self.stage_and_order_model(stage, first_to_act_next_stage)
        bets_and_stacks = self.bets_model(x["bets_and_stacks"])

        z = torch.cat([board_and_hand_emb, stage_and_order_emb, bets_and_stacks], dim=1)

        z = self.act(self.comb1(z))
        z = self.act(self.comb2(z) + z)
        z = self.act(self.comb3(z) + z)

        z = self.normalize(z)
        return self.action_head(z)
