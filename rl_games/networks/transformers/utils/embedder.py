import torch.nn as nn


class Embedder(nn.Module):
    def __init__(self,
                 word_embedding_dim=300,
                 vocab_size=100000,
                 padding_idx=1,
                 pretrained_weight=None,
                 embed_freeze=False,
                 *args, **kwargs):
        super(Embedder, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(pretrained_weight, freeze=embed_freeze) \
            if pretrained_weight is not None else \
            nn.Embedding(vocab_size, word_embedding_dim, padding_idx=padding_idx)
        self.embeddings.weight.requires_grad = not embed_freeze

    def forward_mask(self, mask):
        bsz, seq_len = mask.shape
        new_mask = mask.view(bsz, seq_len, 1)
        new_mask = new_mask.sum(-1)
        new_mask = (new_mask > 0)
        return new_mask

    def forward(self, x, mask=None):
        embed = self.embeddings(x)
        embed = embed if mask is None else embed * self.forward_mask(mask).unsqueeze(-1).float()
        return embed, mask

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        else:
            nn.init.normal_(m.weight)
