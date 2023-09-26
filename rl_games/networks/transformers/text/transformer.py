import torch.nn as nn
from ..utils.transformers import MaskedTransformerClassifier
from ..utils.tokenizer import TextTokenizer
from ..utils.embedder import Embedder

__all__ = [
    'text_transformer_2',
    'text_transformer_4',
    'text_transformer_6',
]


class TextTransformerLite(nn.Module):
    def __init__(self,
                 seq_len=64,
                 word_embedding_dim=300,
                 *args, **kwargs):
        super(TextTransformerLite, self).__init__()
        self.embedder = Embedder(word_embedding_dim=word_embedding_dim,
                                 *args, **kwargs)

        self.classifier = MaskedTransformerClassifier(
            seq_len=seq_len,
            embedding_dim=word_embedding_dim,
            seq_pool=False,
            dropout=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)

    def forward(self, x, mask=None):
        x, mask = self.embedder(x, mask=mask)
        out = self.classifier(x, mask=mask)
        return out


def _text_transformer(num_layers, num_heads, mlp_ratio,
                      *args, **kwargs):
    return TextTransformerLite(num_layers=num_layers,
                               num_heads=num_heads,
                               mlp_ratio=mlp_ratio,
                               *args, **kwargs)


def text_transformer_2(*args, **kwargs):
    return _text_transformer(num_layers=2, num_heads=2, mlp_ratio=1,
                             *args, **kwargs)


def text_transformer_4(*args, **kwargs):
    return _text_transformer(num_layers=4, num_heads=2, mlp_ratio=1,
                             *args, **kwargs)


def text_transformer_6(*args, **kwargs):
    return _text_transformer(num_layers=6, num_heads=4, mlp_ratio=2,
                             *args, **kwargs)
