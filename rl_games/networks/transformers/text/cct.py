import torch.nn as nn
from ..utils.transformers import MaskedTransformerClassifier
from ..utils.tokenizer import TextTokenizer
from ..utils.embedder import Embedder

__all__ = [
    'text_cct_2',
    'text_cct_4',
    'text_cct_6'
]


class TextCCT(nn.Module):
    def __init__(self,
                 seq_len=64,
                 word_embedding_dim=300,
                 embedding_dim=256,
                 kernel_size=2,
                 stride=1,
                 padding=1,
                 pooling_kernel_size=2,
                 pooling_stride=2,
                 pooling_padding=1,
                 *args, **kwargs):
        super(TextCCT, self).__init__()

        self.embedder = Embedder(word_embedding_dim=word_embedding_dim,
                                 *args, **kwargs)

        self.tokenizer = TextTokenizer(n_input_channels=word_embedding_dim,
                                       n_output_channels=embedding_dim,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       pooling_kernel_size=pooling_kernel_size,
                                       pooling_stride=pooling_stride,
                                       pooling_padding=pooling_padding,
                                       max_pool=True,
                                       activation=nn.ReLU)

        self.classifier = MaskedTransformerClassifier(
            seq_len=self.tokenizer.seq_len(seq_len=seq_len, embed_dim=word_embedding_dim),
            embedding_dim=embedding_dim,
            seq_pool=True,
            dropout=0.,
            attention_dropout=0.1,
            stochastic_depth=0.1,
            *args, **kwargs)

    def forward(self, x, mask=None):
        x, mask = self.embedder(x, mask=mask)
        x, mask = self.tokenizer(x, mask=mask)
        out = self.classifier(x, mask=mask)
        return out


def _text_cct(num_layers, num_heads, mlp_ratio, embedding_dim,
              kernel_size=4, stride=None, padding=None,
              *args, **kwargs):
    stride = stride if stride is not None else max(1, (kernel_size // 2) - 1)
    padding = padding if padding is not None else max(1, (kernel_size // 2))

    return TextCCT(num_layers=num_layers,
                   num_heads=num_heads,
                   mlp_ratio=mlp_ratio,
                   embedding_dim=embedding_dim,
                   kernel_size=kernel_size,
                   stride=stride,
                   padding=padding,
                   *args, **kwargs)


def text_cct_2(*args, **kwargs):
    return _text_cct(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                     *args, **kwargs)


def text_cct_4(*args, **kwargs):
    return _text_cct(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                     *args, **kwargs)


def text_cct_6(*args, **kwargs):
    return _text_cct(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=256,
                     *args, **kwargs)
