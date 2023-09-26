import torch.nn as nn
from ..utils.transformers import MaskedTransformerClassifier
from ..utils.tokenizer import TextTokenizer
from ..utils.embedder import Embedder

__all__ = [
    'text_cvt_2',
    'text_cvt_4',
    'text_cvt_6',
]


class TextCVT(nn.Module):
    def __init__(self,
                 seq_len=64,
                 word_embedding_dim=300,
                 embedding_dim=768,
                 patch_size=2,
                 *args, **kwargs):
        super(TextCVT, self).__init__()
        assert seq_len % patch_size == 0, f"sequence length ({seq_len}) has to be" \
                                          f"divisible by patch size ({patch_size})"
        self.embedder = Embedder(word_embedding_dim=word_embedding_dim,
                                 *args, **kwargs)

        self.tokenizer = TextTokenizer(n_input_channels=word_embedding_dim,
                                       n_output_channels=embedding_dim,
                                       kernel_size=patch_size,
                                       stride=patch_size,
                                       padding=0,
                                       max_pool=False,
                                       activation=None)

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


def _text_cvt(num_layers, num_heads, mlp_ratio, embedding_dim,
              patch_size=4, *args, **kwargs):
    return TextCVT(num_layers=num_layers,
                   num_heads=num_heads,
                   mlp_ratio=mlp_ratio,
                   embedding_dim=embedding_dim,
                   patch_size=patch_size,
                   *args, **kwargs)


def text_cvt_2(*args, **kwargs):
    return _text_cvt(num_layers=2, num_heads=2, mlp_ratio=1, embedding_dim=128,
                     *args, **kwargs)


def text_cvt_4(*args, **kwargs):
    return _text_cvt(num_layers=4, num_heads=2, mlp_ratio=1, embedding_dim=128,
                     *args, **kwargs)


def text_cvt_6(*args, **kwargs):
    return _text_cvt(num_layers=6, num_heads=4, mlp_ratio=2, embedding_dim=128,
                     *args, **kwargs)
