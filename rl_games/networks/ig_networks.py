from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import math
from rl_games.networks.transformers.utils.transformers import TransformerClassifier
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from rl_games.algos_torch.network_builder import NetworkBuilder


class EncoderMLPBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = EncoderMLPBuilder.Network(self.params, **kwargs)
        return net

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            if self.embedding_reduction == 'sum' :
                self.red_op = torch.sum
            if self.embedding_reduction == 'mean' :
                self.red_op = torch.mean
            self.mlp = nn.Sequential()
            self.encoders = torch.nn.ModuleList([torch.nn.Linear(num, self.embedding_size,bias=False) for num in self.input_split])
            mlp_input_shape = self.embedding_size

            out_size = self.units[-1]

            self.encoder_act = self.activations_factory.create(self.activation) 
            mlp_args = {
                'input_size' : mlp_input_shape, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            self.mlp = self._build_mlp(**mlp_args)

            self.value = torch.nn.Linear(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            mlp_init = self.init_factory.create(**self.initializer)

            for m in self.modules():         
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)  

        def load(self, params):
            super().load(params)

            return

        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def load(self, params):
            self.embedding_size = params['mlp']['embedding_size']
            self.embedding_reduction = params['mlp']['embedding_reduction']
            self.input_split = list(params['mlp']['input_split'])

            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_space = 'space' in params
            self.central_value = params.get('central_value', False)
            self.is_continuous = True
            self.space_config = params['space']['continuous']
            self.fixed_sigma = self.space_config['fixed_sigma']            

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            outs = torch.split(obs, self.input_split, dim=1)
            encoded_features = [e(o) for e, o in zip(self.encoders, outs)]

            stacked_features = torch.stack(encoded_features)
            out = self.red_op(stacked_features, dim=0)
            out = self.encoder_act(out)
            out = self.mlp(out)
            value = self.value_act(self.value(out))

            if self.central_value:
                return value, None

            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return mu, mu*0 + sigma, value, None


class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self,
    actions_num = 1,
    input_shape = 128,
    input_split = [], 
    seq_pool=True,
    embedding_dim=768,
    num_layers=12,
    num_heads=12,
    mlp_ratio=4.0,
    dropout=0.1,
    attention_dropout=0.1,
    stochastic_depth=0.1,
    split_features=False,
    positional_embedding='none',
    ):
        super(TransformerModel, self).__init__()
        self.input_shape = input_shape
        self.input_split = list(input_split)
        self.split_features = split_features
        if split_features:
            self.encoders = torch.nn.ModuleList([torch.nn.Linear(num, embedding_dim,bias=False) for num in self.input_split])
        else:
            self.projector = torch.nn.Linear(input_shape[1], embedding_dim, bias=False)
        self.actions_num = actions_num
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
        self.transformer_encoder = TransformerClassifier(
            seq_pool=seq_pool,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_classes=actions_num+1,
            dropout=dropout,
            attention_dropout=attention_dropout,
            stochastic_depth=stochastic_depth,
            positional_embedding=positional_embedding,
            sequence_length = len(self.input_split) if split_features else input_shape[0]
        )
        #for m in self.encoders:
        #    nn.init.trunc_normal_(m.weight, std=.02)

    def forward(self, src):
        if self.split_features:
            src = torch.split(src, self.input_split, dim=1)
            src = [e(o) for e, o in zip(self.encoders, src)]
            src = [e.unsqueeze(1) for e in src]
            src = torch.cat(src, dim=1)
        else:
            src = self.projector(src)

        output = self.transformer_encoder(src)
        mu, value = torch.split(output, [self.actions_num,1], dim=1)
        return mu, mu*0 + self.sigma, value, None


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TorchTransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, 
        actions_num = 1,
        input_shape = [4,42],
        input_split = [], 
        seq_pool=True, #unused
        embedding_dim=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        attention_dropout=0.1, #unused
        stochastic_depth=0.1, #unused
        split_features=False, #unused
        positional_embedding='none',):
        '''
        half of params are unused now
        '''
        super(TorchTransformerModel, self).__init__()

        self.actions_num = actions_num
        self.src_mask = None
        self.positional_embedding = positional_embedding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        dim_feedforward = int(mlp_ratio * embedding_dim)
        encoder_layers = TransformerEncoderLayer(d_model = embedding_dim, nhead = num_heads, dim_feedforward = dim_feedforward, 
                                                 dropout = dropout, activation='gelu', batch_first = False)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers, nn.LayerNorm(embedding_dim))
        #self.transformer_encoder = torch.compile(self.transformer_encoder)
        self.decoder = nn.Linear(embedding_dim, actions_num + 1)
        self.last_ln = nn.LayerNorm(embedding_dim)
        self.proj_layer = nn.Linear(input_shape[1], embedding_dim)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

    def forward(self, src):
        src = src.permute(1,0,2)
        src = self.proj_layer(src)
        if not self.positional_embedding == None:
            src = self.pos_encoder(src)

        output = self.transformer_encoder(src)
        output = torch.mean(output, dim=0)
        #output = self.last_ln(output)
        output = self.decoder(output)
        mu, value = torch.split(output, [self.actions_num,1], dim=1)
        return mu, mu*0 + self.sigma, value, None


class TransformerBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = TransformerBuilder.Network(self.params, **kwargs)
        return net

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            self.transformer = TransformerModel(actions_num, input_shape, **self.tranformer_params)

        def load(self, params):
            super().load(params)

            return

        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None
             
        def load(self, params):
            self.tranformer_params = params['transformer']        

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            return self.transformer(obs)


class TorchTransformerBuilder(network_builder.NetworkBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def load(self, params):
        self.params = params

    def build(self, name, **kwargs):
        net = TorchTransformerBuilder.Network(self.params, **kwargs)
        return net

    class Network(network_builder.NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)

            self.transformer = TorchTransformerModel(actions_num, input_shape, **self.tranformer_params)

        def load(self, params):
            super().load(params)

            return

        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return False

        def get_default_rnn_state(self):
            return None

        def load(self, params):
            self.tranformer_params = params['transformer']        

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            return self.transformer(obs)