import torch
from torch import nn
from torchvision import models
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs
import torch.nn.functional as F
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.network_builder import NetworkBuilder, ImpalaSequential


class VisionImpalaBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            self.actions_num = actions_num = kwargs.pop('actions_num')
            full_input_shape = kwargs.pop('input_shape')
            proprio_size = 0 # Number of proprioceptive features
            if type(full_input_shape) is dict:
                input_shape = full_input_shape['camera']
                proprio_shape = full_input_shape['proprio']

                proprio_size = proprio_shape[0]
            else:
                input_shape = full_input_shape

            self.normalize_emb = kwargs.pop('normalize_emb', False)

            self.num_seqs = kwargs.pop('num_seqs', 1)
            self.value_size = kwargs.pop('value_size', 1)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            if self.permute_input:
                input_shape = torch_ext.shape_whc_to_cwh(input_shape)

            self.cnn = self._build_impala(input_shape, self.conv_depths)
            cnn_output_size = self._calc_input_size(input_shape, self.cnn)

            mlp_input_size = cnn_output_size + proprio_size
            if len(self.units) == 0:
                out_size = cnn_output_size
            else:
                out_size = self.units[-1]

            self.running_mean_std = torch.jit.script(RunningMeanStd((mlp_input_size,)))
            self.layer_norm_emb = torch.nn.LayerNorm(mlp_input_size)
            #self.layer_norm_emb = torch.nn.RMSNorm(mlp_input_size)

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    out_size = self.rnn_units
                else:
                    rnn_in_size = mlp_input_size
                    mlp_input_size = self.rnn_units

                self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            mlp_args = {
                'input_size' : mlp_input_size,
                'units' : self.units,
                'activation' : self.activation,
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear
            }

            self.mlp = self._build_mlp(**mlp_args)

            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)
            self.flatten_act = self.activations_factory.create(self.activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
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
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)

            if self.is_discrete:
                mlp_init(self.logits.weight)
            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)

            mlp_init(self.value.weight)

        def norm_emb(self, embedding):
            #with torch.no_grad():
            return self.running_mean_std(embedding) if self.normalize_emb else embedding
                # if len(self.units) == 0:
                #     out_size = cnn_output_size
                # else:
                #     out_size = self.units[-1]

        def forward(self, obs_dict):
            obs = obs_dict['obs']['camera']
            proprio = obs_dict['obs']['proprio']
            if self.permute_input:
                obs = obs.permute((0, 3, 1, 2))

            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)
            states = obs_dict.get('rnn_states', None)

            out = obs
            out = self.cnn(out)
            out = out.flatten(1)
            out = self.flatten_act(out)

            out = torch.cat([out, proprio], dim=1)
            out = self.layer_norm_emb(out)

            if self.has_rnn:
                seq_length = obs_dict.get('seq_length', 1)

                out_in = out
                if not self.is_rnn_before_mlp:
                    out_in = out
                    out = self.mlp(out)

                batch_size = out.size()[0]
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1)

                if len(states) == 1:
                    states = states[0]

                out = out.transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1)
                    dones = dones.transpose(0, 1)
                out, states = self.rnn(out, states, dones, bptt_len)
                out = out.transpose(0, 1)
                out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                if self.rnn_ln:
                    out = self.layer_norm(out)
                if self.is_rnn_before_mlp:
                    out = self.mlp(out)
                if type(states) is not tuple:
                    states = (states,)
            else:
                out = self.mlp(out)

            value = self.value_act(self.value(out))

            if self.is_discrete:
                logits = self.logits(out)
                return logits, value, states

            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return mu, mu*0 + sigma, value, states

        def load(self, params):
            self.separate = False
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_discrete = 'discrete' in params['space']
            self.is_continuous = 'continuous' in params['space']
            self.is_multi_discrete = 'multi_discrete'in params['space']
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)

            if self.is_continuous:
                self.space_config = params['space']['continuous']
                self.fixed_sigma = self.space_config['fixed_sigma']
            elif self.is_discrete:
                self.space_config = params['space']['discrete']
            elif self.is_multi_discrete:
                self.space_config = params['space']['multi_discrete']

            self.has_rnn = 'rnn' in params
            if self.has_rnn:
                self.rnn_units = params['rnn']['units']
                self.rnn_layers = params['rnn']['layers']
                self.rnn_name = params['rnn']['name']
                self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
                self.rnn_ln = params['rnn'].get('layer_norm', False)

            self.has_cnn = True
            self.permute_input = params['cnn'].get('permute_input', True)
            self.conv_depths = params['cnn']['conv_depths']
            self.require_rewards = params.get('require_rewards')
            self.require_last_actions = params.get('require_last_actions')

        def _build_impala(self, input_shape, depths):
            in_channels = input_shape[0]
            layers = nn.ModuleList()
            for d in depths:
                layers.append(ImpalaSequential(in_channels, d))
                in_channels = d
            return nn.Sequential(*layers)

        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            num_layers = self.rnn_layers
            if self.rnn_name == 'lstm':
                return (torch.zeros((num_layers, self.num_seqs, self.rnn_units)),
                            torch.zeros((num_layers, self.num_seqs, self.rnn_units)))
            else:
                return (torch.zeros((num_layers, self.num_seqs, self.rnn_units)))

    def build(self, name, **kwargs):
        net = VisionImpalaBuilder.Network(self.params, **kwargs)
        return net


from torchvision import models, transforms

def preprocess_image(image):
    # Normalize the image using ImageNet's mean and standard deviation
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Mean of ImageNet dataset
        std=[0.229, 0.224, 0.225]   # Std of ImageNet dataset
    )

    # Apply the normalization
    image = normalize(image)

    return image


class VisionBackboneBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params

    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            self.actions_num = kwargs.pop('actions_num')
            full_input_shape = kwargs.pop('input_shape')

            print("Observations shape: ", full_input_shape)

            self.proprio_size = 0 # Number of proprioceptive features
            if isinstance(full_input_shape, dict):
                input_shape = full_input_shape['camera']
                proprio_shape = full_input_shape['proprio']
                self.proprio_size = proprio_shape[0]
            else:
                input_shape = full_input_shape

            self.num_seqs = kwargs.pop('num_seqs', 1)
            self.value_size = kwargs.pop('value_size', 1)

            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            if self.permute_input:
                input_shape = torch_ext.shape_whc_to_cwh(input_shape)

            self.cnn, self.cnn_output_size = self._build_backbone(input_shape, params['backbone'])

            mlp_input_size = self.cnn_output_size + self.proprio_size
            if len(self.units) == 0:
                out_size = self.cnn_output_size
            else:
                out_size = self.units[-1]

            self.layer_norm_emb = torch.nn.LayerNorm((mlp_input_size,))

            if self.has_rnn:
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    out_size = self.rnn_units
                else:
                    rnn_in_size = mlp_input_size
                    mlp_input_size = self.rnn_units

                self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            mlp_args = {
                'input_size': mlp_input_size,
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear
            }

            self.mlp = self._build_mlp(**mlp_args)

            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)
            self.flatten_act = self.activations_factory.create(self.activation)

            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, self.actions_num)
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, self.actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation'])
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation'])
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(torch.zeros(self.actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, self.actions_num)

            mlp_init = self.init_factory.create(**self.initializer)

            for m in self.mlp:
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)

            if self.is_discrete:
                mlp_init(self.logits.weight)
            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)

            mlp_init(self.value.weight)

        def forward(self, obs_dict):
            if self.proprio_size > 0:
                obs = obs_dict['obs']['camera']
                proprio = obs_dict['obs']['proprio']
            else:
                obs = obs_dict['obs']

            if self.permute_input:
                obs = obs.permute((0, 3, 1, 2))

            if self.preprocess_image:
                obs = preprocess_image(obs)

            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)
            states = obs_dict.get('rnn_states', None)

            out = obs
            out = self.cnn(out)
            out = out.flatten(1)
            out = self.flatten_act(out)

            if self.proprio_size > 0:
                out = torch.cat([out, proprio], dim=1)

            out = self.layer_norm_emb(out)

            if self.has_rnn:
                seq_length = obs_dict.get('seq_length', 1)

                out_in = out
                if not self.is_rnn_before_mlp:
                    out_in = out
                    out = self.mlp(out)

                batch_size = out.size()[0]
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1)

                if len(states) == 1:
                    states = states[0]

                out = out.transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1)
                    dones = dones.transpose(0, 1)
                out, states = self.rnn(out, states, dones, bptt_len)
                out = out.transpose(0, 1)
                out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                if self.rnn_ln:
                    out = self.layer_norm(out)
                if self.is_rnn_before_mlp:
                    out = self.mlp(out)
                if not isinstance(states, tuple):
                    states = (states,)
            else:
                out = self.mlp(out)

            value = self.value_act(self.value(out))

            if self.is_discrete:
                logits = self.logits(out)
                return logits, value, states

            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    sigma = self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                return mu, mu * 0 + sigma, value, states

        def load(self, params):
            self.separate = False
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_discrete = 'discrete' in params['space']
            self.is_continuous = 'continuous' in params['space']
            self.is_multi_discrete = 'multi_discrete' in params['space']
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)

            if self.is_continuous:
                self.space_config = params['space']['continuous']
                self.fixed_sigma = self.space_config['fixed_sigma']
            elif self.is_discrete:
                self.space_config = params['space']['discrete']
            elif self.is_multi_discrete:
                self.space_config = params['space']['multi_discrete']

            self.has_rnn = 'rnn' in params
            if self.has_rnn:
                self.rnn_units = params['rnn']['units']
                self.rnn_layers = params['rnn']['layers']
                self.rnn_name = params['rnn']['name']
                self.is_rnn_before_mlp = params['rnn'].get('before_mlp', False)
                self.rnn_ln = params['rnn'].get('layer_norm', False)

            self.has_cnn = True
            self.permute_input = params['backbone'].get('permute_input', True)
            self.require_rewards = params.get('require_rewards')
            self.require_last_actions = params.get('require_last_actions')

        def _build_backbone(self, input_shape, backbone_params):
            backbone_type = backbone_params['type']
            pretrained = backbone_params.get('pretrained', False)
            self.preprocess_image = backbone_params.get('preprocess_image', False)

            if backbone_type == 'resnet18' or backbone_type == 'resnet34':
                if backbone_type == 'resnet18':
                    backbone = models.resnet18(pretrained=pretrained, zero_init_residual=True)
                else:
                    backbone = models.resnet34(pretrained=pretrained, zero_init_residual=True)

                # Modify the first convolution layer to match input shape if needed
                # TODO: add low-res parameter
                backbone.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
                # backbone.maxpool = nn.Identity()
                # if input_shape[0] != 3:
                #     backbone.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False)
                # Remove the fully connected layer
                backbone_output_size = backbone.fc.in_features
                print('backbone_output_size: ', backbone_output_size)
                backbone = nn.Sequential(*list(backbone.children())[:-1])
            elif backbone_type == 'convnext_tiny':
                backbone = models.convnext_tiny(pretrained=pretrained)
                backbone_output_size = backbone.classifier[2].in_features
                backbone.classifier = nn.Identity()

                # Modify the first convolutional layer to work with smaller resolutions
                backbone.features[0][0] = nn.Conv2d(
                    in_channels=input_shape[0],
                    out_channels=backbone.features[0][0].out_channels,
                    kernel_size=3,  # Reduce kernel size to 3x3
                    stride=1,       # Reduce stride to 1 to preserve spatial resolution
                    padding=1,      # Add padding to preserve dimensions after convolution
                    bias=True # False
                )
            elif backbone_type == 'efficientnet_v2_s':
                backbone = models.efficientnet_v2_s(pretrained=pretrained)
                backbone.features[0][0] = nn.Conv2d(input_shape[0], 24, kernel_size=3, stride=1, padding=1, bias=False)
                backbone_output_size = backbone.classifier[1].in_features
                backbone.classifier = nn.Identity()
            elif backbone_type == 'vit_b_16':
                backbone = models.vision_transformer.vit_b_16(pretrained=pretrained)

                # Add a resize layer to ensure the input is correctly sized for ViT
                resize_layer = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)

                backbone_output_size = backbone.heads.head.in_features
                backbone.heads.head = nn.Identity()

                # Combine the resize layer and the backbone into a sequential model
                backbone = nn.Sequential(resize_layer, backbone)
                # # Assuming your input image is a tensor or PIL image, resize it to 224x224
                # #obs = self.resize_transform(obs)
                # backbone = models.vision_transformer.vit_b_16(pretrained=pretrained)

                # backbone_output_size = backbone.heads.head.in_features
                # backbone.heads.head = nn.Identity()
            else:
                raise ValueError(f'Unknown backbone type: {backbone_type}')

            # Optionally freeze the follow-up layers, leaving the first convolutional layer unfrozen
            if backbone_params.get('freeze', False):
                print('Freezing backbone')
                for name, param in backbone.named_parameters():
                    if 'conv1' not in name:  # Ensure the first conv layer is not frozen
                        param.requires_grad = False

            return backbone, backbone_output_size

        def is_separate_critic(self):
            return False

        def is_rnn(self):
            return self.has_rnn

        def get_default_rnn_state(self):
            num_layers = self.rnn_layers
            if self.rnn_name == 'lstm':
                return (torch.zeros((num_layers, self.num_seqs, self.rnn_units)),
                            torch.zeros((num_layers, self.num_seqs, self.rnn_units)))
            else:
                return (torch.zeros((num_layers, self.num_seqs, self.rnn_units)))

    def build(self, name, **kwargs):
        net = VisionBackboneBuilder.Network(self.params, **kwargs)
        return net