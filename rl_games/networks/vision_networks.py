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
            self.use_aux_loss = kwargs.pop('use_aux_loss', True)
            print(kwargs)
            print("params: ", params)
            self.aux_loss_weight = 100.0 # kwargs.pop('aux_loss_weight', 1.0)
            if self.use_aux_loss:
                self.target_key = 'aux_target'
                if 'aux_target' in full_input_shape:
                    self.target_shape = full_input_shape[self.target_key]
                    print("Target shape: ", self.target_shape)

            self.proprio_size = 0 # Number of proprioceptive features
            if type(full_input_shape) is dict:
                input_shape = full_input_shape['camera']
                proprio_shape = full_input_shape['proprio']
                self.proprio_size = proprio_shape[0]
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

            mlp_input_size = cnn_output_size + self.proprio_size
            if self.use_aux_loss:
                mlp_input_size += self.target_shape[0]

            if len(self.units) == 0:
                out_size = cnn_output_size
            else:
                out_size = self.units[-1]

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
                'input_size': mlp_input_size,
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear
            }

            self.mlp = self._build_mlp(**mlp_args)

            self.aux_loss_map = None
            if self.use_aux_loss:
                print("Building aux loss")
                print("cnn_output_size: ", cnn_output_size)
                print("target_shape: ", self.target_shape)
                self.aux_loss_linear = nn.Linear(cnn_output_size, self.target_shape[0])
                self.aux_loss_map = {
                    'aux_dist_loss': None
                }

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

        def get_aux_loss(self):
            return self.aux_loss_map

        def forward(self, obs_dict):
            if self.proprio_size > 0:
                obs = obs_dict['obs']['camera']
                proprio = obs_dict['obs']['proprio']
            else:
                obs = obs_dict['obs']

            if self.use_aux_loss:
                target_obs = obs_dict['obs'][self.target_key]

            if self.permute_input:
                obs = obs.permute((0, 3, 1, 2))

            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)
            states = obs_dict.get('rnn_states', None)

            out = obs
            out = self.cnn(out)
            out = out.flatten(1)
            cnn_out = self.flatten_act(out)

            if self.proprio_size > 0:
                out = torch.cat([cnn_out, proprio], dim=1)

            if self.use_aux_loss:
                y = self.aux_loss_linear(cnn_out)
                out = torch.cat([out, y], dim=1)
                self.aux_loss_map['aux_dist_loss'] = self.aux_loss_weight * torch.nn.functional.mse_loss(y, target_obs)
                # print("aux predicted shape: ", y.shape)
                # print("aux predicted: ", y)
                # print("aux target: ", target_obs)
                # print("delta: ", y - target_obs)
                # print("aux loss: ", self.aux_loss_map['aux_dist_loss'])

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

            self.use_aux_loss = kwargs.pop('use_aux_loss', False)
            self.aux_loss_weight = 100.0
            if self.use_aux_loss:
                self.target_key = 'aux_target'
                if 'aux_target' in full_input_shape:
                    self.target_shape = full_input_shape[self.target_key]
                    print("Target shape: ", self.target_shape)

            print("Observations shape: ", full_input_shape)
            print("Use aux loss: ", self.use_aux_loss)

            self.proprio_size = 0 # Number of proprioceptive features
            if isinstance(full_input_shape, dict):
                input_shape = full_input_shape['camera']
                proprio_shape = full_input_shape['proprio']
                self.proprio_size = proprio_shape[0]

                # # TODO: This is a hack to get the target shape
                # for k, v in full_input_shape.items():
                #     if self.target_key == k:
                #         self.target_shape = v[0]
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
            if self.use_aux_loss:
                mlp_input_size += self.target_shape[0]

            if len(self.units) == 0:
                out_size = self.cnn_output_size
            else:
                out_size = self.units[-1]

            self.layer_norm_emb = torch.nn.LayerNorm(mlp_input_size)

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

            self.aux_loss_map = None
            if self.use_aux_loss:
                self.aux_loss_linear = nn.Linear(self.cnn_output_size, self.target_shape[0])
                self.aux_loss_map = {
                    'aux_dist_loss': None
                }

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

        def get_aux_loss(self):
            return self.aux_loss_map

        def forward(self, obs_dict):
            if self.proprio_size > 0:
                obs = obs_dict['obs']['camera']
                proprio = obs_dict['obs']['proprio']
            else:
                obs = obs_dict['obs']

            if self.use_aux_loss:
                target_obs = obs_dict['obs'][self.target_key]

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
            vis_out = self.flatten_act(out)

            if self.proprio_size > 0:
                out = torch.cat([vis_out, proprio], dim=1)

            if self.use_aux_loss:
                y = self.aux_loss_linear(vis_out)
                out = torch.cat([out, y], dim=1)
                self.aux_loss_map['aux_dist_loss'] = self.aux_loss_weight * torch.nn.functional.mse_loss(y, target_obs)

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

        def _build_backbone(self, input_shape, backbone_params):
            backbone_type = backbone_params.get('type', 'resnet18')
            pretrained = backbone_params.get('pretrained', True)
            modify_first_layer = backbone_params.get('modify_first_layer', False)
            self.preprocess_image = backbone_params.get('preprocess_image', False)

            # Define a mapping from backbone type to required resize size
            resize_size_map = {
                'vit_b_16': 224,           # Must be divisible by 16
                'dinov2_vits14_reg': 196,  # Must be divisible by 14
                'vit_mae': 224,            # Must be divisible by 16
                'vit_tiny': 224,           # ViT-Tiny adjusted for 16x16 patch size and smaller input
                'deit_tiny': 224,          # DeiT-Tiny adjusted similarly
                'deit_tiny_distilled': 224, # DeiT-Tiny distilled version
                'mobilevit_s': 256,        # MobileViT-S
                'efficientformer_l2': 224, # EfficientFormer-L2
                'swinv2': 224,             # Swin Transformer
                # Add other ViT variants as needed
            }

            backbone = None
            backbone_output_size = None
            resize_size = None

            if backbone_type.startswith('resnet'):
                # ResNet handling...
                try:
                    backbone_class = getattr(models, backbone_type)
                    backbone = backbone_class(pretrained=pretrained)
                except AttributeError:
                    raise ValueError(f'Unknown ResNet model: {backbone_type}')

                if modify_first_layer:
                    backbone.conv1 = nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1, bias=False)
                elif input_shape[0] != 3:
                    backbone.conv1 = nn.Conv2d(
                        input_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False
                    )
                backbone_output_size = backbone.fc.in_features
                backbone = nn.Sequential(*list(backbone.children())[:-1])

            elif backbone_type == 'convnext_tiny':
                # ConvNeXt handling...
                backbone = models.convnext_tiny(pretrained=pretrained)
                backbone_output_size = backbone.classifier[2].in_features
                backbone.classifier = nn.Identity()

                backbone.features[0][0] = nn.Conv2d(
                    in_channels=input_shape[0],
                    out_channels=backbone.features[0][0].out_channels,
                    kernel_size=4,   # Adjust kernel size as needed
                    stride=4,        # Adjust stride as needed
                    padding=0,       # Adjust padding as needed
                    bias=False
                )

            elif backbone_type == 'efficientnet_v2_s':
                # EfficientNet handling...
                backbone = models.efficientnet_v2_s(pretrained=pretrained)
                backbone_output_size = backbone.classifier[1].in_features
                backbone.classifier = nn.Identity()

                backbone.features[0][0] = nn.Conv2d(
                    input_shape[0], backbone.features[0][0].out_channels,
                    kernel_size=3, stride=1, padding=1, bias=False
                )

            elif backbone_type.lower() in resize_size_map:
                import timm

                # Unified ViT handling for various ViT models
                resize_size = resize_size_map.get(backbone_type.lower(), 224)  # Default to 224 if not specified

                if backbone_type.lower() == 'vit_tiny':
                    backbone = timm.create_model('vit_tiny_patch16_224', pretrained=pretrained)
                    backbone_output_size = backbone.head.in_features
                elif backbone_type.lower() == 'deit_tiny':
                    backbone = timm.create_model('deit_tiny_patch16_224', pretrained=pretrained)
                    backbone_output_size = backbone.head.in_features
                elif backbone_type.lower() == 'deit_tiny_distilled':
                    backbone = timm.create_model('deit_tiny_distilled_patch16_224', pretrained=pretrained,)
                    backbone_output_size = backbone.head.in_features
                elif backbone_type.lower() == 'mobilevit_s':
                    print("Not working")
                    backbone = timm.create_model('mobilevit_s', pretrained=pretrained)
                    print(backbone)
                    backbone_output_size = backbone.head.fc.in_features
                    print(backbone_output_size)
                elif backbone_type.lower() == 'efficientformer_l2':
                    backbone = timm.create_model('efficientformerv2_s1.snap_dist_in1k', pretrained=pretrained)
                    # print(backbone)
                    # print(backbone.pos_embed.shape)
                    backbone_output_size = backbone.head.in_features
                elif backbone_type.lower() == 'vit_b_16':
                    backbone = models.vision_transformer.vit_b_16(pretrained=pretrained)
                    backbone_output_size = backbone.heads.head.in_features
                elif backbone_type.lower() == 'dinov2_vits14_reg':
                    try:
                        backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', pretrained=pretrained)
                    except Exception as e:
                        raise ValueError(f"Failed to load dinov2_vits14_reg: {e}")
                    backbone_output_size = 384  # As per Dinov2 ViT-S14 regression output
                elif backbone_type.lower() == 'vit_mae':
                    try:
                        backbone = timm.create_model('vit_base_patch16_224.mae', pretrained=pretrained)
                    except Exception as e:
                        raise ValueError(f"Failed to load vit_mae: {e}")
                    backbone_output_size = 768 # Typically 768 for ViT-Base
                elif backbone_type.lower() == 'swinv2':
                    print("Not working")
                    try:
                        backbone = timm.create_model('swinv2_cr_tiny_ns_224.sw_in1k', pretrained=pretrained)
                    except Exception as e:
                        raise ValueError(f"Failed to load swinv2_cr_tiny_ns_224.sw_in1k: {e}")
                    backbone_output_size = 768  # Typically 768 for Swin Transformer
                else:
                    raise ValueError(f'Unknown ViT model type: {backbone_type}')

                # Remove the classification/regression head if present
                if hasattr(backbone, 'heads') and hasattr(backbone.heads, 'head'):
                    backbone.heads.head = nn.Identity()
                elif hasattr(backbone, 'head'):
                    backbone.head = nn.Identity()
                elif hasattr(backbone, 'decoder'):  # For MAE models
                    backbone.decoder = nn.Identity()
                else:
                    print(f"Unable to locate the classification/regression head in {backbone_type} model.")

                # Add a resize layer to ensure the input is correctly sized for the specific ViT model
                resize_layer = nn.Upsample(size=(resize_size, resize_size), mode='bilinear', align_corners=False)

                # Combine the resize layer and the backbone into a sequential model
                backbone = nn.Sequential(resize_layer, backbone)

            else:
                raise ValueError(f'Unknown backbone type: {backbone_type}')

            # Optionally freeze layers except specified layers (for ResNet, ConvNeXt, EfficientNet, ViT)
            if backbone_params.get('freeze', False):
                print('Freezing backbone layers except specified layers')
                for name, param in backbone.named_parameters():
                    if 'vit' in backbone_type.lower():
                        # For all ViT backbones, avoid freezing 'patch_embed.proj' layer
                        if not ('patch_embed.proj' in name):
                            param.requires_grad = False
                    else:
                        # General case for other backbones (e.g., ResNet, ConvNeXt)
                        if 'conv1' not in name and 'features.0.0' not in name:
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