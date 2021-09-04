from algos_torch.network_builder import NetworkBuilder
import torch
form torch import nn
class MuzeroBuilder(NetworkBuilder):
    def __init__(self, **kwargs):
        NetworkBuilder.__init__(self)

    def load(self, params):
        self.params = params
    
    def build(self, name, **kwargs):
        net = MuzeroBuilder.Network(self.params, **kwargs)
        return net
    
    class Network(NetworkBuilder.BaseNetwork):
        def __init__(self, params, **kwargs):
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            obs_dim = kwargs.pop('obs_dim')
            action_dim = kwargs.pop('action_dim')


            NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            mlp_input_shape = input_shape


            actor_mlp_args = {
                'input_size' : obs_dim, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }

            critic_mlp_args = {
                'input_size' : obs_dim + action_dim, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                'dense_func' : torch.nn.Linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            
            self.prediction = self._build_prediciton(self.params['prediction'])
            self.dynamics = self._build_dynamics(self.params['dynamics'])
            self.representation = self._build_representation(self.params['representation'])
"""
            mlp_init = self.init_factory.create(**self.initializer)
            for m in self.modules():         
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)    
"""



        def _build_prediction(self, **kwargs):
            pass

        def _build_dynamics(self, **kwargs):
            pass 

        def forward(self, obs_dict):
            pass
        

        def load(self, params):
            self.units = params['mlp']['units']
            self.activation = params['mlp']['activation']
            self.initializer = params['mlp']['initializer']
            self.is_d2rl = params['mlp'].get('d2rl', False)
            self.norm_only_first_layer = params['mlp'].get('norm_only_first_layer', False)
            self.value_activation = params.get('value_activation', 'None')
            self.normalization = params.get('normalization', None)
            self.has_space = 'space' in params
            self.value_shape = params.get('value_shape', 1)
            self.central_value = params.get('central_value', False)
            self.joint_obs_actions_config = params.get('joint_obs_actions', None)
            self.log_std_bounds = params.get('log_std_bounds', None)

            if self.has_space:
                self.is_discrete = 'discrete' in params['space']
                self.is_continuous = 'continuous'in params['space']
                if self.is_continuous:
                    self.space_config = params['space']['continuous']
                elif self.is_discrete:
                    self.space_config = params['space']['discrete']
            else:
                self.is_discrete = False
                self.is_continuous = False