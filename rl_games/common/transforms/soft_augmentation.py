


class SoftAugmentation():
    def __init__(self, action_space, **kwargs):
        self.action_space = action_space
        self.transform = kwargs.pop('transform', None)

    def get_loss(p_dict, model, input_dict, loss_type):
        '''
        loss_type: 'critic', 'policy', 'both'
        '''
        if self.transform:
            input_dict = self.transform(input_dict)
        loss = 0
        with torch.no_grad():
            q_dict = self.model(input_dict)

        if loss_type == 'policy' or loss_type == 'both':
            loss += model.kl(p_dict, q_dict)
        if loss_type == 'critic' or loss_type == 'both':
            p_value = p_dict['value']
            q_value = q_dict['value']
            loss += (p_value - q_value)**2
        