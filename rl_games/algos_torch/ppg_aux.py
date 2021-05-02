

class PPGAuxLoss:
    def __init__(self, config, kl_div_func, writter)
        self.config = config
        self.kl_div_func = kl_div_func
        self.writter = writter
        self.mini_epoch = config['mini_epochs']
        self.mini_batch = config['minibatch_size']
        