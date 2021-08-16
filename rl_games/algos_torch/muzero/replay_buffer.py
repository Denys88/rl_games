class ReplayBuffer(object):
    def __init__(self, config):
        self.window_size = config['window_size']
        self.batch_size = config['batch_size']
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)