import copy

class NetworkStorage(object):
    def __init__(self):
        self._networks = {}

    def latest_network(self):
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            return None

    def save_network(self, step: int, network):
        self._networks[step] = copy.deepcopy(network)