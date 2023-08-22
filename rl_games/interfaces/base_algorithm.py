from abc import ABC
from abc import abstractmethod, abstractproperty


class BaseAlgorithm(ABC):
    def __init__(self, base_name, config):
        pass

    @abstractproperty
    def device(self):
        pass

    @abstractmethod
    def clear_stats(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def train_epoch(self):
        pass

    @abstractmethod
    def get_full_state_weights(self):
        pass

    @abstractmethod
    def set_full_state_weights(self, weights, set_epoch):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_weights(self, weights):
        pass

    # Get algo training parameters
    @abstractmethod
    def get_params(self, param_name):
        pass

    # Set algo training parameters
    @abstractmethod
    def set_params(self, param_name, param_value):
        pass


