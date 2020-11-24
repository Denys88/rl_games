import torch
import ray


class SharedGradients:
    def __init__(self, shared_model, optimizer):
        self.shared_model = shared_model
        self.optimizer = optimizer
        self.workers_num = 0
        self.device = next(shared_model.parameters()).device
    def zero_grads(self):
        self.workers_num = 0
        for param in self.shared_model.parameters():
            param.grad = None


    def add_gradients(self, gradients):
        self.workers_num += 1
        for grads, shared_param in zip(gradients, self.shared_model.parameters()):
            if shared_param.grad is not None:
                shared_param.grad += grads
            else:
                shared_param._grad = grads.to(self.device)

    def update_gradients(self):
        for shared_param in self.shared_model.parameters():
            if shared_param.grad is not None:
                shared_param.grad /= self.workers_num

        self.optimizer.step()







