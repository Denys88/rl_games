import time


class Timers:
    def __init__(self):
        self.timers = {}

    def start(self, name):
        self.timers[name] = time.time()

    def stop(self, name):
        return time.time() - self.timers[name]
