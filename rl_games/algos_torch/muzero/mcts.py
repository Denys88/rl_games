class MCTS:
    def __init__(self, c_puct=1.0):
        self.c_puct = c_puct
        self.visit_count = {}
        self.value = {}
        self.avg_value = {}
        self.prior_probs = {}

    def clear(self):
        self.visit_count.clear()
        self.value.clear()
        self.avg_value.clear()
        self.prior_probs.clear()       