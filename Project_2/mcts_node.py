class MCTSNode:
    """
    Class for a single node in the search tree.
    """

    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0

        # The prob. of choosing this node (from the parent)
        self.prior = prior

        self.children = {}
        self.game_state = None
        self.reward = 0

    def is_expanded(self):
        return len(self.children) > 0

    # Return the average value of this node (Q(s, a))
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
