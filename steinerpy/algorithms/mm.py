from steinerpy.framework import Framework

class MM(Framework):
    def __init__(self, G, T):
        self.terminals = T
        self.graph = G

        self.completeTree = False
        self.pathConverged = False

    def nominate(self):
        pass

    def update(self):
        pass

    def path_convergence_check(self):
        pass

    def tree_update_check(self):
        pass

    def f_costs_func(self):
        pass

    def h_costs_func(self):
        pass

    