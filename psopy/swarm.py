"""
TODO: swarm doc
"""

class Swarm():
    """
    Swarm is a collection of interacting candidates which try to find the solution
    to a given problem by interacting with one another.
    """

    def __init__(self, queen, **kwargs):
        self.max_iterations = kwargs.get('max_iterations', 1000)
        self.current_iteration = kwargs.get('current_iteration', 0)
        self.topology = kwargs.get('topology', 'global')
        self.pso_type = kwargs.get('pso_type', 'constriction')
        self.verbose = kwargs.get('verbose', False)
        self.num_candidates = len(candidates)

        self.queen = queen
        self.best_fitness = None
        self.targets = []
        self.global_best_candidate = None
        self.glocal_best_fitness = None
        self.global_best_position = None

