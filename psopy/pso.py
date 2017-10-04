"""
pso is an implementation of the Particle Swarm Optimization (PSO) method.

The goal of the pso method is to take a set of such candidates, and through its
algorithm, search a certain space for a better solution. Consider the function
x^2 (x squared). The minimum of this one dimensional function is at 0.0, as no
other value can be given to the function to produce a lower result. The PSO
method attempts to systematically and without any additional input find th
solution.

In order to do this, we start by defining the function which we wish to
optimize, as well as define the boundaries of the search space. For instance,
for our problem, we will constrain the search are on the space of real numbers
from -100.0 to 100.0.

Now, we start with a given set of (typically) random guesses of candidates for
where we think the minimum would be (pretending that we do not already know).
Say that we randomly choose the set of candidates to be [1.2, -4.3, -9.7, 2.0].
PSO method will attempt to find the minimum (0.0 in this case) by successive
evaluation of the function, and the interaction between the set of these
candidates.
"""
# To ignore pylint's "numpy does not contains * member" errors:
# pylint: disable=E1101
import math
import random
import numpy as np


def solver(candidate, **kwargs):
    """The main solver for this module"""
    swarm = Swarm(candidate, **kwargs)
    swarm.run_iterations()
    return swarm.gbest_fit, swarm.gbest_pos

class Swarm(object):
    """
    Swarm is the primary data structure of the PSO package, representing a set
    of candidates (potential solutions), which it randomly scatters around the
    configuration space of the problem.

    The Swarm is also responsible for all the bookkeeping needed in a typical
    PSO run, such as the current and best position of each candidate solution,
    the current global best solution, and so on.

    Finally, the Swarm also contains the various settings/configurations for
    the implementation of the PSO algorithm to use, the neighborhood topology,
    the social and cognitive acceleration coefficients, and so on. All of the
    various settings and methods of the Swarm can be adjusted to implement a
    wide range of the various PSO algorithms with minimal work
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, candidate, **kwargs):
        self.max_iterations = kwargs.get('max_iterations', 1000)
        self.current_iteration = kwargs.get('current_iteration', 0)
        self.num_candidates = kwargs.get('num_candidates', 20)
        assert (self.num_candidates % 10 == 0), "Number of candidates must be divisble by 10"
        self.verbose = kwargs.get('verbose', False)

        self.candidate = candidate
        lower_bounds, upper_bounds = self.candidate.boundaries()
        assert len(lower_bounds) == len(upper_bounds), "Length of upper and lower bounds must match"
        assert np.all(np.greater(upper_bounds, lower_bounds)), "Upper bounds less than lower bounds"
        self.num_dims = len(upper_bounds)

        self.fit = np.zeros(shape=(self.num_candidates), dtype=np.float)
        self.best_fit = np.zeros(shape=(self.num_candidates), dtype=np.float)
        self.target = np.zeros(shape=(self.num_candidates), dtype=np.int)
        self.gbest_id = None
        self.gbest_fit = None
        self.gbest_pos = None

        self.pos = np.random.rand(self.num_candidates, self.num_dims)
        self.pos = self.pos * (lower_bounds - upper_bounds) + upper_bounds
        self.fit = self.candidate.eval_fitness(self.pos)
        self.best_pos = np.copy(self.pos)
        self.best_fit = np.copy(self.fit)
        self.velocity = np.zeros(shape=(self.num_candidates, self.num_dims), dtype=float)
        self.find_gbest()
        return None

    def find_gbest(self):
        """Find the global best candidate in this iteration."""
        self.gbest_id = 0
        self.gbest_fit = self.best_fit[0]
        if self.extrema == "min":
            for i in range(self.num_candidates):
                if self.best_fit[i] < self.gbest_fit:
                    self.gbest_fit = self.best_fit[i]
                    self.gbest_id = i
            self.gbest_pos = np.copy(self.best_pos[self.gbest_id])
        elif self.extrema == "max":
            for i in range(self.num_candidates):
                if self.best_fit[i] > self.gbest_fit:
                    self.gbest_fit = self.best_fit[i]
                    self.gbest_id = i
            self.gbest_pos = np.copy(self.best_pos[self.gbest_id])

        return None

    def run_iterations(self):
        """Run all iterations using the PSO method."""
        while self.current_iteration < self.max_iterations:
            self.iterate()
            self.current_iteration += 1
        return None

    def iterate(self):
        """Run a single PSO iteration."""
        self.update_targets()
        self.update_velocity()
        self.update_position()
        self.check_boundaries()
        self.get_fitness()
        self.update_personal_bests()
        self.find_gbest()
        if self.verbose:
            average_fitness = np.sum(self.fit) / len(self.fit)
            average_best_fitness = np.sum(self.best_fit) / len(self.best_fit)
            print(self.fit)
            print(self.best_fit)
            print("Finished with iteration", self.current_iteration)
            print("Global best:", self.gbest_id, "\tfitness:", self.gbest_fit)
            print("The average fitness in this iteration is", average_fitness)
            print("The average best fitness over all iterations is", average_best_fitness)
        return None

    def update_targets(self):
        """Update the targets for all candidates"""
        topology = self.topology
        if topology == "global":
            self.update_targets_global()
        elif topology == "random":
            self.update_targets_random()
        elif topology == "von neumann":
            self.update_targets_von_neumann()
        else:
            raise NotImplementedError("Unknown topology:", topology)
        return None

    def update_targets_global(self):
        """The specific target update for the global topology"""
        pso_type = self.pso_type
        if pso_type == "standard" or pso_type == "constriction":
            self.target[...] = self.gbest_id
        elif pso_type == "fdr" or pso_type == "fdrs":
            for i in range(self.num_candidates):
                fdr, next_fdr = 0.0, 0.0
                if i == 0:
                    self.target[i] = 0
                    fdr = 1000000000000000000.0
                else:
                    self.target[i] = 0
                    fdr = self.best_fit[0] - self.best_fit[i]
                    fdr /= np.sum(np.power((self.best_pos[0] - self.best_pos[i]), 2))
                for j in range(1, self.num_candidates):
                    if i == j:
                        continue
                    next_fdr = self.best_fit[j] - self.best_fit[i]
                    next_fdr /= np.sum(np.power((self.best_pos[j] - self.best_pos[i]), 2))
                    if self.extrema == 'min':
                        if fdr > next_fdr:
                            fdr = next_fdr
                            self.target[i] = j
                    elif self.extrema == 'max':
                        if fdr < next_fdr:
                            fdr = next_fdr
                            self.target[i] = j
        else:
            raise NotImplementedError('Global topology not implemented for pso type:', pso_type)
        return None

    def update_targets_random(self):
        """The specific target update for the random topology"""
        for i in range(self.num_candidates):
            # randomly pick a target that is not the candidate itself.
            while True:
                target = random.randint(0, self.num_candidates-1)
                if target != i:
                    break

            # we want to accelerate toward the target if it is better, and
            # away from it if it worse. In order to denote a worse candidate,
            # we set the target's index to negative so that we know later
            # to accelerate away.
            if self.extrema == "min":
                if self.fit[i] > self.fit[target]:
                    self.target[i] = target
                else:
                    self.target[i] = -target
            elif self.extrema == "max":
                if self.fit[i] < self.fit[target]:
                    self.target[i] = target
                else:
                    self.target[i] = -target
        return None

    def update_targets_von_neumann(self):
        """The specific target update for the von neumann topology"""

        for i in range(self.num_candidates):
            neighbor = self.get_von_neumann_neighbors(i)
            pso_type = self.pso_type
            if pso_type == "standard" or pso_type == "constriction":
                self.target[i] = neighbor[0]
                neighbor_fitness = self.best_fit[neighbor[0]]
                for i in range(1, 4):
                    if self.extrema == 'min':
                        if self.best_fit[neighbor[i]] < neighbor_fitness:
                            self.target[i] = neighbor[i]
                            neighbor_fitness = self.best_fit[neighbor[i]]
                    elif self.extrema == 'max':
                        if self.best_fit[neighbor[i]] > neighbor_fitness:
                            self.target[i] = neighbor[i]
                            neighbor_fitness = self.best_fit[neighbor[i]]
            elif pso_type == "fdr" or pso_type == "fdrs":
                fdr, next_fdr = 0.0, 0.0
                if i == 0:
                    self.target[i] = neighbor[0]
                    fdr = 1000000000000000000.0
                else:
                    self.target[i] = neighbor[0]
                    fdr = self.best_fit[neighbor[0]] - self.best_fit[i]
                    fdr /= np.sum(np.power((self.best_pos[neighbor[0]] - self.best_pos[i]), 2))
                for j in range(1, 4):
                    next_fdr = self.best_fit[neighbor[j]] - self.best_fit[i]
                    next_fdr /= np.sum(np.power((self.best_pos[neighbor[j]] - self.best_pos[i]), 2))
                    if self.extrema == 'min':
                        if fdr > next_fdr:
                            fdr = next_fdr
                            self.target[i] = neighbor[j]
                    elif self.extrema == 'max':
                        if fdr < next_fdr:
                            fdr = next_fdr
                            self.target[i] = j
            else:
                raise NotImplementedError('von neumann topology for pso type:', pso_type)
        return None

    def get_von_neumann_neighbors(self, candidate_number):
        """This method gets the von neumann neighbors of a given candidate in the swarm"""
        neighbor = np.zeros(shape=(4), dtype=int)
        row_length = int(self.num_candidates/10)  # int casting in case of python 2 use.
        if candidate_number - row_length < 0:  # at the top of grid
            neighbor[0] = self.num_candidates - row_length + 1
        else:
            neighbor[0] = candidate_number - row_length
        if (candidate_number+1)%row_length == 0:  # at the right
            neighbor[1] = candidate_number - (row_length - 1)
        else:
            neighbor[1] = candidate_number + 1
        if (candidate_number+row_length) >= self.num_candidates:  # at bottom of grid
            neighbor[2] = candidate_number - self.num_candidates + row_length
        else:
            neighbor[2] = candidate_number + row_length
        if candidate_number%row_length == 0:
            neighbor[3] = candidate_number + row_length - 1
        else:
            neighbor[3] = candidate_number -1
        return neighbor



    def check_boundaries(self):
        """Check if candidates left the search space, and bring them back."""
        for i in range(self.num_candidates):
            lower_bounds, upper_bounds = self.candidate.boundaries()
            for j, _ in enumerate(upper_bounds):
                if self.pos[i][j] > upper_bounds[j]:
                    self.pos[i][j] = upper_bounds[j]
                    self.velocity[i][j] = 0.0
                if self.pos[i][j] < lower_bounds[j]:
                    self.pos[i][j] = lower_bounds[j]
                    self.velocity[i][j] = 0.0
        return None

    def get_fitness(self):
        """evaluate the fitness of the candidates in the current position."""
        self.fit = self.candidate.eval_fitness(self.pos)
        return None
