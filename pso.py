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


class Candidate(object):
    """
    Canidate is a potential extrema in a n-dimentional space.

    In order to optimize a function using this module, the user must provide
    an implementation of the Candidate class that has the methods eval_fitness()
    and boundaries().
    """

    def eval_fitness(self, pos):
        """
        eval_fitness takes a position in the n-dimensional configuration space
        and evaluates the function to be minimized at that point, returning
        a float.
        """
        raise NotImplementedError("eval_fitness() not implemented")

    def boundaries(self):
        """
        boundaries returns two numpy arrays, for the lower and upper bounds of
        the configuration space in each dimension.

        These boundaries map the regien of space in which the extrema of the
        function to be found.
        """
        raise NotImplementedError("boundaries() not implemented")


def solver(candidates, **kwargs):
    """The main solver for this module"""
    swarm = Swarm(candidates, **kwargs)
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
    def __init__(self, candidates, **kwargs):
        self.cognative_acceleration = kwargs.get('cognative_acceleration', 2.05)
        self.social_acceleration = kwargs.get('social_acceleration', 2.05)
        self.inertial_weight = kwargs.get('inertial_weight', 0.9)
        self.pso_type = kwargs.get('pso_type', 'constriction')
        self.topology = kwargs.get('topology', 'global')
        self.max_iterations = kwargs.get('max_iterations', 1000)
        self.current_iteration = kwargs.get('current_iteration', 0)
        self.velocity_max = kwargs.get('velocity_max', 0.1)
        self.extrema = kwargs.get('extrema', 'min')
        self.verbose = kwargs.get('verbose', False)

        self.candidates = candidates
        self.pos = []
        self.best_pos = []
        self.velocity = []
        self.fit = np.zeros(shape=(len(candidates)), dtype=np.float)
        self.best_fit = np.zeros(shape=(len(candidates)), dtype=np.float)
        self.target = np.zeros(shape=(len(candidates)), dtype=np.int)
        self.gbest_id = None
        self.gbest_fit = None
        self.gbest_pos = None

        for i, _ in enumerate(self.candidates):
            lower_bounds, upper_bounds = self.candidates[i].boundaries()
            assert len(lower_bounds) == len(upper_bounds)
            assert np.all(np.greater(upper_bounds, lower_bounds))
            pos = np.random.rand(len(upper_bounds))
            pos = pos * (lower_bounds - upper_bounds) + upper_bounds
            self.fit[i] = self.candidates[i].eval_fitness(pos)
            self.best_fit[i] = self.fit[i]
            self.pos.append(pos)
            self.best_pos.append(pos)
            self.velocity.append(np.zeros(len(upper_bounds)))

        self.find_gbest()
        return None

    def find_gbest(self):
        """Find the global best candidate in this iteration."""
        self.gbest_id = 0
        self.gbest_fit = self.best_fit[0]
        for i, _ in enumerate(self.candidates):
            if self.best_fit[i] < self.gbest_fit:
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
        pso_type = self.pso_type
        if topology == "global":
            if pso_type == "standard" or pso_type == "constriction":
                self.target[...] = self.gbest_id
            else:
                raise NotImplementedError("Unknown PSO type:", pso_type)
        else:
            raise NotImplementedError("Unknown topology:", topology)
        return None

    def update_velocity(self):
        """Determine the velocity of the candidates based on their targets"""
        pso_type = self.pso_type
        if pso_type == "constriction":
            phi = self.cognative_acceleration + self.social_acceleration
            chi = (2.0 / abs(2.0 - phi - math.sqrt((phi * phi) - (4.0 * phi))))
            for i, _ in enumerate(self.candidates):
                my_target = self.target[i]
                # .shape returns a tuple, so we need to unzip it with the *.
                rand_set1 = np.random.rand(*self.velocity[i].shape)
                rand_set2 = np.random.rand(*self.velocity[i].shape)
                self.velocity[i] = chi * (self.velocity[i] +
                                          (rand_set1 * self.cognative_acceleration *
                                           (self.best_pos[i] - self.pos[i])) +
                                          (rand_set2 * self.social_acceleration *
                                           (self.best_pos[my_target] - self.pos[i])))

        elif pso_type == "standard":
            self.inertial_weight = 0.5 * ((self.max_iterations - self.current_iteration) /
                                          self.max_iterations) + 0.4 * random.random()
            for i, _ in enumerate(self.candidates):
                my_target = self.target[i]
                # .shape returns a tuple, so we need to unzip it with the *.
                rand_set1 = np.random.rand(*self.velocity[i].shape)
                rand_set2 = np.random.rand(*self.velocity[i].shape)
                self.velocity[i] = (self.inertial_weight * self.velocity[i] +
                                    (rand_set1 * self.cognative_acceleration *
                                     (self.best_pos[i] - self.pos[i])) +
                                    (rand_set2 * self.social_acceleration *
                                     (self.best_pos[my_target] - self.pos[i])))
        else:
            raise NotImplementedError("Unknown PSO type:", self.pso_type)

        for i, _ in enumerate(self.candidates):
            for i, _ in enumerate(self.candidates):
                self.velocity[i][self.velocity[i] > self.velocity_max] = self.velocity_max
                self.velocity[i][self.velocity[i] < self.velocity_max] = -self.velocity_max

        return None

    def update_position(self):
        """Move the candidates based on their velocities"""
        for i, _ in enumerate(self.candidates):
            self.pos[i] += self.velocity[i]
        return None

    def check_boundaries(self):
        """Check if candidates left the search space, and bring them back."""
        for i, _ in enumerate(self.candidates):
            lower, upper = self.candidates[i].boundaries()
            for j, _ in enumerate(upper):
                if self.pos[i][j] > upper[j]:
                    self.pos[i][j] = upper[j]
                    self.velocity[i][j] = 0.0
                if self.pos[i][j] < lower[j]:
                    self.pos[i][j] = lower[j]
                    self.velocity[i][j] = 0.0
        return None

    def get_fitness(self):
        """evaluate the fitness of the candidates in the current position."""
        for i, _ in enumerate(self.candidates):
            self.fit[i] = self.candidates[i].eval_fitness(self.pos[i])
        return None

    def update_personal_bests(self):
        """Update the personal best fitness and position if better ones found"""

        for i, _ in enumerate(self.candidates):
            if self.extrema == 'min':
                if self.fit[i] < self.best_fit[i]:
                    self.best_fit[i] = self.fit[i]
                    self.best_pos[i] = np.copy(self.pos[i])
            elif self.extrema == 'max':
                if self.fit[i] > self.best_fit[i]:
                    self.best_fit[i] = self.fit[i]
                    self.best_pos[i] = np.copy(self.pos[i])
        return None
