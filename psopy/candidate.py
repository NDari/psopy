"""
TODO: candidate doc
"""
from abc import ABC, abstractmethod
from math import sqrt
import numpy as np

class Candidate(ABC):
    """ Candidate is a potential extrema in a n-dimensional space. """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, position, **kwargs):
        self.cognative_acceleration = kwargs.get('cognative_acceleration', 2.05)
        self.social_acceleration = kwargs.get('social_acceleration', 2.05)
        self.chi = kwargs.get('chi', self.calculate_chi)
        self.pso_type = kwargs.get('pso_type', 'constriction')
        self.extrema = kwargs.get('extrema', 'min')
        self.upper_bounds = kwargs.get('upper_bounds', None)
        self.lower_bounds = kwargs.get('lower_bounds', None)
        self.position = position
        self.best_position = position
        self.target_position = None
        self.fitness = None
        self.best_fitness = None
        self.target_fitness = None
        self.velocity = np.zeros_like(position)
        self.enforce_input_correctness

    @abstractmethod
    def eval(self):
        """
        eval_fitness takes a set of positions in the n-dimensional configuration
        space and evaluates the function to be optimized at that each point,
        returning a numpy float array. Each position in the set represents
        the location of a candidate in the configuration space. Therefore, the
        length of the returned numpy array must match the number of candidates
        whole positions were passed to this function. The default num_candidates
        is set to 20.
        """
        pass

    def set_target(self, target_candidate=None):
        """
        Sets the target for this candidate. This method should be called in
        each iteration so that the social acceleration can be calculated.
        It is sometimes required (based on the PSO type used) that a candidate
        has no suitable target. In these cases, calling this method without
        a target will achieve the desired result.
        """
        if target_candidate is None:
            self.target_fitness = self.best_fitness
            self.target_position = self.best_position.view()
        else:
            self.target_fitness = target_candidate.best_fitness
            self.target_position = target_candidate.best_position.view()

    def iterate_once(self):
        """
        Run a single PSO iteration for this candidate. This is a convenience method
        which is the same as calling the following methods in order:
            update_velocity()
            update_position()
            apply_boundary_conditions()
            update_personal_best()

        Note that this method expects the target to be set correctly for this
        iteration. For more information, look into the "set_target".
        """
        self.update_velocity()
        self.update_position()
        self.apply_boundary_conditions()
        self.update_personal_best()


    def update_velocity(self):
        """ Determine the velocity of the candidate based on the target """
        # determine if we need to move toward or away from the target.
        direction = -1.0 if self.target_is_less_fit else 1.0

        # .shape returns a tuple, so we need to unzip it with the *.
        rand_set1 = self.cognative_acceleration * np.random.rand(*self.velocity.shape)
        rand_set2 = self.social_acceleration * np.random.rand(*self.velocity.shape)
        cognative_a = rand_set1 * (self.best_position - self.position)
        social_a = (rand_set2 * (self.target_position - direction * self.position))

        self.velocity = self.chi * (self.velocity + cognative_a + social_a)

    def update_position(self):
        """ The position update equation. """
        np.add(self.position, self.velocity, self.position)

    def apply_boundary_conditions(self):
        """
        If the search space is constrained (i.e. upper_bounds and lower_bounds
        were passed to the constructor), check the position of the candidate in
        each dimension. If we have gone beyond the search boundary, set the
        position to the edge, and set the velocity in that dimension to 0.0
        """
        if not self.has_defined_boundaries:
            return
        self.velocity = np.where(self.position > self.upper_bounds, 0.0, self.velocity)
        self.velocity = np.where(self.position < self.lower_bounds, 0.0, self.velocity)
        self.position = np.where(self.position > self.upper_bounds, self.upper_bounds, self.position)
        self.position = np.where(self.position < self.lower_bounds, self.lower_bounds, self.position)

    def update_personal_best(self):
        """ Update the personal best fitness and position if better ones found """
        if self.better_personal_fitness_found:
            np.copyto(self.best_position, self.position)
            self.best_fitness = self.fitness

    @property
    def calculate_chi(self):
        """ Calculate chi based on congnative and social accelerations. """
        phi = self.cognative_acceleration + self.social_acceleration
        return (2.0 / abs(2.0 - phi - sqrt((phi * phi) - (4.0 * phi))))

    @property
    def is_minimizing(self):
        """ Are we minimizing this candidate? """
        return self.extrema == 'min'

    @property
    def is_maximizing(self):
        """ Are we maximizing this candidate? """
        return self.extrema == 'max'

    @property
    def target_is_less_fit(self):
        """ Is the target's fitness worse than the current fitness? """
        if self.is_minimizing and self.target_fitness > self.fitness:
            return True
        elif self.is_maximizing and self.fitness > self.target_fitness:
            return True
        return False

    @property
    def better_personal_fitness_found(self):
        """ Is the current fitness better than the best fitness found thus far? """
        if self.is_minimizing and self.fitness < self.best_fitness:
            return True
        if self.is_maximizing and self.fitness > self.best_fitness:
            return True
        return False

    @property
    def has_defined_boundaries(self):
        """ Is the search space unconstrained? """
        return self.upper_bounds != None and self.lower_bounds != None

    @property
    def enforce_input_correctness(self):
        """ Make sure that the inputs are of the correct values and shapes """
        if self.has_defined_boundaries:
            assert self.lower_bounds.shape == self.upper_bounds.shape
            assert self.position.shape == self.upper_bounds.shape
            assert np.all(np.greater(self.upper_bounds, self.lower_bounds))
            assert np.all(np.greater(self.position, self.lower_bounds))
            assert np.all(np.less(self.position, self.upper_bounds))

