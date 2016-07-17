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

import numpy as np
import math
import random


class Candidate(object):
    """
    Canidate is a potential extrema in a n-dimentional space.

    In order to optimize a function using this module, the user must provide
    an implementation of the Candidate class that has the methods evalFitness()
    and boundaries().
    """

    def evalFitness(self, pos):
        """
        evalFitness takes a position in the n-dimensional configuration space
        and evaluates the function to be minimized at that point, returning
        a float.
        """

        raise NotImplementedError("evalFitness() not implemented")

    def boundaries(self):
        """
        boundaries returns two numpy arrays, for the lower and upper bounds of
        the configuration space in each dimension.

        These boundaries map the regien of space in which the extrema of the
        function to be found.
        """

        raise NotImplementedError("boundaries() not implemented")


class Params(object):
    """Params is the object contains the parameters for a PSO solver."""

    def __init__(self, **kwargs):
        self.c1 = kwargs.get('c1', 2.05)
        self.c2 = kwargs.get('c2', 2.05)
        self.w = kwargs.get('w', 0.9)
        self.psoType = kwargs.get('psoType', 'Constriction')
        self.topology = kwargs.get('topology', 'Global')
        self.maxIterations = kwargs.get('maxIterations', 1000)
        self.currentIteration = kwargs.get('currentIteration', 0)
        self.extrema = kwargs.get('extrema', 'Min')
        self.verbose = kwargs.get('verbose', False)


def solver(candidates, params=Params()):
    s = Swarm(candidates, params)
    s.runIterations()
    return s.gBestFit, s.gBestPos

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

    def __init__(self, candidates, params):
        self.candidates = candidates
        self.pos = []
        self.bPos = []
        self.v = []
        self.fit = np.zeros(shape=(len(candidates)), dtype=np.float)
        self.bFit = np.zeros(shape=(len(candidates)), dtype=np.float)
        self.target = np.zeros(shape=(len(candidates)), dtype=np.int)
        self.params = params
        self.gBestID = None
        self.gBestFit = None
        self.gBestPos = None

        for i in range(len(self.candidates)):
            lowerBounds, upperBounds = self.candidates[i].boundaries()
            assert(len(lowerBounds) == len(upperBounds))
            for j in range(len(lowerBounds)):
                assert(upperBounds[j] > lowerBounds[j])
            pos = np.random.rand(len(upperBounds))
            pos = pos * (lowerBounds - upperBounds) + upperBounds
            self.fit[i] = self.candidates[i].evalFitness(pos)
            self.bFit[i] = self.fit[i]
            self.pos.append(pos)
            self.bPos.append(pos)
            self.v.append(np.zeros(len(upperBounds)))

        self.findGBest()
        return None

    def findGBest(self):
        self.gBestID = 0
        self.gBestFit = self.bFit[0]
        for i in range(len(self.candidates)):
            if self.bFit[i] < self.gBestFit:
                self.gBestFit = self.bFit[i]
                self.gBestID = i
        self.gBestPos = np.copy(self.bPos[self.gBestID])
        return None

    def runIterations(self):
        while self.params.currentIteration < self.params.maxIterations:
            self.iterate()
            self.params.currentIteration += 1
        return None

    def iterate(self):
        self.updateTargets()
        self.updateVelocity()
        self.updatePos()
        self.checkBoundaries()
        self.getFitness()
        self.updatePersonalBests()
        self.findGBest()
        if self.params.verbose:
            x1 = np.sum(self.fit) / len(self.fit)
            x2 = np.sum(self.bFit) / len(self.bFit)
            print(self.fit)
            print(self.bFit)
            print("Finished with iteration", self.params.currentIteration)
            print("Global best:", self.gBestID, "\tfitness:", self.gBestFit)
            print("The average fitness in this iteration is", x1)
            print("The average best fitness over all iterations is", x2)

    def updateTargets(self):
        if self.params.topology == "Global":
            for i in range(len(self.target)):
                self.target[i] = self.gBestID
        else:
            print("Unknown topology:", self.params.topology)
            raise

    def updateVelocity(self):
        if self.params.psoType == "Constriction":
            phi = self.params.c1 + self.params.c2
            chi = (2.0 / abs(2.0 - phi - math.sqrt((phi * phi) - (4.0 * phi))))
            for i in range(len(self.candidates)):
                for j in range(len(self.v[i])):
                    t = self.target[i]
                    self.v[i][j] = chi * (self.v[i][j] +
                                          (random.random() * self.params.c1 *
                                          (self.bPos[i][j] - self.pos[i][j])) +
                                          (random.random() * self.params.c2 *
                                          (self.bPos[t][j] - self.pos[i][j])))
        else:
            print("Unknown PSO type:", self.params.psoType)
            raise

    def updatePos(self):
        for i in range(len(self.candidates)):
            self.pos[i] += self.v[i]

    def checkBoundaries(self):
        for i in range(len(self.candidates)):
            lower, upper = self.candidates[i].boundaries()
            for j in range(len(upper)):
                if self.pos[i][j] > upper[j]:
                    self.pos[i][j] = upper[j]
                    self.v[i][j] = 0.0
                if self.pos[i][j] < lower[j]:
                    self.pos[i][j] = lower[j]
                    self.v[i][j] = 0.0

    def getFitness(self):
        for i in range(len(self.candidates)):
            self.fit[i] = self.candidates[i].evalFitness(self.pos[i])

    def updatePersonalBests(self):
        for i in range(len(self.candidates)):
            if self.fit[i] < self.bFit[i]:
                self.bFit[i] = self.fit[i]
                self.bPos[i] = np.copy(self.pos[i])
