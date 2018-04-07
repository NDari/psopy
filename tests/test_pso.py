# To ignore pylint's "numpy does not contains * member" errors:
# pylint: disable=E1101
import unittest
import numpy as np
#from psopy import Candidate, solver
from context import pso


class CandidateHelper(pso.Candidate):
    def eval_fitness(self, pos):
        num_candidates = pos.shape[0]
        fit = np.zeros(shape=(num_candidates), dtype=float)
        for i in range(num_candidates):
            fit[i] = np.sum(np.power(pos[i], 2))
        return fit

    def boundaries(self):
        return np.array([-5.0, -5.0]), np.array([5.0, 5.0])

class BadCandidateHelper(CandidateHelper):
    def boundaries(self):
        lower_bounds, upper_bounds = super(BadCandidateHelper, self).boundaries()
        return upper_bounds, lower_bounds

class WorseCandidateHelper(CandidateHelper):
    def boundaries(self):
        lower_bounds, upper_bounds = super(WorseCandidateHelper, self).boundaries()
        return lower_bounds, np.delete(upper_bounds, 0)

class TestPSO(unittest.TestCase):
    def test_solver(self):

        try:
            fitness, position = pso.solver(CandidateHelper(), num_candidates=25)
        except Exception as ex:
            self.assertEqual(ex.args[0], "Number of candidates must be divisble by 10",
                             "not catch exception")

        try:
            fitness, position = pso.solver(WorseCandidateHelper())
        except Exception as ex:
            self.assertEqual(ex.args[0], "Length of upper and lower bounds must match",
                             "Did not catch exception")

        try:
            fitness, position = pso.solver(BadCandidateHelper())
        except Exception as ex:
            self.assertEqual(ex.args[0], "Upper bounds less than lower bounds",
                             "Did not catch exception")

        fitness, position = pso.solver(CandidateHelper())
        self.assertEqual(round(fitness), 0.0, "Found fitness is not 0.0")
        self.assertEqual(round(position[0]), 0.0, "Position[0] not at 0")
        self.assertEqual(round(position[1]), 0.0, "Position[1] not at 0")


        fitness, position = pso.solver(CandidateHelper(), pso_type='standard')
        self.assertEqual(round(fitness), 0.0, "Found fitness is not 0.0")
        self.assertEqual(round(position[0]), 0.0, "Position[0] not at 0")
        self.assertEqual(round(position[1]), 0.0, "Position[1] not at 0")

        fitness, position = pso.solver(CandidateHelper(), pso_type='fdr')
        self.assertEqual(round(fitness), 0.0, "Found fitness is not 0.0")
        self.assertEqual(round(position[0]), 0.0, "Position[0] not at 0")
        self.assertEqual(round(position[1]), 0.0, "Position[1] not at 0")

        fitness, position = pso.solver(CandidateHelper(), pso_type='fdrs')
        self.assertEqual(round(fitness), 0.0, "Found fitness is not 0.0")
        self.assertEqual(round(position[0]), 0.0, "Position[0] not at 0")
        self.assertEqual(round(position[1]), 0.0, "Position[1] not at 0")

        fitness, position = pso.solver(CandidateHelper(), topology='random')
        self.assertEqual(round(fitness), 0.0, "Found fitness is not 0.0")
        self.assertEqual(round(position[0]), 0.0, "Position[0] not at 0")
        self.assertEqual(round(position[1]), 0.0, "Position[1] not at 0")

        fitness, position = pso.solver(CandidateHelper(), topology='von neumann')
        self.assertEqual(round(fitness), 0.0, "Found fitness is not 0.0")
        self.assertEqual(round(position[0]), 0.0, "Position[0] not at 0")
        self.assertEqual(round(position[1]), 0.0, "Position[1] not at 0")


if __name__ == "__main__":
    unittest.main()
