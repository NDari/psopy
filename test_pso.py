import numpy as np
import pso
import unittest


class CandidateHelper(pso.Candidate):
    def evalFitness(self, pos):
        return np.sum(np.power(pos, 2))

    def boundaries(self):
        return np.array([-5.0, -5.0]), np.array([5.0, 5.0])

class TestPSO(unittest.TestCase):
    def test_solver(self):
        c = []
        for i in range(10):
            c.append(CandidateHelper())

        fitness, position = pso.solver(c)
        self.assertEqual(round(fitness), 0.0, "Found fitness is not 0.0")
        self.assertEqual(round(position[0]), 0.0, "Position[0] not at 0")
        self.assertEqual(round(position[1]), 0.0, "Position[1] not at 0")

if __name__ == "__main__":
    unittest.main()
