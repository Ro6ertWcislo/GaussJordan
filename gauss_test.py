import unittest
import random

import numpy as np
from gauss import solve

t1 = np.array([[2, 2, -1, 1],
               [-1, 1, 2, 3],
               [3, -1, 4, -1],
               [1, 4, -2, 2]], dtype=np.float64)
t2 = np.array([7, 3, 31, 2], dtype=np.float64)
d1 = np.empty([100, 100], dtype=np.float64)
d2 = np.empty(100, dtype=np.float64)

for i, j in np.ndindex((100, 100)):
    d1[i, j] = random.randrange(0, 100)

for i in range(100):
    d2[i] = random.randrange(0, 100)


class GaussTest(unittest.TestCase):
    def test_little_one(self):
        for i, j in zip(solve(t1, t2), np.linalg.solve(t1, t2)):
            self.assertAlmostEqual(i, j)

    def test_randomized(self):
        for i, j in zip(solve(d1, d2), np.linalg.solve(d1, d2)):
            self.assertAlmostEqual(i, j)


if __name__ == '__main__':
    unittest.main()
