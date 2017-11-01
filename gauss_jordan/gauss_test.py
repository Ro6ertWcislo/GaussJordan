import unittest
import random
from timeit import default_timer as dt

import numpy as np
from gauss_jordan.gauss import solve

t1 = np.array([[2, 2, -1, 1],
               [-1, 1, 2, 3],
               [3, -1, 4, -1],
               [1, 4, -2, 2]], dtype=np.float64)
t2 = np.array([7, 3, 31, 2], dtype=np.float64)
size = 256
d1 = np.empty([size, size])
d2 = np.empty(size)

for i, j in np.ndindex((size, size)):
    d1[i, j] = random.randrange(1, 100)

for i in range(size):
    d2[i] = random.randrange(1, 100)


class GaussTest(unittest.TestCase):
    def test_little_one(self):
        for i, j in zip(solve(t1, t2), np.linalg.solve(t1, t2)):
            self.assertAlmostEqual(i, j)

    def test_randomized(self):
        for i in range(10):
            print(i)
            size = 257
            d1 = np.empty([size, size])
            d2 = np.empty(size)

            for i, j in np.ndindex((size, size)):
                d1[i, j] = random.randrange(1, 100)

            for i in range(size):
                d2[i] = random.randrange(1, 100)

            start = dt()
            np_solution = np.linalg.solve(d1, d2)
            print("Numpy solution lasted {}.".format(dt() - start))
            print(str(np_solution[:100]))

            start = dt()
            my_solution = solve(d1, d2)
            print("My solution lasted {}.".format(dt() - start))
            print(str(my_solution))

            for i, j in zip(my_solution, np_solution):
                if not np.isclose(i, j):
                    print(i, j)
                self.assertAlmostEqual(i, j)


if __name__ == '__main__':
    unittest.main()
