import unittest
import numpy as np
from scipy.linalg import lu
from copy import deepcopy
from gauss_jordan.LU import LU

t1 = np.array([[-2, -2, -1, 1],
               [-1, -1, -2, -3],
               [3, 1, 4, -1],
               [-1, 4, -2, 2]], dtype=np.float64)

a = np.array([[2, -4, -1, 6],
              [-1, 1, 4, -1],
              [3, 8, 1, -4],
              [-7, 3, -1, 2]
              ], dtype=np.float64)

v1 = deepcopy(a)


class MyTestCase(unittest.TestCase):
    def test_simple_cases(self):
        L1, U1 = LU(a)
        P, L2, U2 = lu(v1)
        self.assertAlmostEqual(np.linalg.det(a), np.linalg.det(U1))
        self.assertAlmostEqual(np.linalg.det(U2), np.linalg.det(U1))

        t2 = deepcopy(t1)
        t3 = deepcopy(t1)
        L1, U1 = LU(t1)
        P, L2, U2 = lu(t2)
        self.assertAlmostEqual(np.linalg.det(t3), np.linalg.det(U1))
        self.assertAlmostEqual(np.linalg.det(U2), np.linalg.det(U1))


if __name__ == '__main__':
    unittest.main()
