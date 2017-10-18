import numpy as np
import random
from itertools import product


def solve(matrix: np.ndarray, vector: np.ndarray):
    return _solve(np.c_[matrix, vector])


def scale_vector_by(vector: np.ndarray, value: int):
    """
    Divides every element  by vector[x].
    Doesn't return anything. Changes the given array.
    :param vector: 1-D ndarray
    :param index: position of division factor in vector
    """
    for i in range(len(vector)):
        vector[i] /= value


def reduce_matrix_by(matrix: np.ndarray, x: int):
    """ !!!!!!!!!!! +1 row na dodatkową wynikową
    Resets all values different than matrix[x][x] in column with index x.
    This is accomplished by adding row with index x or subtracting it from other rows.
    :param matrix: square matrix
    :param x: number of column to reduce, and row which will be use to perform reduction
    # waring! diagonal can't have any zeros!
    """
    size, _ = matrix.shape
    for i in range(size):
        scale_vector_by(matrix[x], matrix[x, x])
        if i != x:
            local_div = matrix[i][x]
            matrix[i] = matrix[i] - local_div * matrix[x]


def _solve(matrix):
    size, _ = matrix.shape
    # scale the whole matrix:
    for i in range(size):
        scale_vector_by(matrix[i], np.max(np.absolute(matrix[i][:-1])))
    for i in range(size):
        reduce_matrix_by(matrix, i)
    return matrix[:, -1]


t1 = np.array([[2, 2, -1, 1],
               [-1, 1, 2, 3],
               [3, -1, 4, -1],
               [1, 4, -2, 2]], dtype=np.float64)
t2 = np.array([7, 3, 31, 2], dtype=np.float64)

print(solve(t1, t2))
print(np.linalg.solve(t1, t2))
d1 = np.empty([100, 100], dtype=np.float64)
d2 = np.empty(100, dtype=np.float64)

for i, j in product(range(100), range(100)):
    d1[i, j] = random.randrange(0, 100)

for i in range(100):
    d2[i] = random.randrange(0, 100)
from timeit import default_timer as dt
start = dt()
print(solve(d1, d2))
print(dt()-start)
start = dt()
print(np.linalg.solve(d1, d2))
print(dt()-start)
