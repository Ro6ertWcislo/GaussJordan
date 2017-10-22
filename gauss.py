import numpy as np
import random
from itertools import product


def solve(matrix: np.ndarray, vector: np.ndarray):
    return _solve(np.c_[matrix, vector])


def scale_vector_by(vector: np.ndarray, value: int):
    """
    Divides every element  by value.
    Doesn't return anything. Changes the given array.
    :param vector: 1-D ndarray
    :param value: division factor
    """
    for i in range(len(vector)):
        vector[i] /= value


def reduce_matrix_by(matrix: np.ndarray, x: int):
    """
    Resets all values different than matrix[x][x+1] in column with index x.
    This is accomplished by adding row with index x or subtracting it from other rows.
    :param matrix: square matrix + result column
    :param x: number of column to reduce, and row which will be use to perform reduction
    # waring! diagonal can't have any zeros!
    """
    size, _ = matrix.shape
    for i in range(size):
        scale_vector_by(matrix[x], matrix[x, x])
        if i is not x:
            local_div = matrix[i][x]
            matrix[i] = matrix[i] - local_div * matrix[x]


def shuffle_rows(M: np.ndarray, position: int):
    """

    :param M:
    :param position: position on diagonal
    :return:
    """
    column = M[:, position]
    # rows above are were already reduced, so they shouldn't be taken into consideration
    reduced_column = column[position:]
    max_row = position + np.argmax(np.absolute(reduced_column))
    if max_row is not position:
        M[[max_row, position]] = M[[position, max_row]]


def _solve(M: np.ndarray):
    """
    Reduces matrix to following state:
    |1 0 ... 0 y1|
    |0 1 ... 0 y2|
    |    ...     |
    |0 0 ... 1 yn|
    y1 to yn are the result of the equation system
    :param M: square matrix + result column
    :return: result column
    """
    size, _ = M.shape
    # scale the whole matrix:
    for i in range(size):
        scale_vector_by(M[i], np.max(np.absolute(M[i][:-1])))
    for i in range(size):
        shuffle_rows(M, i)
        reduce_matrix_by(M, i)
    return M[:, -1]


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
print(dt() - start)
start = dt()
print(np.linalg.solve(d1, d2))
print(dt() - start)
