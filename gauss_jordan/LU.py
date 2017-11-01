import numpy as np

from gauss_jordan.gauss import scale_vector_by


def shuffle_rows(M: np.ndarray, N, position: int) -> (int, int) or None:
    """
    Looks for biggest absolute value in i'th column.
    Then, if biggest value is found in row x function swaps row i with row x.
    Example: position = 1
    column 1 swapped with column 2    row 1 swapped with row 3
     | 1 1 4 1 1|     | 1 1 4 1 1|
     | 0 3 3 2 2| ->  | 0 8 2 6 1|
     | 0 1 5 7 1|     | 0 1 5 7 1|
     | 0 8 2 6 1|     | 0 3 3 2 2|
    :param M: square matrix + result_column
    :param position: position on diagonal
    :return: Tuple with swapped columns or None if rows were swapped. Changes the given ndarray
    """
    column = M[:, position]
    # rows above were already reduced, so they shouldn't be taken into consideration
    reduced_column = column[position:]
    max_row = position + np.argmax(np.absolute(reduced_column))

    if max_row is not position:
        # swap rows
        M[[max_row, position]] = M[[position, max_row]]
        N[[max_row, position]] = N[[position, max_row]]


def half_reduce_matrix_by(M: np.ndarray, N: np.ndarray, x: int):
    """
    Zeros values below M[x,x]  in x'th column
    """

    size, _ = M.shape
    N[x, x] = 1.0
    for i in range(size - 1, x, -1):
        local_div = M[i][x] / M[x, x]
        N[i, x] = local_div
        M[i] = M[i] - local_div * M[x]


def LU(U: np.ndarray):
    """
    Performs LU reduction.
    """
    size, _ = U.shape
    L = np.zeros((size, size))
    for i in range(size):
        scale_vector_by(U[i], np.max(np.absolute(U[i][:-1])))
    for i in range(size):
        shuffle_rows(U, L, i)
        half_reduce_matrix_by(U, L, i)
    L[size - 1, size - 1] = 1
    return L, U


def dummyLU(U: np.ndarray):
    """
    Performs LU reduction.
    """
    size, _ = U.shape
    L = np.zeros((size, size))
    for i in range(size):
        half_reduce_matrix_by(U, L, i)
    L[size - 1, size - 1] = 1
    return L, U


t1 = np.array([[2, 2, -1, 1],
               [-1, 1, 2, 3],
               [3, -1, 4, -1],
               [1, 4, -2, 2]], dtype=np.float64)
t2 = np.array([7, 3, 31, 2], dtype=np.float64)

d1 = np.array([[5, 3, 2], [1, 2, 0], [3, 0, 4]], dtype=np.float64)
a = np.array([[2, -4, -1, 6],
              [-1, 1, 4, -1],
              [3, 8, 1, -4],
              [7, 3, -1, 2]
              ], dtype=np.float64)
from copy import deepcopy

v1 = deepcopy(a)
from scipy.linalg import lu

L1, U1 = LU(a)
P, L2, U2 = lu(v1)
