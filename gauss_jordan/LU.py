import numpy as np
from copy import deepcopy

from gauss_jordan.gauss import scale_vector_by


def shuffle_rows(M: np.ndarray, N, position: int, deviders: np.ndarray) -> (int, int) or None:
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
        deviders[max_row], deviders[position] = deviders[position], deviders[max_row]


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


def max_abs_val(A: np.ndarray):
    index = np.argmax(np.absolute(A))
    return A[index]


def LU(A: np.ndarray):
    """
    Performs LU reduction.
    deviders array is introduced,because scaling cause the determinant of the matrix change.
    The deviders are saved there and the matrix is re-scaled at the end of the computation
    """
    size, _ = A.shape
    L = np.zeros((size, size))
    deviders = np.zeros(size)
    U = deepcopy(A)
    for i in range(size):
        devider = max_abs_val(U[i][:-1])
        scale_vector_by(U[i], devider)
        deviders[i] = devider
    for i in range(size):
        shuffle_rows(U, L, i, deviders)
        half_reduce_matrix_by(U, L, i)
    L[size - 1, size - 1] = 1
    return L, deviders * U
