import numpy as np

from gauss import scale_vector_by


def shuffle_rows(M: np.ndarray, position: int) -> (int, int) or None:
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


def half_reduce_matrix_by(M: np.ndarray, x: int):
    """!!!!!!!!!!!!!!!!!!!
    Resets all values different than matrix[x][x+1] in column with index x.
    This is accomplished by adding row with index x or subtracting it from other rows.
    Example:
    | 1. 0.5    -0.5    1.    3.5   |       | 1  0 -0.71..  0.71..  2.57.. |
    | 0. 1.1(6)  0.5    0.(6) 2.1(6)|  -->  | 0  1  0.42..  0.57..  1.85.. |
    | 0.-0.625   1.375 -1.    5.125 |  -->  | 0  0  1.64.. -0.64..  6.28.. |
    | 0. 0.375  -0.375  0.75 -0.375 |       | 0  0 -0.53..  0.53.. -1.07.. |

    :param M: square matrix + result column
    :param x: number of column to reduce, and row which will be use to perform reduction
    # waring! diagonal can't have any zeros!
    """
    size, _ = M.shape
    scale_vector_by(M[x], M[x, x])
    for i in range(x + 1, size):
        local_div = M[i][x]
        M[i] = M[i] - local_div * M[x]


def half_by(M: np.ndarray, N: np.ndarray, x: int):
    """!!!!!!!!!!!!!!!!!!!
    Resets all values different than matrix[x][x+1] in column with index x.
    This is accomplished by adding row with index x or subtracting it from other rows.
    Example:
    | 1. 0.5    -0.5    1.    3.5   |       | 1  0 -0.71..  0.71..  2.57.. |
    | 0. 1.1(6)  0.5    0.(6) 2.1(6)|  -->  | 0  1  0.42..  0.57..  1.85.. |
    | 0.-0.625   1.375 -1.    5.125 |  -->  | 0  0  1.64.. -0.64..  6.28.. |
    | 0. 0.375  -0.375  0.75 -0.375 |       | 0  0 -0.53..  0.53.. -1.07.. |

    :param M: square matrix + result column
    :param x: number of column to reduce, and row which will be use to perform reduction
    # waring! diagonal can't have any zeros!
    """

    size, _ = M.shape
    N[x, x] = 1.0
    #scale_vector_by(M[x], M[x, x])
    for i in range(size - 1, x, -1):
        local_div = M[i][x]/M[x,x]

        N[i, x] = local_div
        M[i] = M[i] - local_div * M[x]


def LU(matrix: np.ndarray, output_vector: np.ndarray):
    """ Adds column output_vector to matrix. Calls _solve on concatenated Matrix"""
    return LLU(np.c_[matrix, output_vector])


def LLU(M: np.ndarray):
    """
    Reduces matrix to following state:
    |1   0 ... 0 y1|
    |x21 1 ... 0 y2|
    |      ...     |
    |xn1 xn2...1 yn|
    :param M: square matrix + result column
    """
    size, _ = M.shape
    N = np.zeros((size, size))
    # for i in range(size):
    #     scale_vector_by(M[i], np.max(np.absolute(M[i][:-1])))
    for i in range(size):
        #shuffle_rows(M, i)
        half_by(M, N, i)
    N[size-1,size-1]=1
    return M, N  # [:, -1]


t1 = np.array([[2, 2, -1, 1],
               [-1, 1, 2, 3],
               [3, -1, 4, -1],
               [1, 4, -2, 2]], dtype=np.float64)
t2 = np.array([7, 3, 31, 2], dtype=np.float64)

d1 = np.array([[5,3,2],[1,2,0],[3,0,4]],dtype=np.float64)

x,y = LLU(d1)
print(x)
print(y)