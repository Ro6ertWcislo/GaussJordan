import numpy as np


def solve(matrix: np.ndarray, output_vector: np.ndarray):
    """ Adds column output_vector to matrix. Calls _solve on concatenated Matrix"""
    return _solve(np.c_[matrix, output_vector])


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
    Example:
    | 1. 0.5    -0.5    1.    3.5   |       | 1  0 -0.71..  0.71..  2.57.. |
    | 0. 1.1(6)  0.5    0.(6) 2.1(6)|  -->  | 0  1  0.42..  0.57..  1.85.. |
    | 0.-0.625   1.375 -1.    5.125 |  -->  | 0  0  1.64.. -0.64..  6.28.. |
    | 0. 0.375  -0.375  0.75 -0.375 |       | 0  0 -0.53..  0.53.. -1.07.. |

    :param matrix: square matrix + result column
    :param x: number of column to reduce, and row which will be use to perform reduction
    # waring! diagonal can't have any zeros!
    """
    size, _ = matrix.shape
    scale_vector_by(matrix[x], matrix[x, x])
    for i in range(size):
        if i is not x:
            local_div = matrix[i][x]
            matrix[i] = matrix[i] - local_div * matrix[x]


def shuffle(M: np.ndarray, position: int) -> (int, int) or None:
    """
    Looks for biggest absolute value in i'th row and i'th column.
    Then, if biggest value is found in row x function swaps row i with row x.
    If biggest value is found in column y, function swaps column i with column y,
    then appends column_swap with tuple (i,y).
    (We have to swap columns backwards before returning the result of the equation)
    Example: position = 1
    column 1 swapped with column 2    row 1 swapped with row 3
    | 1 4 1 1 1|    | 1 1 4 1 1|       | 1 1 4 1 1|     | 1 1 4 1 1|
    | 0 3 9 2 2| -> | 0 9 3 2 2|  or   | 0 3 3 2 2| ->  | 0 8 2 6 1|
    | 0 5 1 7 1|    | 0 1 5 7 1|       | 0 1 5 7 1|     | 0 1 5 7 1|
    | 0 2 5 6 1|    | 0 5 2 6 1|       | 0 8 2 6 1|     | 0 3 3 2 2|
    :param M: square matrix + result_column
    :param position: position on diagonal
    :return: Tuple with swapped columns or None if rows were swapped. Changes the given ndarray
    """
    column = M[:, position]
    row = M[position]
    # rows above were already reduced, so they shouldn't be taken into consideration
    reduced_column = column[position:]
    # columns on the left were already reduced, so they shouldn't be taken into consideration
    reduced_row = row[position:]
    # we do not look at the alst element because it belongs to "result column"
    max_column = position + np.argmax(np.absolute(reduced_row[:-1]))
    max_row = position + np.argmax(np.absolute(reduced_column))

    if M[position, max_column] > M[max_row, position]:
        if max_column is not position:
            M[:, [max_column, position]] = M[:, [position, max_column]]
            return (max_column, position)
    else:
        if max_row is not position:
            # swap rows
            M[[max_row, position]] = M[[position, max_row]]


def to_diagonal(M: np.ndarray) -> np.ndarray:
    """
    :param M: Diagonal matrix with shuffled columns and added result column e.g.
    |0 0 1 0 9|
    |1 0 0 0 3|
    |0 0 0 1 8|
    |0 1 0 0 5|
    :return: "Fixed" diagonal Matrix e.g.
    |1 0 0 0 3|
    |0 1 0 0 5|
    |0 0 1 0 9|
    |0 0 0 1 8|
    """
    size, _ = M.shape
    for col, row in np.ndindex((size, size)):
        if np.isclose(M[row, col], 1.0):
            M[[col, row]] = M[[row, col]]
    return M


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
    swapped_columns = []
    # scale the whole matrix:
    for i in range(size):
        scale_vector_by(M[i], np.max(np.absolute(M[i][:-1])))
    for i in range(size):
        # swap to have better numeric stability
        swapped_column = shuffle(M, i)
        if swapped_column:
            swapped_columns.append(swapped_column)
        reduce_matrix_by(M, i)
    # swap all column back where they were
    for col1, col2 in swapped_columns[::-1]:
        M[:, [col1, col2]] = M[:, [col2, col1]]
    return to_diagonal(M)[:, -1]
