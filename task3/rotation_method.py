import numpy
import math


def get_rotation_matrix(matrix, i, j):
    c = matrix[i][i] / (math.sqrt(matrix[i][i] ** 2 + matrix[i][j] ** 2))
    s = - matrix[i][j] / (math.sqrt(matrix[i][i] ** 2 + matrix[i][j] ** 2))

    t = numpy.eye(matrix.shape[0])
    t[i][i] = c
    t[j][j] = c
    t[i][j] = - s
    t[j][i] = s

    return t


def solve_with_rotation_method(a, b):
    n = a.shape[0]
    q = numpy.eye(n)
    r = a

    for i in range(n - 1):
        for j in range(i + 1, n):
            t = get_rotation_matrix(a, i, j)
            r = numpy.dot(t, r)
            b = numpy.dot(t, b)
            q = numpy.dot(q, numpy.linalg.inv(t))

    x = numpy.linalg.solve(r, b)
    return q, r, x
