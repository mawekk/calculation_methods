import math
import random
import numpy


def generate_vector(size):
    return numpy.random.rand(size)


def generate_discharged_symmetrical_matrix(size):
    non_zero_count = math.floor(size ** 2 * 0.1) - size
    matrix = numpy.diag(numpy.random.randint(100, 1000, size=size))

    while non_zero_count > 0:
        i = random.randint(1, size - 1)
        j = random.randint(0, i - 1)
        matrix[i][j] = random.randint(1, 5)
        matrix[j][i] = matrix[i][j]
        non_zero_count -= 2

    return matrix


def book_matrix():
    matrix = numpy.array([[7.35272, 0.88255, -2.270052],
                          [0.88255, 5.58351, 0.528167],
                          [-2.27005, 0.528167, 4.430329]])
    vector = numpy.array([1, 0, 0])

    return matrix, vector


def hilbert_matrix():
    size = random.randint(3, 5)
    matrix = numpy.array([[1 / (i + j + 1) for i in range(size)] for j in range(size)])
    vector = generate_vector(size)

    return matrix, vector


def tridiagonal_matrix():
    size = random.randint(3, 10)

    matrix = numpy.zeros((size, size))
    vector = generate_vector(size)

    for i in range(size):
        matrix[i, i] = numpy.random.randint(4, 10)
        if i > 0:
            matrix[i, i - 1] = numpy.random.randint(1, matrix[i, i] // 2)
        if i < size - 1:
            matrix[i, i + 1] = numpy.random.randint(1, matrix[i, i] // 2)

    return matrix, vector


def get_test_data():
    return [book_matrix(), hilbert_matrix(), tridiagonal_matrix()]
