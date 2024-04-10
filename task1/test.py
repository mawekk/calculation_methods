import random

import numpy


def generate_vector(size):
    return numpy.random.rand(size)


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
