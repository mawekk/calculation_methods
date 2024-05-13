from numpy import linalg, shape
from math import sqrt

VALUE = 10 ** (-2) - 10 ** (-8)


def calculate_spectrum_criterion(matrix):
    return linalg.norm(matrix) * linalg.norm(linalg.inv(matrix))


def calculate_volume_criterion(matrix):
    volume_criterion = 1

    size = shape(matrix)[0]
    for i in range(size):
        volume_criterion *= sqrt(sum(matrix[i, j] ** 2 for j in range(size)))

    return volume_criterion / linalg.det(matrix)


def calculate_angular_criterion(matrix):
    return max(linalg.norm(matrix[i, :]) * linalg.norm(linalg.inv(matrix)[:, i]) for i in range(shape(matrix)[0]))


def calculate_error(matrix, vector):
    return linalg.norm(
        linalg.solve(matrix, vector) - linalg.solve(matrix * VALUE, vector * VALUE))