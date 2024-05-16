import numpy
import math


def calculate_eigenvalue(x, new_x, new_y=None):
    if new_y is None:
        return math.sqrt(numpy.dot(new_x, new_x) / numpy.dot(x, x))
    return numpy.dot(new_x, new_y) / numpy.dot(x, new_y)


def calculate_posterior_error(x, new_x, value):
    return numpy.linalg.norm(new_x - value * x) / numpy.linalg.norm(x)


def power_method(a, x, eps=10 ** -6):
    iters = 0
    error = eps + 1
    new_x = 0
    value = 0

    while error >= eps:
        new_x = numpy.dot(a, x)
        value = calculate_eigenvalue(x, new_x)
        iters += 1

        if numpy.linalg.norm(new_x) > 100:
            new_x /= numpy.linalg.norm(new_x)

        error = calculate_posterior_error(x, new_x, value)
        x = new_x

    return new_x, value, iters


def scalar_product_method(a, x, y, eps=10 ** -6):
    iters = 0
    error = eps + 1
    new_x = 0
    value = 0

    while error >= eps:
        new_x = numpy.dot(a, x)
        new_y = numpy.dot(numpy.transpose(a), y)
        value = calculate_eigenvalue(x, new_x, new_y)
        iters += 1

        if numpy.linalg.norm(new_x) > 100:
            new_x /= numpy.linalg.norm(new_x)
        if numpy.linalg.norm(new_y) > 100:
            new_y /= numpy.linalg.norm(new_y)

        error = calculate_posterior_error(x, new_x, value)
        x = new_x
        y = new_y

    return new_x, value, iters
