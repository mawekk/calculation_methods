import numpy


def convert_to_an_iterative_form(a, b):
    n = a.shape[0]
    h = numpy.zeros((n, n))
    g = numpy.zeros(n)

    for i in range(n):
        for j in range(n):
            h[i][j] = - a[i][j] / a[i][i]
            if i == j:
                h[i][j] = 0
        g[i] = b[i] / a[i][i]

    return h, g


def solve_with_simple_iteration_method(a, b, eps=10 ** -6):
    n = a.shape[0]
    x = numpy.zeros(n)
    error = eps + 1
    iters = 0

    h, g = convert_to_an_iterative_form(a, b)
    evaluation = numpy.linalg.norm(h) / (1 - numpy.linalg.norm(h))

    while error >= eps:
        iters += 1
        new_x = numpy.dot(h, x) + g
        error = numpy.linalg.norm(new_x - x) * evaluation
        x = new_x

    return new_x, iters


def solve_with_seidel_method(a, b, eps=10 ** -6):
    n = a.shape[0]
    x = numpy.zeros(n)
    error = eps + 1
    iters = 0

    h, g = convert_to_an_iterative_form(a, b)
    l = numpy.tril(h, -1)
    r = numpy.triu(h)

    evaluation = numpy.linalg.norm(h) / (1 - numpy.linalg.norm(h))
    helping_matrix = numpy.linalg.inv(numpy.eye(n) - l)

    while error >= eps:
        iters += 1
        new_x = numpy.dot(numpy.dot(helping_matrix, r), x) + numpy.dot(helping_matrix, g)
        error = numpy.linalg.norm(new_x - x) * evaluation
        x = new_x

    return new_x, iters
