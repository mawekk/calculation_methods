import numpy


def get_max_nondiag_element(a):
    abs_a = numpy.abs(a)
    numpy.fill_diagonal(abs_a, 0)

    return numpy.unravel_index(numpy.argmax(abs_a), a.shape)


def get_optimal_element(a):
    i = numpy.argmax(numpy.sum(a ** 2, 1) - numpy.diag(a ** 2))
    row = numpy.abs(a[i])
    row[i] = 0
    j = numpy.argmax(row)
    return i, j


def get_rotation_matrix(a, i, j):
    if a[i][i] == a[j][j]:
        phi = numpy.pi / 4
    else:
        phi = numpy.arctan2(-2 * a[i][j], a[i][i] - a[j][j])
    c = numpy.cos(phi)
    s = numpy.sin(phi)

    t = numpy.eye(a.shape[0])
    t[i][i] = c
    t[j][j] = c
    t[i][j] = - s
    t[j][i] = s

    return t


def get_nondiag_elements_sum(a):
    elements_sum = 0
    for i in range(a.shape[0]):
        for j in range(a.shape[0]):
            if i != j:
                elements_sum += a[i][j] ** 2

    return elements_sum


def jacobi_method(a, strategy, eps=10 ** -6):
    iters = 0

    while get_nondiag_elements_sum(a) >= eps:
        i, j = strategy(a)
        t = get_rotation_matrix(a, i, j)
        iters += 1

        a = numpy.dot(numpy.dot(numpy.transpose(t), a), t)

    eigenvalues = numpy.diag(a)
    return eigenvalues, iters, check_eigenvalues(a, eigenvalues)


def get_gershgorin_circles(a):
    circles = []
    for i in range(a.shape[0]):
        radius = numpy.sum(numpy.abs(a[i])) - numpy.abs(a[i, i])
        circles.append([a[i, i] - radius, a[i, i] + radius])
    circles.sort()

    return circles


def check_eigenvalues(a, values):
    count = 0
    circles = get_gershgorin_circles(a)
    for value in values:
        for start, end in circles:
            if start <= value <= end:
                count += 1
                break

    if count == values.shape[0]:
        return "✔"
    else:
        return "✖"

