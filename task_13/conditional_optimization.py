import time
import numpy

from scipy.optimize import minimize


def get_penalty_function(phis, psis, p):
    h_phi = lambda x: sum(max(0, phi(x)) ** p for phi in phis)
    h_psi = lambda x: sum(abs(psi(x)) for psi in psis)
    return lambda x: h_phi(x) + h_psi(x)


def solve_with_penalty(f, phis, psis, x_0, alpha, coeff, eps=10 ** -9):
    start = time.time()
    penalty_function = get_penalty_function(phis, psis, 2)
    iters = 0

    while alpha * penalty_function(x_0) >= eps:
        theta = lambda x: f(x) + alpha * penalty_function(x)
        x_0 = minimize(theta, x_0).x

        iters += 1
        alpha *= coeff

    return x_0, iters, time.time() - start


def get_barrier_value(value):
    if value == 0:
        return float('inf')
    return - 1 / value


def solve_with_barrier(f, phis, x_0, mu, coeff, eps=10 ** -9):
    start = time.time()
    barrier_function = lambda x: sum(get_barrier_value(phi(x)) for phi in phis)
    iters = 0

    while mu * barrier_function(x_0) >= eps:
        theta = lambda x: f(x) + mu * barrier_function(x)
        x_min = minimize(theta, x_0).x

        for phi in phis:
            if phi(x_min) > 0:
                return x_0, iters, time.time() - start

        x_0 = x_min

        iters += 1
        mu *= coeff

    return x_0, iters, time.time() - start


def get_square(x):
    return sum(numpy.power(x, 2))


def get_projection(values):
    return numpy.array([max(value, 0) for value in values])


def get_modified_lagrange(f, phis, a, x, l):
    values = numpy.array([phi(x) for phi in phis])
    lambda_plus_values = get_projection(l + a * values)
    return f(x) + 1 / (2 * a) * get_square(lambda_plus_values) - 1 / (2 * a) * get_square(l)


def solve_with_modified_lagrange(f, phis, x_0, alpha, l_0, a, eps=10 ** -9):
    start = time.time()
    theta = lambda x, x_k, l_k: 1 / 2 * get_square(x - x_k) + alpha * get_modified_lagrange(f, phis, a, x, l_k)
    iters = 0
    x_prev = x_0 + 1

    while numpy.sum(numpy.abs(x_0 - x_prev)) >= eps:
        theta_prev = lambda x: theta(x, x_0, l_0)
        x_prev = x_0
        x_0 = minimize(theta_prev, x_0).x

        values = numpy.array([phi(x_0) for phi in phis])
        l_0 = get_projection(l_0 + a * values)
        iters += 1

    return x_0, iters, time.time() - start


