import math

import numdifftools
import numpy
import scipy


def diff(f, n=1):
    return numdifftools.Derivative(f, 1e-2, n=n)


def scalar_product(f, g, a, b):
    return scipy.integrate.quad(lambda x: f(x) * g(x), a, b)[0]


def coord_func(n, a=1, b=1):
    return lambda x: (1 - x ** 2) * scipy.special.eval_jacobi(n, a, b, x)


class BoundaryProblem:
    """
    вид уравнения:
    -p(x) * y'' + q(x) * y' + r(x) * y = f
    граничные условия:
    alpha1 * y(a) - alpha2 * y'(a) = alpha
    beta1 * y(b) + beta2 * y'(b) = beta
    |alpha1| + |alpha2| != 0, alpha1 * alpha2 >= 0
    |beta1| + |beta2| != 0, beta1 * beta2 >= 0
    """

    def __init__(self, p, q, r, f, a, b, alpha, beta, alpha1, alpha2, beta1, beta2):
        self.p = p
        self.q = q
        self.r = r
        self.f = f

        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1 = beta1
        self.beta2 = beta2

    def solve_with_grid_method(self):
        step_and_accuracy = []
        prev_grid_y = []
        n = 10

        while n < 10 ** 6:
            h = (self.b - self.a) / n
            grid_x = []
            grid_y = []
            s_t_array = []

            for i in range(n + 1):
                x_i = self.a + i * h
                grid_x.append(x_i)

                if i == 0:
                    a_i = 0
                    b_i = h * self.alpha1 + self.alpha2
                    c_i = self.alpha2
                    g_i = -h * self.alpha
                elif i == n:
                    a_i = self.beta2
                    b_i = h * self.beta1 + self.beta2
                    c_i = 0
                    g_i = -h * self.beta
                else:
                    a_i = -self.p(x_i) - self.q(x_i) * h / 2
                    c_i = -self.p(x_i) + self.q(x_i) * h / 2
                    b_i = a_i + c_i - h ** 2 * self.r(x_i)
                    g_i = h ** 2 * self.f(x_i)

                if i == 0:
                    s = c_i / b_i
                    t = -g_i / b_i
                else:
                    prev_s, prev_t = s_t_array[i - 1]
                    s = c_i / (b_i - a_i * prev_s)
                    t = (a_i * prev_t - g_i) / (b_i - a_i * prev_s)

                s_t_array.append((s, t))

            y = 0
            for i in range(n, -1, -1):
                s, t = s_t_array[i]
                y = s * y + t
                grid_y.append(y)
            grid_y.reverse()

            if len(prev_grid_y) > 0:
                accuracy = max([abs(grid_y[i] - prev_grid_y[i // 2]) for i in range(0, n + 1, 2)])
                step_and_accuracy.append((n, accuracy))
            prev_grid_y = grid_y.copy()
            n *= 2

        return grid_x, grid_y, step_and_accuracy

    def solve_with_galerkin_method(self, n):
        omegas = [coord_func(i) for i in range(n)]
        matrix = numpy.zeros((n, n))
        vector = numpy.zeros(n)

        for i in range(n):
            for j in range(n):
                lw_j = lambda x: (- self.p(x) * diff(omegas[j], 2)(x) +
                                  self.q(x) * diff(omegas[j])(x) +
                                  self.r(x) * omegas[j](x))

                matrix[i, j] = scalar_product(lw_j, omegas[i], self.a, self.b)
            vector[i] = scalar_product(self.f, omegas[i], self.a, self.b)

        coeffs = numpy.linalg.solve(matrix, vector)

        return lambda x: sum(coeffs[i] * omegas[i](x) for i in range(n))

    def solve_with_collocation_method(self, n):
        omegas = [coord_func(i) for i in range(n)]
        points = [math.cos((2 * i - 1) / 2 * n) * math.pi for i in range(1, n + 1)]
        matrix = numpy.zeros((n, n))
        vector = numpy.zeros(n)

        for i in range(n):
            for j in range(n):
                lw_j = lambda x: (- self.p(x) * diff(omegas[j], 2)(x) +
                                  self.q(x) * diff(omegas[j])(x) +
                                  self.r(x) * omegas[j](x))

                matrix[i, j] = lw_j(points[i])
            vector[i] = self.f(points[i])

        coeffs = numpy.linalg.solve(matrix, vector)

        return lambda x: sum(coeffs[i] * omegas[i](x) for i in range(n))