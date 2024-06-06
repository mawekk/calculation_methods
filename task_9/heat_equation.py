import numpy


class HeatEquation:
    """
    вид уравнения:
    u_t(x, t) = k * u_xx(x, t) + f(x, t)
    k > 0, 0 < x < a, 0 < t <= T
    начальное условие:
    u(x, 0) = mu(x)
    0 <= x <= a
    граничные условия:
    u(0, t) = mu1(t), u(a, t) = mu2(t)
    0 <= t <= T
    """

    def __init__(self, k, f, a, t, mu, mu1, mu2):
        self.k = k
        self.f = f
        self.a = a
        self.t = t
        self.mu = mu
        self.mu1 = mu1
        self.mu2 = mu2

    def solve_with_explicit_scheme(self, n, m):
        xs = numpy.linspace(0, self.a, n + 1)
        ts = numpy.linspace(0, self.t, m + 1)

        h = self.a / n
        tau = self.t / m

        u = numpy.zeros((m + 1, n + 1))

        for i in range(n + 1):
            u[0, i] = self.mu(xs[i])

        for k in range(1, m + 1):
            u[k, 0] = self.mu1(ts[k])
            u[k, n] = self.mu2(ts[k])

        for k in range(1, m + 1):
            for i in range(1, n):
                u[k, i] = u[k - 1, i] + tau * (
                        self.k * (u[k - 1, i + 1] - 2 * u[k - 1, i] + u[k - 1, i - 1]) / h ** 2
                        + self.f(xs[i], ts[k - 1])
                )

        return xs, ts, u

    def solve_with_implicit_scheme(self, n, m):
        xs = numpy.linspace(0, self.a, n + 1)
        ts = numpy.linspace(0, self.t, m + 1)

        h = self.a / n
        tau = self.t / m

        u = numpy.zeros((m + 1, n + 1))

        for i in range(n + 1):
            u[0, i] = self.mu(xs[i])

        for k in range(1, m + 1):
            u[k, 0] = self.mu1(ts[k])
            u[k, n] = self.mu2(ts[k])

        matrix = numpy.zeros((n + 1, n + 1))

        matrix[0, 0] = 1
        for i in range(1, n):
            matrix[i, i - 1] = - tau * self.k / h ** 2
            matrix[i, i] = 1 + 2 * tau * self.k / h ** 2
            matrix[i, i + 1] = - tau * self.k / h ** 2
        matrix[n, n] = 1

        for k in range(1, m + 1):
            g = numpy.zeros(n + 1)

            g[0] = self.mu1(ts[k])
            for i in range(1, n):
                g[i] = u[k - 1, i] + self.f(xs[i], ts[k])
            g[n] = self.mu2(ts[k])

            u[k] = numpy.linalg.solve(matrix, g)

        return xs, ts, u
