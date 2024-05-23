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
