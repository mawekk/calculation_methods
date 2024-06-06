import numpy
from conditional_optimization import solve_with_penalty, solve_with_barrier, solve_with_modified_lagrange


def main():
    f = lambda x: x
    phi = lambda x: 7 - x
    print("Функция f(x) = x\n"
          "Ограничениe:\n"
          "7 - x <= 0\n")
    x, iters, time = solve_with_penalty(f, [phi], [], numpy.array([5]), 100.0, 1.5)
    print("Метод штрафных функций\n"
          f"Ответ: {x}\n"
          f"Количество итераций: {iters}\n"
          f"Время: {time}\n")
    x, iters, time = solve_with_barrier(f, [phi], numpy.array([10]), 1.0, 0.9)
    print("Метод барьерных функций\n"
          f"Ответ: {x}\n"
          f"Количество итераций: {iters}\n"
          f"Время: {time}\n")
    x, iters, time = solve_with_modified_lagrange(f, [phi], numpy.array([10]), 0.01, numpy.array([0.1]), 0.1)
    print("Метод модифицированных функций Лагранжа\n"
          f"Ответ: {x}\n"
          f"Количество итераций: {iters}\n"
          f"Время: {time}\n")

    f = lambda x: (x[0] - 10) ** 2 + (x[1] - 10) ** 2
    phis = [lambda x: -x[0], lambda x: -x[1]]
    print("Функция f(x) = (x_0 - 10) ^ 2 + (x_1 - 10) ^ 2\n"
          "Ограничения:\n"
          "x_0 >= 0, x_1 >= 0\n")
    x, iters, time = solve_with_penalty(f, phis, [], numpy.array([-15, -15]), 100.0, 1.5)
    print("Метод штрафных функций\n"
          f"Ответ: {x}\n"
          f"Количество итераций: {iters}\n"
          f"Время: {time}\n")
    x, iters, time = solve_with_barrier(f, phis, numpy.array([30, 6]), 1.0, 0.9)
    print("Метод барьерных функций\n"
          f"Ответ: {x}\n"
          f"Количество итераций: {iters}\n"
          f"Время: {time}\n")
    x, iters, time = solve_with_modified_lagrange(f, phis, numpy.array([30, 6]), 0.1, numpy.array([0.1]), 0.1)
    print("Метод модифицированных функций Лагранжа\n"
          f"Ответ: {x}\n"
          f"Количество итераций: {iters}\n"
          f"Время: {time}\n")


if __name__ == "__main__":
    main()
