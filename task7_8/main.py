import math
import numpy
from boundary_problem import BoundaryProblem
import matplotlib.pyplot as plt

COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def main():
    p = lambda x: -1 / (x - 3)
    q = lambda x: 1 + x / 2
    r = lambda x: - math.exp(x / 2)
    f = lambda x: 2 - x

    a = -1
    b = 1
    alpha = 0
    beta = 0
    alpha1 = 1
    alpha2 = 0
    beta1 = 1
    beta2 = 0

    print("Уравнение:")
    print(f" -1/(x - 3) * y'' + (1 + x / 2) * y' + e ^ (x / 2) * y = 2 - x")
    print("\nГраничные условия:")
    print(f"{alpha1} * y({a}) - {alpha2} * y'({a}) = {alpha}\n"
          f"{beta1} * y({b}) + {beta2} * y'({b}) = {beta}")

    problem = BoundaryProblem(p, q, r, f, a, b, alpha, beta, alpha1, alpha2, beta1, beta2)

    grid_x, grid_y, step_and_accuracy = problem.solve_with_grid_method()

    steps, accuracies = [], []
    for step, accuracy in step_and_accuracy:
        steps.append(step)
        accuracies.append(accuracy)

    plt.subplot(2, 2, 1)
    plt.plot(steps, accuracies, color=COLORS[0], marker='o', markersize=5)
    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    plt.xlabel("Шаг сетки")
    plt.ylabel("Точность")
    plt.title("Сеточный метод.\nЗависимость точности от шага сетки")

    plt.subplot(2, 2, 2)
    plt.plot(grid_x, grid_y, color=COLORS[1], label=f"n = {len(grid_y) - 1}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Сеточный метод")

    for i in range(3):
        n = 2 + i * 2
        galerkin_y = problem.solve_with_galerkin_method(n)
        galerkin_xs = numpy.linspace(a, b, 100)
        galerkin_ys = []
        for x in galerkin_xs:
            galerkin_ys.append(galerkin_y(x))
        plt.subplot(2, 2, 3)
        plt.plot(galerkin_xs, galerkin_ys, color=COLORS[i], label=f"n = {n}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Метод Галеркина")

    for i in range(5):
        n = 10 + i * 5
        collocation_y = problem.solve_with_collocation_method(n)
        collocation_xs = numpy.linspace(a, b, 100)
        collocation_ys = []
        for x in collocation_xs:
            collocation_ys.append(collocation_y(x))

        plt.subplot(2, 2, 4)
        plt.plot(collocation_xs, collocation_ys, color=COLORS[2 + i], label=f"n = {n}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"Метод коллокаций")

    plt.subplots_adjust(hspace=0.5)
    plt.show()


if __name__ == "__main__":
    main()
