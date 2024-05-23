import math
from grid_method import BoundaryProblem
import matplotlib.pyplot as plt


def main():

    p = lambda x: -1 / (x - 3)
    q = lambda x: 1 + x / 2
    r = lambda x: math.exp(x / 2)
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

    plt.subplot(1, 2, 1)
    plt.plot(steps, accuracies, color='b', marker='o', markersize=5)
    plt.xscale('log', base=10)
    plt.yscale('log', base=10)
    plt.xlabel("Шаг сетки")
    plt.ylabel("Точность")
    plt.title("Зависимость точности от шага сетки")

    plt.subplot(1, 2, 2)
    plt.plot(grid_x, grid_y, color='g')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Приближение функции y(x)\nШаг сетки = {len(grid_y) - 1}")

    if input("> Показать графики? y/n: ") == 'y':
        plt.show()
    else:
        exit()


if __name__ == "__main__":
    main()
