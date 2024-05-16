import numpy
from partial_problem_of_eigenvalues import power_method, scalar_product_method
import outputs

DATA = [numpy.array([[1.48213, 0.05316, 1.08254],
                     [0.05316, 1.13958, 0.01617],
                     [1.08254, 0.01617, 1.48271]]),
        numpy.array([[8.67313, 1.041039, -2.677712],
                     [1.041039, 6.586211, 0.623016],
                     [-2.677712, 0.623016, 5.225935]]),
        numpy.array([[1.00449, 0, 0],
                     [0, 0.73999, 0],
                     [0, 0, 1.086]])]

HEADERS = ["Точность", "Степенной метод", "Метод скалярных произведений"]


def main():
    print("Демонстрация работы методов:")
    a = DATA[0]
    outputs.print_matrix(a, "Матрица А:")

    x = numpy.array([1, 1, 1])
    y = numpy.array([-1, -10, 100])
    max_value = max(map(abs, numpy.linalg.eig(a)[0]))
    eps = 10 ** -6

    print(f"Точное максимальное по модулю собственное число: {max_value}\n")

    new_x, value, _ = power_method(a, x)
    print("Степенной метод:\n"
          f"Собственное число с точностью ε = {eps}: {value}")
    outputs.print_matrix(new_x, "Собственный вектор:")
    print(f"Ax = {numpy.dot(a, new_x)}\n"
          f"λx = {numpy.dot(value, new_x)}\n")

    new_x, value, _ = scalar_product_method(a, x, y)
    print("Метод скалярных произведений:\n"
          f"Собственное число с точностью ε = {eps}: {value}")
    outputs.print_matrix(new_x, "Собственный вектор:")
    print(f"Ax = {numpy.dot(a, new_x)}\n"
          f"λx = {numpy.dot(value, new_x)}")

    outputs.empty_print()

    for a in DATA:
        rows = []
        outputs.print_matrix(a, "Матрица А:")
        for i in range(5, 11):
            eps = 10 ** -i
            rows.append(
                [f"ε = {eps}", f"{power_method(a, x, eps)[2]} итераций",
                 f"{scalar_product_method(a, x, y, eps)[2]} итераций"])

        outputs.print_table(HEADERS, rows)


if __name__ == "__main__":
    main()
