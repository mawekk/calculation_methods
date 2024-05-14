import random
import numpy
from test_data import generate_discharged_symmetrical_matrix, generate_vector, book_matrix
from iterative_methods import solve_with_simple_iteration_method, solve_with_seidel_method
import outputs

HEADERS = ["Точность", "Метод простых итераций", "Метод Зейделя"]


def main():
    print("Демонстрация работы методов:")
    a, b = book_matrix()
    outputs.print_matrix(a, "Матрица А =")
    outputs.print_matrix(b, "Вектор b =")

    print()
    print(f"Точное значение x = {numpy.linalg.solve(a, b)}")
    print()
    eps = 10 ** -10
    print(f"ε = {eps}")
    x, iters = solve_with_simple_iteration_method(a, b, eps)
    print(f"Метод простых итераций. Значение x = {x}, количество итераций = {iters}")
    x, iters = solve_with_seidel_method(a, b, eps)
    print(f"Метод Зейделя. Значение x = {x}, количество итераций = {iters}")

    size = random.randint(200, 300)
    a = generate_discharged_symmetrical_matrix(size)
    b = generate_vector(size)

    outputs.empty_print()

    print(f"Порядок матрицы А = {size}")
    rows = []
    for i in range(5, 11):
        eps = 10 ** -i
        rows.append(
            [f"ε = {eps}", f"{solve_with_simple_iteration_method(a, b, eps)[1]} итераций",
             f"{solve_with_seidel_method(a, b, eps)[1]} итераций"])

    outputs.print_table(HEADERS, rows)


if __name__ == "__main__":
    main()
