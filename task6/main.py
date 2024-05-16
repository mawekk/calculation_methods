import numpy
import complete_problem_of_eigenvalues as cp
import test_data
import outputs

HEADERS = ["Точность", "Макс по модулю недиагональный", "Оптимальный", "Попадание в область"]


def main():
    print("Демонстрация работы метода Якоби:")
    a = test_data.hilbert_matrix(3)[0]
    outputs.print_matrix(a, "Матрица А:")

    eigenvalues, _ = numpy.linalg.eig(a)
    print(f"\nТочные собственные числа: {sorted(eigenvalues)}\n")

    values, iters, check = cp.jacobi_method(a, cp.get_max_nondiag_element)
    print(f"Стратегия: максимальный по модулю недиагональный элемент\n"
          f"Собственные числа с точностью ε = {10 ** -6}: {sorted(values)}\n"
          f"Количество итераций: {iters}\n"
          f"Попадание в круги Гершгорина: {check}")
    print()

    values, iters, check = cp.jacobi_method(a, cp.get_optimal_element)
    print(f"Стратегия: оптимальный элемент\n"
          f"Собственные числа с точностью ε = {10 ** -6}: {sorted(values)}\n"
          f"Количество итераций: {iters}\n"
          f"Попадание в круги Гершгорина: {check}")

    for n in [5, 8, 10]:
        outputs.empty_print()
        print(f"Матрица Гильберта порядка {n}")
        a = test_data.hilbert_matrix(n)[0]

        rows = []
        for i in range(1, 7):
            eps = 10 ** -i
            _, max_iters, max_check = cp.jacobi_method(a, cp.get_max_nondiag_element, eps)
            _, opt_iters, opt_check = cp.jacobi_method(a, cp.get_optimal_element, eps)

            rows.append(
                [f"ε = {eps}", f"{max_iters} итераций",
                 f"{opt_iters} итераций", f"{max_check}/{opt_check}"])

        outputs.print_table(HEADERS, rows)


if __name__ == "__main__":
    main()
