from rotation_method import solve_with_rotation_method
import numpy
from task1.test import get_test_data
from outputs import print_table, empty_print, print_matrix
import task1.criteria as c

HEADERS = ["Матрица", "Спектральный критерий", "Объемный критерий", "Угловой критерий"]
MATRIX_NAMES = ["A", "Q", "R"]


def main():
    data = get_test_data()

    for (matrix, vector) in data:
        print_matrix(matrix, "Матрица A = ")
        print_matrix(vector, "Вектор b = ")

        q, r, x = solve_with_rotation_method(matrix, vector)
        error = numpy.linalg.norm(
            numpy.linalg.solve(matrix, vector) - x)

        print_matrix(q, "Матрица Q = ")
        print_matrix(r, "Матрица R = ")
        print_matrix(numpy.dot(q, r), "Произведение матриц QR =")
        print(f"Расхождение = {error}")

        rows = []
        for i, m in enumerate([matrix, q, r]):
            rows.append([MATRIX_NAMES[i], c.calculate_spectrum_criterion(m), c.calculate_volume_criterion(m),
                         c.calculate_angular_criterion(m)])

        print_table(HEADERS, rows, 15, '')
        empty_print()


if __name__ == "__main__":
    main()
