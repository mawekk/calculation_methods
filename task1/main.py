from test_data import get_test_data
import criteria as c
from outputs import print_table, empty_print, print_matrix

HEADERS = ["Спектральный критерий", "Объемный критерий", "Угловой критерий", "Расхождение"]


def main():
    data = get_test_data()

    for (matrix, vector) in data:
        print_matrix(matrix, "Матрица A = ")
        print_matrix(vector, "\nВектор b = ")
        print_table(HEADERS, [[c.calculate_spectrum_criterion(matrix), c.calculate_volume_criterion(matrix),
                               c.calculate_angular_criterion(matrix), c.calculate_error(matrix, vector)]], 15, '')
        empty_print()


if __name__ == "__main__":
    main()
