import numpy
import test
import calculations as c
from outputs import print_table, empty_print

HEADERS = ["Спектральный критерий", "Объемный критерий", "Угловой критерий", "Расхождение"]


def print_matrix_and_vector(matrix, vector):
    print("Матрица A = ")
    for i in range(numpy.shape(matrix)[0]):
        print(matrix[i])

    print("\nВектор b = ")
    for i in range(numpy.shape(vector)[0]):
        print(vector[i])


def main():
    data = test.get_test_data()
    for (matrix, vector) in data:
        print_matrix_and_vector(matrix, vector)
        print_table(HEADERS, [[c.calculate_spectrum_criterion(matrix), c.calculate_volume_criterion(matrix),
                              c.calculate_angular_criterion(matrix), c.calculate_error(matrix, vector)]], 15, '')
        empty_print()


if __name__ == "__main__":
    main()
