from tabulate import tabulate
import numpy


def empty_print():
    print("================================================================================================")


def print_table(headers, rows, accuracy=15, t='e'):
    print(tabulate(rows, headers=headers, tablefmt="rounded_grid", floatfmt=get_format(accuracy, t)))


def get_format(accuracy, t): return f'.{accuracy}{t}'


def print_matrix(matrix, text=""):
    if text != "":
        print(text)
    with numpy.printoptions(precision=3, suppress=True):
        result = str(matrix)
        result = result.replace(" [", "").replace("]", "").lstrip("[[")
        print(result)
