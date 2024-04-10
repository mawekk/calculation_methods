from tabulate import tabulate


def empty_print():
    print("================================================================================================")


def print_table(headers, rows, accuracy=15, t='e'):
    print(tabulate(rows, headers=headers, tablefmt="rounded_grid", floatfmt=get_format(accuracy, t)))


def get_format(accuracy, t): return f'.{accuracy}{t}'
