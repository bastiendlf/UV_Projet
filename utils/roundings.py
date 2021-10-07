import numpy as np


def floor(x):
    """
    • The floor operator rounds towards the smallest nearest integer (i.e., towards negative infinity): floor(1.5) = 1
    and floor(−1.5) =−2
    :param x: number to around
    :return: rounded value
    """

    return np.floor(x)


def halfup(x):
    """
    • The halfup operator is similar to round(·) except in the case of a tie, where values are rounded towards the
    largest nearest integer: halfup(1.5) =2 and halfup(−1.5) =−1.
    :param x: number
    :return: rounded value
    """
    return np.floor(x + 0.5)


def trunc(x):
    """
    • The trunc operator rounds towards the nearest integer with the smaller magnitude (i.e., towards zero):
    trunc(1.5) = 1 and trunc(−1.5) =−1.
    :param x: number
    :return: rounded value
    """
    return np.trunc(x)


def round(x):
    return np.around(x)


if __name__ == '__main__':
    x = 1.5
    neg_x = x * -1

    print('****ROUND****')
    print(f"round({x}) = {round(x)} => should be 2")
    print(f"round({neg_x}) = {round(neg_x)} => should be -2")

    print('\n\n****HALFUP****')
    print(f"halfup({x}) = {halfup(x)} => should be 2")
    print(f"halfup({neg_x}) = {halfup(neg_x)} => should be -1")

    print('\n\n****TRUNC****')
    print(f"trunc({x}) = {trunc(x)} => should be 1")
    print(f"trunc({neg_x}) = {trunc(neg_x)} => should be -1")

    print('\n\n****FLOOR****')
    print(f"floor({x}) = {floor(x, )} => should be 1")
    print(f"floor({neg_x}) = {floor(neg_x)} => should be -2")
