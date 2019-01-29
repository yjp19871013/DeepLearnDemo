import numpy as np


def and_load_data_set():
    return np.array([[0, 1], [0, 0], [1, 0], [1, 1]]), np.array([[0, 0, 0, 1]]).T


def or_load_data_set():
    return np.array([[0, 1], [0, 0], [1, 0], [1, 1]]), np.array([[1, 0, 1, 1]]).T


def nand_load_data_set():
    return np.array([[0, 1], [0, 0], [1, 0], [1, 1]]), np.array([[1, 1, 1, 0]]).T


def xor_load_data_set():
    return np.array([[0, 1], [0, 0], [1, 0], [1, 1]]), np.array([[1, 0, 1, 0]]).T
