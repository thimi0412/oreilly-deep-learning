import numpy as np
from exercises2 import numerical_diff

def function_2(x):
    return np.sum(x**2)


def function_temp1(x0):
    return x0*x0 + 4.0**2.0


def function_temp2(x1):
    return 3.0**2.0 + x1 * x1

if __name__ == '__main__':
    print(numerical_diff(function_temp1, 3.0))
    print(numerical_diff(function_temp2, 4.0))