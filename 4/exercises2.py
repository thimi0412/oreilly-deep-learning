# %%
import numpy as np
import matplotlib.pyplot as plt


'''
数値微分
'''
def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

'''
y = 0.01x^2 + 0.1x
'''
def function_1(x):
    return 0.01 * x **2 + 0.1 * x 

'''
接戦を引く
'''
def tangent_line(f, x):
    d = numerical_diff(f, x)
    y = f(x) - d*x
    return lambda t: d*t + y

if __name__ == '__main__':
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    

    tf = tangent_line(function_1, 5)
    y2 = tf(x)

    tf = tangent_line(function_1, 10)
    y3 = tf(x)

    plt.plot(x, y)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.legend(['f(x)', 'f\'(5)', 'f\'(10)'])
    plt.show()

    print('f(x) = 0.01x^2 + 0.1x の微分')
    dx_5 = numerical_diff(function_1, 5)
    print('f\'(5): 解析では0.2, 数値微分では{}'.format(dx_5))
    dx_10 = numerical_diff(function_1, 10)
    print('f\'(10): 解析では0.3, 数値微分では{}'.format(dx_10))

