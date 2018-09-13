# %%
import numpy as np
import matplotlib.pylab as plt

'''
ReLU関数
'''
def relu(x):
    return np.maximum(0, x)


'''
シグモイド関数
'''
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


'''
ステップ関数
'''
def step_function(x):
    return np.array(x > 0, dtype=np.int)

if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    # y = sigmoid(x)
    # y = step_function(x)
    # plt.plot(x, y)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()