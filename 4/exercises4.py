import numpy as np

'''
勾配を計算
'''
def numerical_gradient(f, X):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(X)

    for idx in range(X.size):
        tmp_val = X[idx]
        # f(x+h)
        X[idx] = tmp_val + h
        fxh1 = f(X)
        
        # f(x-h)
        X[idx] = tmp_val - h
        fxh2 = f(X)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        X[idx] = tmp_val
    
    return grad


'''
勾配下降法
'''
def gradient_descent(f, init_X, lr=0.01, step_num=100):
    X = init_X

    for i in range(step_num):
        grad = numerical_gradient(f, X)
        X -= lr * grad
    
    return X


def function_1(x):
    return x[0]**2 + x[1]**2

if __name__ == '__main__':
    init_X = np.array([-3.0, 4.0])
    result = gradient_descent(function_1, init_X=init_X, lr=0.1, step_num=100)
    print(result)