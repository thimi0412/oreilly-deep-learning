import numpy as np
from minist import load_mnist

'''
2乗和誤差
'''
def mean_squared_error(y, t):
    return 0.5 * np.sum((y -t)**2)


'''
交差エントロピー誤差
'''
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


if __name__ == '__main__':
    (X_train, t_train), (X_test, t_test) = load_mnist(flatten=True, normalize=False)

    print(X_train.shape)
    print(t_train.shape)

    train_size = X_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    X_batch = X_train[batch_mask]
    t_batch = t_train[batch_mask]