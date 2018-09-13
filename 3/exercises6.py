import pickle
import numpy as np
from minist import load_mnist
from exercises1 import sigmoid
from exercises2 import indentity_function

def get_data():
    (_, _), (X_test, t_test) = load_mnist(flatten=True, normalize=False)
    return X_test, t_test


def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, X):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(X, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = indentity_function(a3)

    return y


if __name__ == '__main__':
    X, t = get_data()
    network = init_network()

    batch_size = 100
    accuracy_cnt = 0

    for i in range(0, len(X), batch_size):
        X_batch = X[i: i + batch_size]
        y_batch = predict(network, X_batch)
        p = np.argmax(y_batch, axis=1) # 確率がもっとも高い要素のインデックスを取得
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
    
    print("Accuracy:" + str(float(accuracy_cnt) / len(X)))