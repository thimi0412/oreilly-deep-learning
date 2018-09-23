import numpy as np
from exercises1 import cross_entropy_error
from exercises4 import numerical_gradient

'''
ソフトマックス関数
'''
def softmax(a):
    c = np.max(a)
    exp_a  = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    

    def predict(self, x):
        return np.dot(x, self.W)
    

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

if __name__ == '__main__':
    net = simpleNet()
    print(net.W)
    print('-'*20)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    print(p)
    print(np.argmax(p))
    print('-'*20)

    t =np.array([0, 0, 1])
    print(net.loss(x, t))

    f = lambda w: net.loss(x, t)
    dW = numerical_gradient(f, net.W)
    
    print(dW)

    