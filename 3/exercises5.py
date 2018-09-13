from minist import load_mnist

(X_train, X_test), (y_train, y_test) = load_mnist(flatten=True, normalize=False)

print(X_train.shape)