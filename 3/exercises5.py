from minist import load_mnist
import numpy as np
from PIL import Image

def image_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


if __name__ == '__main__':
    (X_train, t_train), (X_test, t_test) = load_mnist(flatten=True, normalize=False)
    img = X_train[0]
    label = t_train[0]
    print(label)

    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)

    image_show(img)
