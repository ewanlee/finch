from mlp_ae import Autoencoder
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    mnist = np.load('/media/DB/Student/WHL/finch/tensorflow-models/autoencoder/mnist.npz')
    X_train = mnist['x_train']
    y_train = mnist['y_train']
    X_test = mnist['x_test']
    y_test = mnist['y_test']
    X_train = (X_train / 255.0).reshape(-1, 28 * 28)
    X_test = (X_test / 255.0).reshape(-1, 28 * 28)

    auto = Autoencoder(28 * 28, [256, 128, 32, 2])
    auto.fit(X_train, val_data=X_test)
    X_test_2d = auto.transform(X_test)
    X_test_pred = auto.predict(X_test)

    plt.switch_backend('agg')
    plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test)
    plt.savefig('2d.png')
    plt.imshow(X_test[21].reshape(28, 28))
    plt.savefig('raw.png')
    plt.imshow(X_test_pred[21].reshape(28, 28))
    plt.savefig('pred.png')
