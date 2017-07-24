from conv_ae import ConvAE
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    mnist = np.load('/media/DB/Student/WHL/finch/tensorflow-models/autoencoder/mnist.npz')
    X_train = mnist['x_train']
    y_train = mnist['y_train']
    X_test = mnist['x_test']
    y_test = mnist['y_test']
    X_train = (X_train/255.0).reshape(-1, 28, 28, 1)
    X_test = (X_test/255.0).reshape(-1, 28, 28, 1)

    ae = ConvAE((28, 28), 1)
    ae.fit(X_train, X_test, n_epoch=10)
    X_test_pred = ae.predict(X_test)

    plt.switch_backend('agg')
    plt.imshow(X_test[21].reshape(28, 28))
    plt.savefig('raw.png')
    plt.imshow(X_test_pred[21].reshape(28, 28))
    plt.savefig('pred.png')
