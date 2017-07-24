from conv_2d_clf import Conv2DClassifier
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    mnist = np.load('/media/DB/Student/WHL/finch/tensorflow-models/autoencoder/mnist.npz')
    X_train = mnist['x_train']
    y_train = mnist['y_train']
    X_test = mnist['x_test']
    y_test = mnist['y_test']
    X_train = (X_train / 255.0)[:, :, :, np.newaxis]
    X_test = (X_test / 255.0)[:, :, :, np.newaxis]

    clf = Conv2DClassifier((28,28), 1, 10)
    log = clf.fit(X_train, y_train, val_data=(X_test, y_test))
    pred = clf.predict(X_test)

    final_acc = (pred == y_test).mean()
    print("final testing accuracy: %.4f" % final_acc)