from conv_2d_clf import Conv2DClassifier
import numpy as np
import tensorflow as tf
import utils


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = utils.load_cifar10_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    clf = Conv2DClassifier((32,32), 3, 10)
    log = clf.fit(X_train, y_train, val_data=(X_test, y_test))
    pred = clf.predict(X_test)

    final_acc = (pred == y_test).mean()
    print("final testing accuracy: %.4f" % final_acc)
