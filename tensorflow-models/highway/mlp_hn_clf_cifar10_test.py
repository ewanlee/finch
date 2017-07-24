from mlp_hn_clf import HighwayClassifier
import numpy as np
import tensorflow as tf
import utils


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = utils.load_cifar10_data()

    X_train = (X_train/255.0).mean(axis=3).reshape(-1, 32*32)
    X_test = (X_test/255.0).mean(axis=3).reshape(-1, 32*32)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    clf = HighwayClassifier(32*32, 10, 20)
    log = clf.fit(X_train, y_train, n_epoch=30, en_exp_decay=False, val_data=(X_test, y_test))
    pred = clf.predict(X_test)

    final_acc = (pred == y_test).mean()
    print("final testing accuracy: %.4f" % final_acc)