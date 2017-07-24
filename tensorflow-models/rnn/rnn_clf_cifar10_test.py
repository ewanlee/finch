from rnn_clf import RNNClassifier
import utils

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = utils.load_cifar10_data()

    X_train = (X_train / 255.0).mean(axis=3)
    X_test = (X_test / 255.0).mean(axis=3)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    print(X_train.shape)

    clf = RNNClassifier(n_in=32, n_seq=32, n_out=10)
    log = clf.fit(X_train, y_train, val_data=(X_test, y_test))
    pred = clf.predict(X_test)
    final_acc = (pred == y_test).mean()
    print("final testing accuracy: %.4f" % final_acc)
