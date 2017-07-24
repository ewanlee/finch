from conv_ae import ConvAE
import matplotlib.pyplot as plt
import utils


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = utils.load_cifar10_data()
    X_train = (X_train/255.0).reshape(-1, 32, 32, 3)
    X_test = (X_test/255.0).reshape(-1, 32, 32, 3)

    ae = ConvAE((32, 32), 3)
    ae.fit(X_train, X_test, n_epoch=50)
    X_test_pred = ae.predict(X_test)

    print("Plotting...")
    plt.switch_backend('agg')
    plt.imshow(X_test[21].reshape(32,32,3))
    plt.savefig('cifar10_raw')
    plt.imshow(X_test_pred[21].reshape(32,32,3))
    plt.savefig('cifar10_pred')
