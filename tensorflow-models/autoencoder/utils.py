import tensorflow as tf
import sys
import os
from six.moves import cPickle
import numpy as np
from tensorflow.contrib.keras.python.keras import backend as K

def lrelu(x, leak=0.2, name='lrelu'):
    """ Leaky rectifier
    Parameters
    ----------
    x: Tensor
        The tensor to apply the nolinearity to.
    leak: float, optional
        Leakage parameter.
    name: str, optional
        Variable scope to use.
    Returns
    -------
    x: Tensor
        Output of the nolinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def load_batch(fpath, label_key='labels'):
    f = open(fpath, 'rb')
    if sys.version_info < (3,):
        d = cPickle.load(f)
    else:
        d = cPickle.load(f, encoding='bytes')
        # decode utf8
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
    f.close()
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels

def load_cifar10_data(path='/media/DB/Student/WHL/finch/tensorflow-models/autoencoder/cifar-10-batches-py/'):
    num_train_samples = 50000

    x_train = np.zeros((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.zeros((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
       fpath = os.path.join(path, 'data_batch_' + str(i))
       data, labels = load_batch(fpath)
       x_train[(i - 1) * 10000 : i * 10000, :, :, :] = data
       y_train[(i - 1) * 10000 : i * 10000] = labels

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)
