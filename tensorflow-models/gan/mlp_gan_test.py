from mlp_gan import MLP_GAN
import numpy as np
import tensorflow as tf

batch_size = 64
X_size = 15
G_size = 5
x_range = np.vstack([np.linspace(-1, 1, X_size) for _ in range(batch_size)])

def load_data():
    a = np.random.uniform(1, 2, size=batch_size)[:, np.newaxis]
    data = a * np.power(x_range, 2) + (a - 1)
    return data

if __name__ == '__main__':
    model = MLP_GAN(G_size, X_size)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for step in range(3000):
        rand_data = np.random.randn(batch_size, G_size)
        real_data = load_data()

        G_loss, D_loss, D_prob, G_prob, loss = sess.run([model.G_loss, model.D_loss,
            model.X_true_prob, model.G_true_prob, model.l2_loss],
            {model.G_in: rand_data, model.X_in: real_data})
        print("G loss: %.4f | D loss: %.4f | D prob: %.4f | G prob: %.4f | l2 loss: %.4f " %
              (G_loss, D_loss, D_prob.mean(), G_prob.mean(), loss))
