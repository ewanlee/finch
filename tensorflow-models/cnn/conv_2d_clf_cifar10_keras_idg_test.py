from conv_2d_clf import Conv2DClassifier
import tensorflow as tf
import utils

BATCH_SIZE = 128
N_EPOCH = 10

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = utils.load_cifar10_data()
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    model = Conv2DClassifier((32,32), 3, 10)
    datagen = tf.contrib.keras.preprocessing.image.ImageDataGenerator()
    datagen.fit(X_train)

    global_step = 0
    model.sess.run(tf.global_variables_initializer())
    for epoch in range(N_EPOCH):
        for local_step, (X_batch, Y_batch) in enumerate(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)):
            if local_step > len(X_train) // BATCH_SIZE:
                break
            lr = model.decrease_lr(True, global_step, N_EPOCH, len(X_train), BATCH_SIZE)
            _, loss, acc = model.sess.run([model.train_op, model.loss, model.acc],
                                          {model.X:X_batch, model.Y:Y_batch.squeeze(),
                                           model.lr:lr, model.keep_prob:0.5, model.train_flag:True})
            global_step += 1
            if local_step % 50 == 0:
                print ("Epoch %d/%d | Step %d/%d | train_loss: %.4f | train_acc: %.4f | lr: %.4f"
                        %(epoch+1, N_EPOCH, local_step, len(X_train)//BATCH_SIZE, loss, acc, lr))

        val_loss_list, val_acc_list = [], []
        for X_test_batch, Y_test_batch in zip(model.gen_batch(X_test, BATCH_SIZE),
                                              model.gen_batch(y_test, BATCH_SIZE)):
            v_loss, v_acc = model.sess.run([model.loss, model.acc],
                                           {model.X:X_test_batch, model.Y:Y_test_batch.squeeze(),
                                            model.keep_prob:1.0, model.train_flag:False})
            val_loss_list.append(v_loss)
            val_acc_list.append(v_acc)
        val_loss, val_acc = model.list_avg(val_loss_list), model.list_avg(val_acc_list)

        print ("Epoch %d/%d | train_loss: %.4f | train_acc: %.4f |" % (epoch+1, N_EPOCH, loss, acc),
                "test_loss: %.4f | test_acc: %.4f |" % (val_loss, val_acc),
                "lr: %.4f" % (lr) )
