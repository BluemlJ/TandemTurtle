"""
TODO optimize hyperparameters
TODO NextMove/policy
TODO shuffle data after every epoch
"""

if True:
    import tensorflow as tf
    import numpy as np
    from nn_tf import NeuralNetwork, sign_metric
    from tensorflow.keras.models import load_model
    import config_training as cf
    from tensorflow.keras.callbacks import TensorBoard, Callback
    import tensorflow.keras.backend as K
    import time

else:
    import tensorflow as tf
    import numpy as np
    from nn_tf import NeuralNetwork, sign_metric
    from keras.models import load_model
    import config_training as cf
    from keras.callbacks import TensorBoard
    import time


def train(model):
    # fix random seed for reproducibility
    np.random.seed(0)

   # load data to X and Y
    print("Loading data")
    # remove from here
    model.load_data()

    # set up and print layer structure
    print(model.model.summary())

    # Compile model
    print("Compiling model")

    losses = {
        "policy_head": "categorical_crossentropy",
        "value_head": "mean_squared_error",
    }
    loss_weights = {"policy_head": 10.0, "value_head": 1.0}

    metrics = ["accuracy", sign_metric]
    optimizer = "adam"
    # optimizer = "sgd"
    # lr_callback = MyLearningRateScheduler()

    model.model.compile(loss=losses, optimizer=optimizer, metrics=metrics, loss_weights=loss_weights)

    # Visualize with Tensorboard
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()), update_freq="batch")

    # Fit the model
    print("Fitting model")

    # for epoch in range(cf.EPOCHS):
    #     train_on_batches(model)

    model.model.fit(model.train_data_generator,
                    steps_per_epoch=model.n_train,
                    epochs=cf.EPOCHS,
                    verbose=1,
                    validation_data=model.validation_data_generator,
                    validation_steps=model.n_val,
                    callbacks=[tensorboard])

    # TODO  is this automatically on gpu? cluster
    # Save the model
    print("Saving model")
    model.model.save("model_save")
    # evaluate the model and print the results.
    # print("Evaluating model")
    # self.evaluate_model()


def test(model):
    np.random.seed(0)

    # load data to X and Y
    print("Loading data")
    # remove from here
    model.load_data()

    # set up and print layer structure
    print(model.model.summary())

    # Compile model
    print("Compiling model")

    losses = {
        "policy_head": "categorical_crossentropy",
        "value_head": "mean_squared_error",
    }
    loss_weights = {"policy_head": 10.0, "value_head": 1.0}

    metrics = ["accuracy", sign_metric]
    optimizer = "adam"

    model.model.compile(loss=losses, optimizer=optimizer, metrics=metrics, loss_weights=loss_weights)

    print("----")
    print("Train:")
    model.model.evaluate(model.train_data_generator, steps=2)
    print("---")
    print("Validation:")
    model.model.evaluate(model.validation_data_generator, steps=2)
    print("---")
    print("Test:")
    model.model.evaluate(model.test_data_generator, steps=2)
    print("---")


def load_pretrained(model_path):
    print("Load Model:")
    model = load_model(model_path, custom_objects={'sign_metric': sign_metric})
    return model


def train_on_batches(model):
    train_iter = model.train_data_generator.make_initializable_iterator()
    val_iter = model.validation_data_generator.make_initializable_iterator()

    train_batch = train_iter.get_next()
    val_batch = val_iter.get_next()

    # TODO:
    with tf.Session() as sess:
        sess.run([train_iter.initializer, val_iter.initializer])

        for i in range(5):
            x, y = sess.run(train_batch)
            loss = model.model.train_on_batch(x, y)
            print(loss)


def evaluate_model(self):
    # caculate accuracy by hand:
    correct = 0
    cor_idx = []
    for i in range(self.n_val):
        sample = next(self.test_data_generator)
        inputs = sample[0]
        labels = sample[1]

        print(inputs["input_1"])
        predictions = self.model.predict(inputs)
        for j in range(cf.BATCH_SIZE):
            pred = np.argmax(predictions[1][j])
            tru = np.argmax(labels["policy_head"][j])
            if pred == tru:
                correct += 1
                cor_idx.append(pred)

    acc = float(correct) / (cf.BATCH_SIZE * self.n_val)
    print("ACC:", acc)
    cor_idx = sorted(list(set(cor_idx)))

    print(cor_idx)
    print(len(cor_idx))

    scores_test = self.model.evaluate(self.test_data_generator, steps=self.n_test)
    scores_train = self.model.evaluate(self.train_data_generator, steps=self.n_train)
    print("Metric names: ", self.model.metrics_names)
    print("EVAUATION TEST: ", scores_test)
    print("EVAUATION TRAIN: ", scores_train)
    # print("\nTest data accuracy %s: %.2f%%" % (self.model.metrics_names[1], scores_test[1] * 100))
    # print("\nTraining data accuracy %s: %.2f%%" % (self.model.metrics_names[1], scores_train[1] * 100))


def softmax_cross_entropy_with_logits(y_true, y_pred):
    print("Loss used :-)")
    p = y_pred
    pi = y_true

    zero = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)

    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

    return loss


class MyLearningRateScheduler(Callback):
    """Learning rate scheduler.
    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self):
        super(MyLearningRateScheduler, self).__init__()

    def on_batch_begin(self, step, logs=None):
        epoch_percent = step * cf.BATCH_SIZE / cf.N_SAMPLES

        if epoch_percent >= 2.0:
            lr = 0.001
        elif epoch_percent >= 0.65:
            lr = -0.025 * epoch_percent + 0.051
        elif epoch_percent >= 0.42:
            lr = -1.3 * epoch_percent + 0.898
        else:
            lr = (epoch_percent + 0.01) * (0.35 / 0.42)

        print(f"Set learning rate {lr} at step {step}")
        K.set_value(self.model.optimizer.lr, lr)


def main():
    network = NeuralNetwork()
    if cf.RESTORE_CHECKPOINT:
        path = "checkpoints/old_model"
        if cf.GDRIVE_FOLDER:
            path = "/content/" + path
        print("Restore Checkpoint from ", path)
        network.model = load_pretrained(path)
    else:
        network.create_network()

    # test(network)
    # exit()
    train(network)
    exit()
    network.evaluate()


if __name__ == "__main__":
    main()
