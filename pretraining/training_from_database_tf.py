"""
TODO optimize hyperparameters
TODO NextMove/policy
TODO shuffle data after every epoch
"""
import tensorflow as tf
import numpy as np
from nn_tf import NeuralNetwork, sign_metric
from tensorflow.keras.models import load_model
import config_training as cf
from keras.callbacks import TensorBoard
import time
from keras.optimizers import SGD


def train(model):
    # fix random seed for reproducibility
    np.random.seed(0)

    # load data to X and Y
    print("Loading data")
    # remove from here
    model.load_data()

    # set up and print layer structure
    print("Creating model")
    model.create_network()
    print(model.model.summary())

    # Compile model
    print("Compiling model")

    losses = {
        "policy_head": "categorical_crossentropy",
        "value_head": "mean_squared_error",
    }
    loss_weights = {"policy_head": 10.0, "value_head": 1.0}

    metrics = ["accuracy", sign_metric]
    # optimizer = "adam"
    optimizer = "sgd"

    model.model.compile(loss=losses, optimizer=optimizer, metrics=metrics, loss_weights=loss_weights)

    # Vsualizo with Tensorboard
    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

    # Fit the model
    print("Fitting model")

    model.model.fit(model.train_data_generator,
                    steps_per_epoch=model.n_train,
                    epochs=cf.EPOCHS,
                    verbose=1,
                    validation_data=model.validation_data_generator,
                    validation_steps=model.n_val,
                    use_multiprocessing=True,
                    callbacks=[tensorboard])

    # TODO  is this automatically on gpu? cluster
    # Save the model
    print("Saving model")
    model.model.save("model_save")
    # evaluate the model and print the results.
    # print("Evaluating model")
    # self.evaluate_model()


def load_pretrained(model_path):
    print("Load Model:")
    model = load_model(model_path)
    return model


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


def learning_rate(epoch):
    """
    50 000 000
    # at 0.5 1e8 lr = 0.35
    # at 0.75 1e8 lr = 0.05
    atfer that linaer until 2
    than constantly

    This assumes around 15 000 000 total samples
    epoch 3
    """
    # TODO:
    raise NotImplementedError

    print("Current Epoch starting at 0?: ", epoch)
    processed_samples = (epoch-1) * cf.N_SAMPLES

    if epoch > 20:
        return 0.005
    elif epoch > 7:
        return 0.05
    elif epoch > 4:
        pass
    """
    weird stuff
    if processed_samples > 200_000_000:
        return 0.005
    elif processed_samples > 75_000_000:
        return 75_000_000 * 0.05 / processed_samples
    """


def main():
    network = NeuralNetwork()
    if cf.RESTORE_CHECKPOINT:
        path = "checkpoints/old_model"
        if cf.GDRIVE_FOLDER:
            path = "/content/" + path
        print("Restore Checkpoint from ", path)
        network.model = load_pretrained(path)

    train(network)
    exit()
    network.evaluate()


if __name__ == "__main__":
    main()
