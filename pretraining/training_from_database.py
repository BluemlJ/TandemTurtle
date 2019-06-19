"""
TODO optimize hyperparameters
TODO NextMove/policy
TODO shuffle data after every epoch
"""
import numpy as np
from nn import NeuralNetwork
from keras.models import load_model
import config_training as cf
from data_generator import generate_value_batch, num_samples, generate_value_policy_batch


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
    # lossWeights = {"category_output": 1.0, "color_output": 1.0}
    # softmax_cross_entropy_with_logits

    model.model.compile(loss=losses, optimizer='adam',
                       metrics=["accuracy", "binary_accuracy", "categorical_accuracy"])
    # Maybe try: optimizer=SGD(lr=self.learning_rate, momentum = cf.MOMENTUM) (like model.py)
    # Maybe try:  metrics=['accuracy']

    # visualizo with Tensorboard
    # tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

    # Fit the model
    print("Fitting model")
    model.model.fit_generator(model.train_data_generator,
                             steps_per_epoch=model.n_train,
                             epochs=cf.EPOCHS,
                             verbose=1,
                             validation_data=model.validation_data_generator,
                             validation_steps=model.n_val)
                             #callbacks=[tensorboard])

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

    scores_test = self.model.evaluate_generator(self.test_data_generator, steps=self.n_test)
    scores_train = self.model.evaluate_generator(self.train_data_generator, steps=self.n_train)
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


def main():
    network = NeuralNetwork()
    network.model = load_pretrained("model_save")
    #train(network)
    exit()
    network.evaluate()


if __name__ == "__main__":
    main()