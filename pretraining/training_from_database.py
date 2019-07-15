"""
TODO optimize hyperparameters
TODO NextMove/policy
TODO shuffle data after every epoch
"""

print("THIS MODEL SHOULD NOT BE USED ANYMORE")
exit(0)

import numpy as np
from nn import NeuralNetwork
from tensorflow.keras.models import load_model
import config_training as cf
import keras.backend as K
from data_generator import generate_value_batch, num_samples, generate_value_policy_batch


def train(network):
    # fix random seed for reproducibility
    np.random.seed(0)

    # Create network if not existent
    if not network.model:
        print("Creating model")
        network.create_network()
    print(network.model.summary())

    # Compile model
    print("Compiling model")

    losses = {
        "policy_head": "categorical_crossentropy",
        "value_head": "mean_squared_error",
    }

    # TODO: add own metric for value accuracy so that it checks if the sign of value is correct
    # Can be done by def metric ...
    # K.sign(
    def sign_metric(y_true, y_pred):
        return K.mean(K.equal(K.sign(y_pred), y_true))

    network.model.compile(loss=losses, optimizer='adam',
                          metrics=["accuracy", sign_metric])
    # RMS Prop verwenden
    # SGD with momentum

    # Visualizo with Tensorboard
    # tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

    # Fit the model
    print("Fitting model")

    network.model.fit_generator(network.train_data_generator,
                             steps_per_epoch=network.n_train,
                             epochs=cf.EPOCHS,
                             verbose=1,
                             validation_data=network.validation_data_generator,
                             validation_steps=network.n_val)
                             #callbacks=[tensorboard])
    print("Saving model")
    network.model.save("checkpoints/model_save_big")

    # train_on_batches(network)


def train_on_batches(network):
    network.model.predi

def load_pretrained(model_path):
    print("Load Pretrained Model:")
    model = load_model(model_path)
    return model


def evaluate_model(network):
    # caculate accuracy by hand:
    correct = 0
    cor_idx = []
    for i in range(network.n_val):
        sample = next(network.test_data_generator)
        inputs = sample[0]
        labels = sample[1]

        print("shapes:")
        print(inputs["input_1"].shape)
        # print(inputs["input_1"])

        single_in = np.expand_dims(inputs["input_1"][0], axis=0)
        single_in2 = np.expand_dims(inputs["input_2"][0], axis=0)

        inputs = {
            "input_1": single_in,
            "input_2": single_in2
        }
        print("single in shap: ", single_in.shape)
        print("types: ", type(single_in))
        print(inputs)

        predictions = network.model.predict(inputs)

        print(predictions)
        exit()
        for j in range(cf.BATCH_SIZE):
            pred = np.argmax(predictions[1][j])
            tru = np.argmax(labels["policy_head"][j])
            if pred == tru:
                correct += 1
                cor_idx.append(pred)

    acc = float(correct) / (cf.BATCH_SIZE * network.n_val)
    print("ACC:", acc)
    cor_idx = sorted(list(set(cor_idx)))

    print(cor_idx)
    print(len(cor_idx))

    scores_test = network.model.evaluate_generator(network.test_data_generator, steps=network.n_test)
    scores_train = network.model.evaluate_generator(network.train_data_generator, steps=network.n_train)
    print("Metric names: ", network.model.metrics_names)
    print("EVAUATION TEST: ", scores_test)
    print("EVAUATION TRAIN: ", scores_train)


def main():
    print("DERICATED PLEASE USE _tf")
    exit(-1)

    network = NeuralNetwork()
    if cf.RESTORE_CHECKPOINT:
        path = "checkpoints/old_model"
        if cf.GDRIVE_FOLDER:
            path = "/content/" + path
        print("Restore Checkpoint from ", path)
        network.model = load_pretrained(path)

    print("Loading data")
    network.load_data()

    train(network)

    evaluate_model(network)


if __name__ == "__main__":
    main()
