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


def train(network):
    # fix random seed for reproducibility
    np.random.seed(0)

    # load data to X and Y
    print("Loading data")
    # remove from here
    network.load_data()

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
    # lossWeights = {"category_output": 1.0, "color_output": 1.0}
    # softmax_cross_entropy_with_logits

    network.model.compile(loss=losses, optimizer='adam',
                       metrics=["accuracy", "binary_accuracy", "categorical_accuracy"])
    # Maybe try: optimizer=SGD(lr=self.learning_rate, momentum = cf.MOMENTUM) (like model.py)
    # Maybe try:  metrics=['accuracy']

    # visualizo with Tensorboard
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

    # TODO  is this automatically on gpu? cluster
    # Save the model
    print("Saving model")
    network.model.save("checkpoints/model_save_big")
    # evaluate the model and print the results.
    # print("Evaluating model")
    # self.evaluate_model()


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
    # print("\nTest data accuracy %s: %.2f%%" % (self.model.metrics_names[1], scores_test[1] * 100))
    # print("\nTraining data accuracy %s: %.2f%%" % (self.model.metrics_names[1], scores_train[1] * 100))


def main():
    network = NeuralNetwork()
    if cf.RESTORE_CHECKPOINT:
        path = "checkpoints/model_save_big"
        if cf.GDRIVE_FOLDER:
            path = "/content/" + path
        print("Restore Checkpoint from ", path)
        network.model = load_pretrained(path)
    train(network)
    # exit()
    print("Loading data")
    # remove from here
    # network.load_data()
    evaluate_model(network)


if __name__ == "__main__":
    main()
