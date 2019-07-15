from pretraining.nn_tf import NeuralNetwork, sign_metric
from tensorflow.keras.models import load_model
import os


def load_save_and_load(path_to_nn=""):
    nn = NeuralNetwork()
    model = nn.model
    path = os.getcwd()
    model.save_weights(path + path_to_nn + ".h5")

    nn = NeuralNetwork()
    model = nn.model
    model.load_weights(path + path_to_nn + ".h5")
    return model


def load_nn(path_to_nn="", save_weights=False, load_weights=False):
    # Load pre trained model if path exists
    if path_to_nn == "":
        print("Path empty, loading clean nn")
        nn = NeuralNetwork()
        model = nn.model
    else:
        path = os.getcwd()
        if save_weights:

            model = load_model(path + path_to_nn,
                               custom_objects={'sign_metric': sign_metric})
            model.save_weights(path + path_to_nn + ".h5")
        if load_weights:
            print("Load weights of nn from ", path + path_to_nn + ".h5")
            nn = NeuralNetwork()
            model = nn.model
            model.load_weights(path + path_to_nn + ".h5")
        else:
            path = os.getcwd()
            print("Load nn from ", path + path_to_nn)
            model = load_model(path + path_to_nn,
                               custom_objects={'sign_metric': sign_metric})

    return model


def save_nn(path_to_nn, model):
    # TODO
    pass
