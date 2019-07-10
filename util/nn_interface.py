from pretraining.nn_tf import NeuralNetwork, sign_metric
from tensorflow.keras.models import load_model
import os


def load_nn(path_to_nn=""):
    # Load pre trained model if path exists
    if path_to_nn == "":
        print("Path empty, loading clean nn")
        nn = NeuralNetwork()
        model = nn.model
    else:
        path = os.getcwd()
        print("Load nn from ", path + path_to_nn)
        model = load_model(path + path_to_nn,
                           custom_objects={'sign_metric': sign_metric})

    return model


def save_nn(path_to_nn, model):
    # TODO
    pass
