from pretraining.nn import NeuralNetwork
from keras.models import load_model
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
        nn = NeuralNetwork()
        nn.model = load_model(path + path_to_nn)
        model = nn.model
    return model


def save_nn(path_to_nn, model):
    # TODO
    pass
