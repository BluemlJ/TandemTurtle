from pretraining.nn_tf import NeuralNetwork, sign_metric
from tensorflow.keras.models import load_model
import os
import time


def save_weights(path_to_nn):
    path = os.getcwd()

    # If file is already existent, skip this part
    if not os.path.isfile(path + path_to_nn + ".h5"):
        model = load_model(path + path_to_nn,
                           custom_objects={'sign_metric': sign_metric})
        model.save_weights(path + path_to_nn + ".h5")


def load_nn(path_to_nn="", save_weights_bool=False, load_weights=False):
    """
    Load NN from path, if not specified will create new and empty model

    :param path_to_nn: the path to nn
    :param save_weights_bool: If true will save current model path as weights, will check if
    already saved. Set to false to enable faster performance if weights already saved
    :param load_weights: If true will load .h5 as weights. Will only work if model config
    is the same as the loaded model. Set to false if unsure of architecture
    :return:
    """
    # Load pre trained model if path exists
    st_time = time.time()
    if path_to_nn == "":
        print("Path empty, loading clean nn")
        nn = NeuralNetwork()
        model = nn.model
    else:
        path = os.getcwd()
        if save_weights_bool:
            save_weights(path_to_nn)
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

    print(f"Time for loading one model with load weigths {load_weights}: ", time.time() - st_time)
    return model


def save_nn(path_to_nn, model):
    # TODO
    pass
