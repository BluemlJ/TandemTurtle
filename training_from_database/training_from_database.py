import logging
import training_from_database.config_training as configt
#import loss

import numpy as np

import matplotlib.pyplot as plt

from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from keras.optimizers import SGD
from keras import regularizers

#from loss import softmax_cross_entropy_with_logits

import logger as lg

import keras.backend as K

class neural_network():

    def __init__(self):
        self.reg_const = configt.REG_CONST
        self.learning_rate = configt.LEARNING_RATE
        self.input_dim = configt.INPUT_DIM
        self.output_dim = configt.OUTPUT_DIM_VALUE_HEAD
        self.X = None
        self.Y = None
        self.model = None

    def load_data(self):
        # fix random seed for reproducibility
        np.random.seed(0)

        # load  dataset #TODO: load data in form "gamestateRepresentation" -> later winner
        dataset = np.loadtxt("test.csv", delimiter=";")

        # split into input (X) and output (Y) variables
        self.X = dataset[:, 0:2]  # TODO adjust to dataset dimension
        self.Y = dataset[:, 2]

    def train_from_database(self):
        # load data to X and Y
        self.load_data()

        # set up layer structure
        self.create_model()

        # Compile model #todo look at model.py
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Fit the model
        self.model.fit(self.X, self.Y, epochs=configt.EPOCHS, batch_size=configt.BATCH_SIZE)  # automatically on gpu? cluster

        # evaluate the model
        self.evaluate_model()

    def create_model(self):
        model = Sequential()
        self.add_convolutional_layer(256, 3)
        # TODO: other Layers.

    def add_convolutional_layer(self, filters, kernel_size):
        self.model.add(Conv2D(
            filters = filters
            , kernel_size = kernel_size
            , data_format="channels_first"
            , padding = 'same'
            , use_bias=False
            , activation='linear'
            , kernel_regularizer = regularizers.l2(configt.REG_CONST)
            ))

    def evaluate_model(self):
        scores = self.model.evaluate(self.X, self.Y) #TODO here only test data.
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1] * 100))


