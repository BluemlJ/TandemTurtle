"""
TODO optimize hyperparameters
TODO NextMove/policy
TODO shuffle data after every epoch
"""
import numpy as np
# from keras.optimizers import adam  # , SGD
from keras import regularizers
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add
from keras.models import Model
from math import ceil

import loss
import  config_training as cf
from data_generator import generate_value_batch, num_samples


class NeuralNetwork:

    def __init__(self):
        self.model = None
        self.train_data_generator = None
        self.validation_data_generator = None
        self.test_data_generator = None
        self.in_dim = (34, 8, 8)
        self.n_train = None
        self.n_val = None
        self.n_test = None

    ########
    # train_from_database:
    # main training function - loads data, creates and trains a network
    ########
    def train_from_database(self):
        # fix random seed for reproducibility
        np.random.seed(0)

        # load data to X and Y
        self.load_data()

        # set up and print layer structure
        self.create_network()
        print(self.model.summary())

        # Compile model
        self.model.compile(loss=loss.softmax_cross_entropy_with_logits, optimizer='adam')
        # Maybe try: optimizer=SGD(lr=self.learning_rate, momentum = cf.MOMENTUM) (like model.py)
        # Maybe try:  metrics=['accuracy']

        # Fit the model
        self.model.fit_generator(self.test_data_generator, steps_per_epoch=ceil(self.n_train / cf.BATCH_SIZE),
                                 epochs=cf.EPOCHS,
                                 verbose=1, validation_data=self.validation_data_generator,
                                 validation_steps=self.n_val)

        # TODO  automatically on gpu? cluster

        # evaluate the model and print the results.
        self.evaluate_model()

    def load_data(self):
        self.train_data_generator = generate_value_batch(cf.BATCH_SIZE,"data/position.train", "data/result.train", False)
        self.validation_data_generator = generate_value_batch(cf.BATCH_SIZE,"data/position.validation", "data/result.validation", False)
        self.test_data_generator = generate_value_batch(cf.BATCH_SIZE,"data/position.test", "data/result.test", False)

        self.n_train = num_samples("data/result.train")
        self.n_val = num_samples("data/result.validation")
        self.n_test = num_samples("data/result.test")

    def create_network(self):
        # create input
        main_input = Input(shape=self.in_dim)

        # apply convolutional layer
        x = self.convolutional_layer(main_input)

        # apply residual layers
        for i in range(cf.NR_RESIDUAL_LAYERS):
            x = self.residual_layer(x)

        # apply value head
        x = self.value_head(x)

        # create model
        self.model = Model(inputs=[main_input], outputs=[x])

    @staticmethod
    def convolutional_layer(x):
        x = Conv2D(
            filters=cf.NR_CONV_FILTERS,
            kernel_size=cf.KERNEL_SIZE_CONVOLUTION,
            data_format="channels_first",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(cf.REG_CONST)
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)
        return x

    def residual_layer(self, x_in):
        x = self.convolutional_layer(x_in)
        x = Conv2D(
            filters=cf.NR_CONV_FILTERS,
            kernel_size=cf.KERNEL_SIZE_CONVOLUTION,
            data_format="channels_first",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(cf.REG_CONST)
        )(x)
        x = BatchNormalization(axis=1)(x)

        # Skip connection
        x = add([x_in, x])
        x = LeakyReLU()(x)
        return x

    @staticmethod
    def value_head(x):
        x = Conv2D(
            filters=1,
            kernel_size=(1, 1),
            data_format="channels_first",
            padding='same',
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(cf.REG_CONST)
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        # flatten x in order to put it into the dense NN
        x = Flatten()(x)

        # first fully connected layer
        x = Dense(
            cf.SIZE_VALUE_HEAD_HIDDEN,
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(cf.REG_CONST)
        )(x)

        x = LeakyReLU()(x)

        # second fully connected layer
        x = Dense(
            1,
            use_bias=False,
            activation='tanh',
            kernel_regularizer=regularizers.l2(cf.REG_CONST),
            name='value_head',
        )(x)

        return x

    def evaluate_model(self):
        scores_test = self.model.evaluate_generator(self.test_data_generator, steps=self.n_test)
        scores_train = self.model.evaluate_generator(self.train_data_generator, steps=self.n_train)
        print("\nTest data accuracy %s: %.2f%%" % (self.model.metrics_names[1], scores_test[1] * 100))
        print("\nTraining data accuracy %s: %.2f%%" % (self.model.metrics_names[1], scores_train[1] * 100))


nn = NeuralNetwork()
nn.train_from_database()
