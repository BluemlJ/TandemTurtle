"""
Neural Network architecture we use for training and playing
on the magnificent Bughouse chess game
"""
import numpy as np
# from keras.optimizers import adam  # , SGD
from keras import regularizers
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add, concatenate
from keras.models import Model
from math import ceil
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
import time
# import config_training as cf
import pretraining.config_training as cf
# from data_generator import generate_value_batch, num_samples, generate_value_policy_batch
from pretraining.data_generator import generate_value_batch, num_samples, generate_value_policy_batch


class NeuralNetwork:

    def __init__(self):
        self.model = None
        self.train_data_generator = None
        self.validation_data_generator = None
        self.test_data_generator = None
        self.in_dim = (34, 8, 8)
        self.out_dim_value_head = 1
        self.out_dim_policy_head = 2272     # TODO: maybe make it dynamic not hard coded
        self.n_train = None
        self.n_val = None
        self.n_test = None
        # Todo define these values already here

        self.create_network()

    def load_data(self):
        self.train_data_generator = generate_value_policy_batch(
            cf.BATCH_SIZE,
            cf.GDRIVE_FOLDER + "data/position.train",
            cf.GDRIVE_FOLDER + "data/result.train",
            cf.GDRIVE_FOLDER + "data/nm.train")
        self.validation_data_generator = generate_value_policy_batch(
            cf.BATCH_SIZE,
            cf.GDRIVE_FOLDER + "data/position.validation",
            cf.GDRIVE_FOLDER + "data/result.validation",
            cf.GDRIVE_FOLDER + "data/nm.validation")
        self.test_data_generator = generate_value_policy_batch(
            cf.BATCH_SIZE,
            cf.GDRIVE_FOLDER + "data/position.test",
            cf.GDRIVE_FOLDER + "data/result.test",
            cf.GDRIVE_FOLDER + "data/nm.test")

        # TODO remove, it is for debugging
        sample = next(self.train_data_generator)
        print("Train data generator shapes")
        [[print(f"{key}: {x[key].shape}") for key in x] for x in sample]

        if cf.TEST_MODE:
            print("\n\n\n --------------------------------------- \n")
            print("RUNNING IN TEST MODE")
            print("\n --------------------------------------- \n\n\n")
            self.n_train = 1
            self.n_val = 1
            self.n_test = 1
        else:
            self.n_train = ceil(num_samples(cf.GDRIVE_FOLDER + "data/result.train") / cf.BATCH_SIZE)
            self.n_val = ceil(num_samples(cf.GDRIVE_FOLDER + "data/result.validation") / cf.BATCH_SIZE)
            self.n_test = ceil(num_samples(cf.GDRIVE_FOLDER + "data/result.test") / cf.BATCH_SIZE)

    def create_network(self):
        # create input
        board_input = Input(shape=self.in_dim)

        # apply convolutional layer
        with tf.name_scope(f"conv_input"):
            x = self.convolutional_layer(board_input)

        # apply residual layers
        for i in range(cf.NR_RESIDUAL_LAYERS):
            with tf.name_scope(f"res_{i}"):
                x = self.residual_layer(x)

        board_model = Model(inputs=board_input, outputs=x, name="board_model")

        main_input = Input(shape=self.in_dim, name="input_1")
        side_board_input = Input(shape=self.in_dim, name="input_2")

        out_main = board_model(main_input)
        out_side_board = board_model(side_board_input)

        with tf.name_scope(f"concatination"):
            concatenated = concatenate([out_main, out_side_board])

        # apply policy head and value head
        y = self.policy_head(concatenated)
        x = self.value_head(concatenated)

        self.model = Model(inputs=[main_input, side_board_input],
                           outputs=[x, y])
        # create model

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

    def value_head(self, x):
        with tf.name_scope("value_head"):
            x = Conv2D(
                filters=cf.NR_CONV_FILTERS_VALUE_HEAD,
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
                self.out_dim_value_head,
                use_bias=False,
                activation='tanh',
                kernel_regularizer=regularizers.l2(cf.REG_CONST),
                name='value_head',
            )(x)

        return x

    def policy_head(self, x):
        with tf.name_scope("policy_head"):
            x = Conv2D(
                filters=cf.NR_CONV_FILTERS_POLICY_HEAD,
                kernel_size=(1, 1),
                data_format="channels_first",
                padding='same',
                use_bias=False,
                activation='linear',
                kernel_regularizer=regularizers.l2(cf.REG_CONST)
            )(x)

            x = BatchNormalization(axis=1)(x)
            x = LeakyReLU()(x)

            x = Flatten()(x)

            # check if softmax makes sense here (it should since we use cross entropy) Was "linaer" before (I removed the to do because I think it does make sense.
            # TODO
            x = Dense(
                self.out_dim_policy_head,
                use_bias=False,
                activation='softmax',
                kernel_regularizer=regularizers.l2(cf.REG_CONST),
                name='policy_head'
            )(x)

        return x
