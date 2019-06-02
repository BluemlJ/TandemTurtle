
from databaseTraining import config_training as cf
import databaseTraining.loss as loss

import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, LeakyReLU, add
# from keras.optimizers import adam  # , SGD
from keras import regularizers


class NeuralNetwork:

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.model = None

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
        self.model.fit(self.X_train, self.Y_train, epochs=cf.EPOCHS, batch_size=cf.BATCH_SIZE)
        # automatically on gpu? cluster

        # evaluate the model and print the results.
        self.evaluate_model()

    def load_data(self):

        # load  dataset #TODO: load data in form "gameStateRepresentation" -> later winner
        dataset = np.loadtxt("test.csv", delimiter=";")

        # split into input (X) and output (Y) variables
        self.X_train = dataset[100:, 0:2]  # TODO adjust to dataset dimension.
        self.Y_train = dataset[100:, 2]
        self.X_test = dataset[:100, 0:2]
        self.Y_test = dataset[:100, 2]

    def create_network(self):

        # create input
        main_input = Input(shape=self.X_train.size())  # TODO: does size return a tuple?

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
        scores_test = self.model.evaluate(self.X_test, self.Y_test)
        scores_train = self.model.evaluate(self.X_train, self.Y_train)
        print("\nTest data accuracy %s: %.2f%%" % (self.model.metrics_names[1], scores_test[1] * 100))
        print("\nTraining data accuracy %s: %.2f%%" % (self.model.metrics_names[1], scores_train[1] * 100))


nn = NeuralNetwork()
nn.train_from_database()
