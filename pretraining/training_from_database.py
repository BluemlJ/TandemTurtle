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
from tensorflow.python.keras.callbacks import TensorBoard
import time
import pretraining.config_training as cf
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

    ########
    # train_from_database:
    # main training function - loads data, creates and trains a network
    ########
    def train_from_database(self):
        # fix random seed for reproducibility
        np.random.seed(0)

        # load data to X and Y
        print("Loading data")
        self.load_data()

        # set up and print layer structure
        print("Creating model")
        self.create_network()
        print(self.model.summary())

        # Debugging
        import tensorflow as tf
        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)


        from tensorflow.python.client import device_lib
        print(device_lib.list_local_devices())

        # Compile model
        print("Compiling model")

        losses = {
            "policy_head": "categorical_crossentropy",
            "value_head": "mean_squared_error",
        }
        # lossWeights = {"category_output": 1.0, "color_output": 1.0}
        # softmax_cross_entropy_with_logits

        self.model.compile(loss=losses, optimizer='adam',
                           metrics=["accuracy", "binary_accuracy", "categorical_accuracy"],
                           options=run_opts)
        # Maybe try: optimizer=SGD(lr=self.learning_rate, momentum = cf.MOMENTUM) (like model.py)
        # Maybe try:  metrics=['accuracy']

        # visualizo with Tensorboard
        # tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))

        # Fit the model
        print("Fitting model")
        self.model.fit_generator(self.train_data_generator,
                                 steps_per_epoch=self.n_train,
                                 epochs=cf.EPOCHS,
                                 verbose=1,
                                 validation_data=self.validation_data_generator,
                                 validation_steps=self.n_val)
                                 #callbacks=[tensorboard])



        # TODO  is this automatically on gpu? cluster
        # Save the model
        print("Saving model")
        self.model.save("model_save")
        # evaluate the model and print the results.
        print("Evaluating model")
        self.evaluate_model()

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
            self.n_train = 1000
            self.n_val = 10
            self.n_test = 10
        else:
            self.n_train = ceil(num_samples(cf.GDRIVE_FOLDER + "data/result.train") / cf.BATCH_SIZE)
            self.n_val = ceil(num_samples(cf.GDRIVE_FOLDER + "data/result.validation") / cf.BATCH_SIZE)
            self.n_test = ceil(num_samples(cf.GDRIVE_FOLDER + "data/result.test") / cf.BATCH_SIZE)

    def create_network(self):
        # create input
        main_input = Input(shape=self.in_dim)

        # apply convolutional layer
        x = self.convolutional_layer(main_input)

        # apply residual layers
        for i in range(cf.NR_RESIDUAL_LAYERS):
            x = self.residual_layer(x)

        # apply policy head and value head
        y = self.policy_head(x)
        x = self.value_head(x)

        print("policy head: ", y)
        print("value head: ", x)
        # create model
        self.model = Model(inputs=[main_input], outputs=[x, y])

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
            self.out_dim_value_head,
            use_bias=False,
            activation='tanh',
            kernel_regularizer=regularizers.l2(cf.REG_CONST),
            name='value_head',
        )(x)

        return x

    def policy_head(self, x):
        x = Conv2D(
            filters=2, kernel_size=(1, 1), data_format="channels_first", padding='same', use_bias=False, activation='linear', kernel_regularizer=regularizers.l2(cf.REG_CONST)
        )(x)

        x = BatchNormalization(axis=1)(x)
        x = LeakyReLU()(x)

        x = Flatten()(x)

        # TODO check if softmax makes sense here (it should since we use cross entropy) Was "linaer" before
        x = Dense(
            self.out_dim_policy_head, use_bias=False, activation='softmax', kernel_regularizer=regularizers.l2(cf.REG_CONST), name='policy_head'
        )(x)

        return (x)

    def load_and_evaluate(self):
        self.load_data()

        from keras.models import load_model

        self.model = load_model("model_save")

        print("Evaluating model")
        self.evaluate_model()


    def evaluate_model(self):
        # caculate accuracy by hand:
        correct = 0
        cor_idx = []
        for i in range(self.n_val):
            sample = next(self.test_data_generator)
            inputs = sample[0]
            labels = sample[1]

            print(inputs["input_1"])
            predictions = self.model.predict(inputs)
            for j in range(cf.BATCH_SIZE):
                pred = np.argmax(predictions[1][j])
                tru = np.argmax(labels["policy_head"][j])
                if pred == tru:
                    correct += 1
                    cor_idx.append(pred)

        acc = float(correct) / (cf.BATCH_SIZE * self.n_val)
        print("ACC:", acc)
        cor_idx = sorted(list(set(cor_idx)))

        print(cor_idx)
        print(len(cor_idx))

        scores_test = self.model.evaluate_generator(self.test_data_generator, steps=self.n_test)
        scores_train = self.model.evaluate_generator(self.train_data_generator, steps=self.n_train)
        print("Metric names: ", self.model.metrics_names)
        print("EVAUATION TEST: ", scores_test)
        print("EVAUATION TRAIN: ", scores_train)
        # print("\nTest data accuracy %s: %.2f%%" % (self.model.metrics_names[1], scores_test[1] * 100))
        # print("\nTraining data accuracy %s: %.2f%%" % (self.model.metrics_names[1], scores_train[1] * 100))


def softmax_cross_entropy_with_logits(y_true, y_pred):
    print("Loss used :-)")
    p = y_pred
    pi = y_true

    zero = tf.zeros(shape=tf.shape(pi), dtype=tf.float32)
    where = tf.equal(pi, zero)

    negatives = tf.fill(tf.shape(pi), -100.0)
    p = tf.where(where, negatives, p)

    loss = tf.nn.softmax_cross_entropy_with_logits(labels=pi, logits=p)

    return loss


nn = NeuralNetwork()
# nn.train_from_database()
nn.load_and_evaluate()