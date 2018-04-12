'''
By adidinchuk. adidinchuk@gmail.com.
https://github.com/adidinchuk/tf-neural-net
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#import utils as u


class Network:

    def __init__(self, in_vector_shape, out_vector_shape, hidden_layer_shape, activations):\

        # validate hidden_layer_shape and activation lengths
        if len(activations) != len(hidden_layer_shape) + 1:
            raise ValueError('activations length must be 1 greater then the length of hidden_layer_shape')

        self.session = tf.Session()
        self.feature_shape = in_vector_shape
        self.target_shape = out_vector_shape
        self.hidden_layer_shape = hidden_layer_shape
        self.activations = activations

        self.features = tf.placeholder(shape=[None, self.feature_shape], dtype=tf.float32)
        self.target = tf.placeholder(shape=[None, self.target_shape], dtype=tf.float32)
        self.hidden_layers_A, self.hidden_layers_b, self.output = [], [], []

        # first A + b layer
        self.hidden_layers_A.append(tf.Variable(tf.random_normal(
            shape=[self.feature_shape, self.hidden_layer_shape[0]])))
        self.hidden_layers_b.append(tf.Variable(tf.random_normal(shape=[self.hidden_layer_shape[0]])))

        # generate layers
        for layer in range(len(self.hidden_layer_shape) - 1):
            self.hidden_layers_A.append(tf.Variable(tf.random_normal(
                shape=[self.hidden_layer_shape[layer], self.hidden_layer_shape[layer + 1]])))
            self.hidden_layers_b.append(tf.Variable(tf.random_normal(
                shape=[self.hidden_layer_shape[layer + 1]])))

        # last A + b layer
        self.hidden_layers_A.append(tf.Variable(tf.random_normal(
            shape=[self.hidden_layer_shape[len(hidden_layer_shape) - 1], self.target_shape])))
        self.hidden_layers_b.append(tf.Variable(tf.random_normal(shape=[self.target_shape])))

        # first activation
        self.output.append(self.activate(activations[0], tf.add(tf.matmul(
            self.features, self.hidden_layers_A[0]), self.hidden_layers_b[0])))

        # link each layer
        for layer in range(1, len(self.hidden_layer_shape)):
            self.output.append(self.activate(activations[layer],  tf.add(tf.matmul(
                self.output[len(self.output)-1],
                self.hidden_layers_A[layer]),
                self.hidden_layers_b[layer])))

        # last activation
        self.final_output = self.activate(activations[len(activations)-1], tf.add(tf.matmul(
            self.output[len(self.output)-1],
            self.hidden_layers_A[len(self.hidden_layers_A)-1]),
            self.hidden_layers_b[len(self.hidden_layers_b)-1]))

        self.loss, self.optimization, self.training_step = None, None, None

        # accuracy calculation functions
        correct_prediction = tf.equal(tf.round(self.final_output), self.target)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # initialize variables
        self.init = tf.global_variables_initializer()
        self.session.run(self.init)

    def train(self, train_inputs, train_outputs, test_inputs, test_outputs, learning_rate=0.05,
              loss_function='l2', batch_size=200, epochs=10000, plot=False, auto_balance=False):

        self.init_loss_function(loss_function)
        self.optimization = tf.train.GradientDescentOptimizer(learning_rate)
        self.training_step = self.optimization.minimize(self.loss)

        training_loss_results, testing_loss_results = [], []
        training_accuracy_results, testing_accuracy_results = [], []

        for iteration in range(epochs):
            batch_index = []
            # normalize the imbalance
            if auto_balance:
                classes = np.unique(train_outputs)
                for classification in classes:
                    batch_index = np.append(batch_index, np.random.choice(
                        np.where(train_outputs == classification)[0], int(batch_size / len(classes))))
                    batch_index = batch_index.astype(int)
            else:
                batch_index = np.random.choice(len(train_inputs), size=batch_size)

            self.session.run(self.training_step, feed_dict={
                self.features: train_inputs[batch_index], self.target: train_outputs[batch_index]})

            # if plotting, generate loss and accuracy values
            if plot:
                training_accuracy, training_loss = self.generate_step_tracking_data(train_inputs, train_outputs)
                training_loss_results.append(training_loss)
                training_accuracy_results.append(training_accuracy)

                testing_accuracy, testing_loss = self.generate_step_tracking_data(test_inputs, test_outputs)
                testing_loss_results.append(testing_loss)
                testing_accuracy_results.append(testing_accuracy)

            # print results
            if (iteration+1) % (epochs / 5) == 0:
                # if not plotting, get intermittent accuracy and loss
                if not plot:
                    training_accuracy, training_loss = self.generate_step_tracking_data(train_inputs, train_outputs)
                    training_loss_results.append(training_loss)
                    training_accuracy_results.append(training_accuracy)

                    testing_accuracy, testing_loss = self.generate_step_tracking_data(test_inputs, test_outputs)
                    testing_loss_results.append(testing_loss)
                    testing_accuracy_results.append(testing_accuracy)

                #u.print_progress(iteration, epochs, training_loss_results[-1], training_accuracy_results[-1])

        #if plot:
            #u.plot_loss(training_loss_results, testing_loss_results)
            #u.plot_accuracy(training_accuracy_results, testing_accuracy_results)

    # TODO: add functionality to select between activation functions
    def activate(self, activation, ax):
        if activation == 'sigmoid':
            return self.sigmoid_activation(ax)
        elif activation == 'linear':
            return self.linear_activation(ax)
        elif activation == 'relu':
            return self.relu_activation(ax)
        else:
            print('requested activation function ' + activation + ' not found, using sigmoid')
            return self.sigmoid_activation(ax)

    # Activation functions #
    # Relu
    def relu_activation(self, ax):
        return tf.maximum(0., ax)

    # Linear
    def linear_activation(self, ax):
        return ax

    # Exponential
    def sigmoid_activation(self, ax):
        return tf.divide(1., tf.add(1., tf.exp(tf.negative(ax))))

    def init_loss_function(self, loss_function):
        if loss_function == 'l2':
            self.loss_l2()
        elif loss_function == 'l1':
            self.loss_l1()
        elif loss_function == 'cross_entropy':
            self.cross_entropy()
        else:
            print('requested loss function ' + loss_function + ' not found, using L2')
            self.loss_l2()

    # Loss functions #
    # L2 loss
    def loss_l2(self):
        self.loss = tf.reduce_mean(tf.square(self.final_output - self.target))

    # L1 loss function L = |a|
    def loss_l1(self):
        self.loss = tf.reduce_mean(tf.abs(self.final_output - self.target))

    # Cross Entropy Loss function
    def cross_entropy(self):
        self.loss = tf.reduce_mean(tf.multiply(self.target, tf.log(self.final_output)) + tf.multiply(
            tf.subtract(1., self.target), tf.log(tf.subtract(1., self.final_output))))

    def generate_step_tracking_data(self, inputs, targets):
        accuracy = self.session.run(self.accuracy, feed_dict={self.features: inputs, self.target: targets})
        loss = self.session.run(self.loss, feed_dict={self.features: inputs, self.target: targets})
        return accuracy, loss
