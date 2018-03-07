'''
By adidinchuk. adidinchuk@gmail.com.
https://github.com/adidinchuk/tf-neural-net
'''

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import utils as u

class Network:

    # TODO activations must be 1 longer then hidden layer shape (validate)
    def __init__(self, in_vector_shape, out_vector_shape, hidden_layer_shape, activations):
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

        correct_prediction = tf.equal(tf.round(self.final_output), self.target)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.init = tf.global_variables_initializer()
        self.session.run(self.init)

    # TODO: add plotting functionality
    def train(self, train_inputs, train_outputs, test_inputs, test_outputs, learning_rate=0.05,
              loss_function='l2', batch_size=200, epochs=10000, plot=False, auto_balance=False):

        self.init_loss_function(loss_function)
        self.optimization = tf.train.GradientDescentOptimizer(learning_rate)
        self.training_step = self.optimization.minimize(self.loss)

        training_loss, testing_loss, training_accuracy, testing_accuracy = [], [], [], []
        positive = []

        #p = np.where(inputs == [1])[0]

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

            training_loss.append(self.session.run(self.loss, feed_dict={
                self.features: train_inputs, self.target: train_outputs}))
            testing_loss.append(self.session.run(self.loss, feed_dict={
                self.features: test_inputs, self.target: test_outputs}))

            #positive.append(self.session.run(self.accuracy, feed_dict={
            #    self.features: inputs[p], self.target: outputs[p]}))

            training_accuracy.append(self.session.run(self.accuracy, feed_dict={
                self.features: train_inputs, self.target: train_outputs}))
            testing_accuracy.append(self.session.run(self.accuracy, feed_dict={
                self.features: test_inputs, self.target: test_outputs}))

        u.plot_loss(training_loss. testing_loss)
        u.plot_accuracy(training_accuracy. testing_accuracy)

    # TODO: add functionality to select between activation functions
    def activate(self, activation, input):
        return tf.divide(1., tf.add(1., tf.exp(input)))

    # TODO: add additional loss functions
    def init_loss_function(self, loss):
        if loss == 'l2':
            self.loss_l2()

    def loss_l2(self):
        self.loss = tf.reduce_mean(tf.square(self.final_output - self.target))
