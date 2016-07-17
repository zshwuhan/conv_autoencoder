'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf

import time
from tensorflow.examples.tutorials.mnist import input_data

from src.model.utils import *
from src.visualization.visualize import *


class CNN:
    def __init__(self, hyp_param, observers=None):
        self.hyp_param = hyp_param

        self.observers = observers

        with tf.name_scope("X"):
            self.x = tf.placeholder(tf.float32, [None, hyp_param["n_input"]], name="X")
        with tf.name_scope("Y"):
            self.y = tf.placeholder(tf.float32, [None, hyp_param["n_classes"]], name="Y")
        with tf.name_scope("Dropout"):
            self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        # Store layers weight & bias
        conv1_n_filters = 32
        conv2_n_filters = 16
        with tf.name_scope("CNN"):
            with tf.name_scope("weights"):
                weights = {
                    # 5x5 conv, 1 input, 32 outputs
                    'wc1': tf.Variable(tf.random_normal([5, 5, 1, conv1_n_filters])),
                    # 5x5 conv, 32 inputs, 64 outputs
                    'wc2': tf.Variable(tf.random_normal([5, 5, conv1_n_filters, conv2_n_filters])),
                    # fully connected, 7*7*64 inputs, 1024 outputs
                    'wd1': tf.Variable(tf.random_normal([7 * 7 * conv2_n_filters, 1024])),
                    # 1024 inputs, 10 outputs (class prediction)
                    'out': tf.Variable(tf.random_normal([1024, hyp_param["n_classes"]]))
                }
            with tf.name_scope("biases"):
                biases = {
                    'bc1': tf.Variable(tf.random_normal([conv1_n_filters])),
                    'bc2': tf.Variable(tf.random_normal([conv2_n_filters])),
                    'bd1': tf.Variable(tf.random_normal([1024])),
                    'out': tf.Variable(tf.random_normal([hyp_param["n_classes"]]))
                }

            self.convolutions, self.fc1 = [], None

            self.output = self.compute_output(self.x, weights, biases)

        # Define loss and optimizer
        with tf.name_scope("Cost"):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output, self.y), name="cost")
            tf.scalar_summary('cost', self.cost)
        with tf.name_scope("Optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=hyp_param["learning_rate"]).minimize(self.cost)

        with tf.name_scope("Evaluation"):
            with tf.name_scope("Correct_prediction"):
                self.output_softmax = tf.nn.softmax(self.output, name="softmax")
                self.prediction = tf.argmax(self.output_softmax, 1, name="max_value")
                self.correct_pred = tf.equal(self.prediction, tf.argmax(self.y, 1),
                                             name="correct_prediction")
            with tf.name_scope("Accuracy"):
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name="accuracy")
                tf.scalar_summary("accuracy", self.accuracy)

        self.train_writer = None
        self.merged = None

        self.feed_dict = dict()
        self.timestamp = time.strftime("%Y_%B_%d__%H:%M", time.localtime())
        self.tb_path = os.path.join(DataPath.base, "tf_logs", self.timestamp)

        self.saver = tf.train.Saver()
        self.model_path = self.tb_path + ".tmp"

    def compute_output(self, _x, _weights, _biases):
        with tf.name_scope("Reshaping_and_transposing"):
            _x = tf.reshape(_x, shape=[-1, 28, 28, 1])

        self.convolutions.append(self.conv_layer(_x, _weights["wc1"], _biases["bc1"], 2))
        self.convolutions.append(self.conv_layer(self.convolutions[-1], _weights["wc2"], _biases["bc2"], 2))

        # Reshape conv2 output to fit fully connected layer input
        with tf.name_scope("Fully_connected"):
            self.fc1 = tf.reshape(self.convolutions[-1], [-1, _weights['wd1'].get_shape().as_list()[0]])

            self.fc1 = tf.matmul(self.fc1, _weights['wd1']) + _biases['bd1']
            self.fc1 = tf.nn.relu(self.fc1)
            # self.fc1 = tf.sigmoid(self.fc1)

            self.fc1 = tf.nn.dropout(self.fc1, self.hyp_param["dropout"])

            output = tf.matmul(self.fc1, _weights['out']) + _biases['out']

        return output

    def conv_layer(self, _inp, _weights, _biases, k):
        with tf.name_scope("Convolution_layer"):
            conv = self.conv2d(_inp, _weights, _biases)
            conv = self.maxpool2d(conv, k=k)
        return conv

    @staticmethod
    def conv2d(x, w, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    @staticmethod
    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def train(self, data, batch_size, training_iters, display_step=20):
        assert training_iters % batch_size == 0

        with tf.Session() as sess:
            self.merged = tf.merge_all_summaries()
            self.train_writer = tf.train.SummaryWriter(os.path.join(self.tb_path, "train"), sess.graph)
            sess.run(tf.initialize_all_variables())

            for step in range(int(training_iters / batch_size)):
                batch_x, batch_y = data.train.next_batch(batch_size)
                self.feed_dict = {self.x: batch_x, self.y: batch_y, self.keep_prob: self.hyp_param["dropout"]}
                sess.run(self.optimizer, feed_dict=self.feed_dict)
                if step % display_step == 0:
                    self.observers.notify(self, sess=sess, data=data, step=step)

            self.saver.save(sess, self.model_path)

    def run_function(self, model_path, fn, feed_dict):
        with tf.Session() as sess:
            self.saver.restore(sess, model_path)
            return sess.run(fn, feed_dict=feed_dict)

    def test(self, data, batch_size, model_path):
        test_data = data.test.images[:batch_size]
        test_label = data.test.labels[:batch_size]
        feed_dict = {self.x: test_data, self.y: test_label, self.keep_prob: 1}
        test_acc = self.run_function(model_path, self.accuracy, feed_dict)
        return test_acc

    def plot_filter(self, layer):
        batch_x, batch_y = mnist.train.next_batch(10)
        one_image = batch_x[9]
        fd = {self.x: np.reshape(one_image, [1, 784], order='F'), self.keep_prob: 1.0}
        units = self.run_function(self.model_path, layer, fd)

        filters = units.shape[3]
        print("n_filters:", filters)
        plt.figure(1, figsize=(20, 20))

        for i in range(0, filters):
            plt.subplot(int(np.sqrt(filters)) + 1, int(np.sqrt(filters)) + 1, i + 1)
            plt.title('Filter ' + str(i))
            plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")
            frame = plt.gca()
            frame.axes.get_xaxis().set_ticks([])
            frame.axes.get_yaxis().set_ticks([])
        plt.show()