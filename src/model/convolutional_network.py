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
    def __init__(self, hyp_param, filter_depths=(1, 10, 10, 20), filter_sizes=(3, 3, 3, 3), observers=None):
        self.hyp_param = hyp_param

        self.observers = observers

        with tf.name_scope("X"):
            self.x = tf.placeholder(tf.float32, [None, hyp_param["n_input"]], name="X")
            self.y = self.x

        encoder = []
        shapes = []
        with tf.name_scope("Reshaping_and_transposing"):
            reshaped = tf.reshape(self.x, shape=[-1, 28, 28, 1])
            self.y = tf.reshape(self.y, shape=[-1, 28, 28, 1])
            curr_input = reshaped
        with tf.name_scope("Autoencoder"):
            with tf.name_scope("Encoder"):
                for i, filter_size in enumerate(filter_sizes[1:]):
                    n_input = curr_input.get_shape().as_list()[3]
                    n_output = filter_depths[i]
                    output, w = self.conv_layer(curr_input, filter_size, n_input, n_output,
                                                name="Convolution_layer" + str(i))
                    encoder.append(w)
                    shapes.append(curr_input.get_shape().as_list())
                    curr_input = output

            with tf.name_scope("Feature_space"):
                self.z = curr_input
            encoder.reverse()
            shapes.reverse()
            with tf.name_scope("Decoder"):
                for i, filter_size in enumerate(shapes):
                    w = encoder[i]
                    shape = shapes[i]
                    output = self.deconv_layer(curr_input, w, shape, name="Deconvolution_layer" + str(i))
                    curr_input = output

        self.output = curr_input
        self.imgsum_real = tf.image_summary("Real_image", reshaped)
        self.imgsum_gen = tf.image_summary("Generated_image", self.output)

        # Define loss and optimizer
        with tf.name_scope("Cost"):
            self.cost = tf.reduce_mean(tf.square(self.output - self.y), name="cost")
            tf.scalar_summary('cost', self.cost)
        with tf.name_scope("Optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=hyp_param["learning_rate"]).minimize(self.cost)

        # with tf.name_scope("Evaluation"):
        #     with tf.name_scope("Correct_prediction"):
        #         self.output_softmax = tf.nn.softmax(self.output, name="softmax")
        #         self.prediction = tf.argmax(self.output_softmax, 1, name="max_value")
        #         self.correct_pred = tf.equal(self.prediction, tf.argmax(self.y, 1),
        #                                      name="correct_prediction")
        #     with tf.name_scope("Accuracy"):
        #         self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name="accuracy")
        #         tf.scalar_summary("accuracy", self.accuracy)
        # self.accuracy = -1

        self.train_writer = None
        self.merged = None

        self.feed_dict = dict()
        self.timestamp = time.strftime("%Y_%B_%d__%H:%M", time.localtime())
        self.tb_path = os.path.join(DataPath.base, "tf_logs", self.timestamp)

        self.saver = tf.train.Saver()
        self.model_path = self.tb_path + ".tmp"

    @staticmethod
    def conv_layer(_inp, filter_size, n_input, n_output, name="Convolution_layer", stride=2):
        with tf.variable_scope(name):
            w = tf.get_variable("weights", [filter_size, filter_size, n_input, n_output],
                                initializer=tf.truncated_normal_initializer())
            b = tf.Variable(tf.constant(0.0, shape=[n_output], dtype=tf.float32),
                            trainable=True, name='biases')
            conv = tf.nn.conv2d(_inp, w, strides=[1, stride, stride, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, b)
            conv = tf.nn.elu(conv)
        return conv, w

    @staticmethod
    def deconv_layer(_inp, w, shape, name="Deconvolution_layer", stride=2):
        with tf.name_scope(name):
            b = tf.Variable(tf.zeros([w.get_shape().as_list()[2]]))
            output_shape = tf.pack([tf.shape(_inp)[0], shape[1], shape[2], shape[3]])
            deconv = tf.nn.conv2d_transpose(_inp, w, output_shape=output_shape, strides=[1, stride, stride, 1],
                                            padding="SAME")
            deconv = tf.nn.bias_add(deconv, b)
            deconv = tf.nn.elu(deconv)
        return deconv

    def train(self, data, batch_size, training_iters, display_step=2000):
        assert training_iters % batch_size == 0

        with tf.Session() as sess:
            self.merged = tf.merge_all_summaries()
            self.train_writer = tf.train.SummaryWriter(os.path.join(self.tb_path, "train"), sess.graph)
            sess.run(tf.initialize_all_variables())

            for step in range(int(training_iters / batch_size)):
                batch_x, batch_y = data.train.next_batch(batch_size)
                self.feed_dict = {self.x: batch_x}
                sess.run(self.optimizer, feed_dict=self.feed_dict)
                if step % display_step == 0 and self.observers is not None:
                    self.observers.notify(self, sess=sess, data=data, step=step)

            self.saver.save(sess, self.model_path)

    def run_function(self, model_path, fn, feed_dict=None):
        if feed_dict is None:
            feed_dict = dict()
        with tf.Session() as sess:
            self.saver.restore(sess, model_path)
            return sess.run(fn, feed_dict=feed_dict)

    def test(self, data, batch_size, model_path):
        test_data = data.test.images[:batch_size]
        test_label = data.test.labels[:batch_size]
        feed_dict = {self.x: test_data, self.y: test_label, self.keep_prob: 1}
        test_acc = self.run_function(model_path, self.accuracy, feed_dict)
        return test_acc

    def plot_filter(self, weights):
        # batch_x, batch_y = mnist.train.next_batch(10)
        # one_image = batch_x[9]
        # fd = {self.x: np.reshape(one_image, [1, 784], order='F'), self.keep_prob: 1.0}
        # units = self.run_function(self.model_path, layer, fd)

        weights = self.run_function(self.model_path, tf.reduce_sum(weights, 2))
        filters = weights.shape[2]
        print("n_filters:", filters)
        plt.figure(1, figsize=(20, 20))

        for i in range(0, filters):
            plt.subplot(int(np.sqrt(filters)), int(np.sqrt(filters)), i + 1)
            plt.title('Filter ' + str(i))
            plt.imshow(weights[:, :, i], interpolation="nearest", cmap="gray")
            frame = plt.gca()
            frame.axes.get_xaxis().set_ticks([])
            frame.axes.get_yaxis().set_ticks([])
        plt.show()
