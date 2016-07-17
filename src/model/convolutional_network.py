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
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


class CNN:
    def __init__(self, hyp_param):
        self.hyp_param = hyp_param
        self.x = tf.placeholder(tf.float32, [None, hyp_param["n_input"]])
        self.y = tf.placeholder(tf.float32, [None, hyp_param["n_classes"]])
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        # Store layers weight & bias
        conv1_n_filters = 32
        conv2_n_filters = 16
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

        biases = {
            'bc1': tf.Variable(tf.random_normal([conv1_n_filters])),
            'bc2': tf.Variable(tf.random_normal([conv2_n_filters])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([hyp_param["n_classes"]]))
        }

        self.conv1, self.conv2, self.fc1 = None, None, None

        self.output = self.compute_output(self.x, weights, biases)

        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.output, self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=hyp_param["learning_rate"]).minimize(self.cost)

        # Evaluate model
        self.correct_pred = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.saver = tf.train.Saver()
        self.model_path = "conv_test.tmp"

    def compute_output(self, _x, _weights, _biases):
        # Reshape input picture
        _x = tf.reshape(_x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        self.conv1 = self.conv2d(_x, _weights['wc1'], _biases['bc1'])
        # Max Pooling (down-sampling)
        self.conv1 = self.maxpool2d(self.conv1, k=2)

        # Convolution Layer
        self.conv2 = self.conv2d(self.conv1, _weights['wc2'], _biases['bc2'])
        # Max Pooling (down-sampling)
        self.conv2 = self.maxpool2d(self.conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        self.fc1 = tf.reshape(self.conv2, [-1, _weights['wd1'].get_shape().as_list()[0]])
        self.fc1 = tf.add(tf.matmul(self.fc1, _weights['wd1']), _biases['bd1'])
        self.fc1 = tf.nn.relu(self.fc1)
        # Apply Dropout
        self.fc1 = tf.nn.dropout(self.fc1, self.hyp_param["dropout"])

        # Output, class prediction
        return tf.add(tf.matmul(self.fc1, _weights['out']), _biases['out'])

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

    def train(self, data, batch_size, training_iters, display_step=10):
        assert training_iters % batch_size == 0

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())

            for step in range(int(training_iters / batch_size)):
                batch_x, batch_y = data.train.next_batch(batch_size)
                sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y,
                                                    self.keep_prob: self.hyp_param["dropout"]})
                if step % display_step == 0:
                    loss, acc = sess.run([self.cost, self.accuracy], feed_dict={self.x: batch_x,
                                                                                self.y: batch_y,
                                                                                self.keep_prob: 1.})
                    print("Iter " + str(step * batch_size) + ", Minibatch Loss= " +
                          "{:.6f}".format(loss) + ", Training Accuracy= " +
                          "{:.5f}".format(acc))
            print("Optimization Finished!")

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
        one_image = batch_x[1]
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


hyperparameters = {"learning_rate": 0.001,
                   "training_iters": 100000,
                   "batch_size": 100,
                   "n_input": 784,
                   "n_classes": 10,
                   "dropout": 0.75}

nn = CNN(hyperparameters)
nn.train(mnist, hyperparameters["batch_size"], hyperparameters["training_iters"])
test_accuracy = nn.test(mnist, 256, nn.model_path)

print("Test accuracy:", test_accuracy)
nn.plot_filter(nn.conv2)
