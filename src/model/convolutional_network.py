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
from src.model.downloaded_examples.DCGAN_tensorflow.ops import *


class DCGAAE:
    def __init__(self, hyp_param, filter_depths=(64, 128, 256, 512, 1024, 3192), filter_sizes=(3, 3, 3, 3, 3, 2),
                 inp_shape_size=(64, 64, 3), observers=None, corrupt=True):
        assert len(filter_depths) == len(filter_sizes)
        self.hyp_param = hyp_param

        self.observers = observers

        with tf.name_scope("X"):
            self.x_reshaped = tf.placeholder(tf.float32,
                                             [None, inp_shape_size[0], inp_shape_size[1], inp_shape_size[2]], name="X")

        with tf.name_scope("Reshaping_and_transposing"):
            self.x = tf.reshape(self.x_reshaped, shape=[-1, inp_shape_size[0], inp_shape_size[1], inp_shape_size[2]])
            if corrupt:
                curr_input = self.corrupt(self.x)
            else:
                curr_input = self.x

        self.gen_output, self.z = self.autoencoder(curr_input, filter_depths, filter_sizes)

        discr_filter_depths = (128, 256, 512, 1024)
        filter_sizes = (3, 3, 3, 3)
        self.discr_on_gen, self.discr_on_gen_lin = self.discriminator(self.gen_output,
                                                                      filter_depths=discr_filter_depths,
                                                                      filter_sizes=filter_sizes)

        self.discr_on_real, self.discr_on_real_lin = self.discriminator(self.x, filter_depths=discr_filter_depths,
                                                                        filter_sizes=filter_sizes, reuse=True)

        # Define loss and optimizer
        with tf.name_scope("Generator_loss"):
            self.sigmoid_half_change = tf.Variable(200, dtype=tf.float32)
            self.sigmoid_factor = tf.Variable(20, dtype=tf.float32)
            self.step = tf.Variable(tf.zeros([]), dtype=tf.float32)
            # self.step = tf.placeholder(tf.float32, shape=1, name="global_step")
            self.sigmoid = tf.sigmoid((self.step - self.sigmoid_half_change) / self.sigmoid_factor)

            self.reconstr_loss = tf.reduce_mean(tf.square(self.gen_output - self.x),
                                                name="Reconstruction_loss")
            # training the generator on generated images
            self.tgen_genimages = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.discr_on_gen_lin,
                                                                                         tf.ones_like(
                                                                                             self.discr_on_gen)))

            # total loss for the generator training
            self.gen_loss = self.reconstr_loss + self.tgen_genimages
            tf.scalar_summary("generator_loss_TOTAL", self.gen_loss)
            tf.scalar_summary("sigmoid_factor", self.sigmoid)
            tf.scalar_summary("generator_loss_reconstruction", self.reconstr_loss)
            tf.scalar_summary("generator_loss_on_generated", self.tgen_genimages)

        with tf.name_scope("Discriminator_loss"):
            # training the discriminator on real mages
            self.tdiscr_realimages = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(self.discr_on_real_lin, tf.ones_like(self.discr_on_real)))
            # training the discriminator on generated images, we want the discriminator to recognize the generated
            # images
            self.tdiscr_genimages = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.discr_on_gen_lin,
                                                                                           tf.zeros_like(
                                                                                               self.discr_on_gen)))
            # total loss for the discriminator training
            self.discr_loss = self.tdiscr_genimages + self.tdiscr_realimages
            tf.scalar_summary("discrimination_loss_TOTAL", self.discr_loss)
            tf.scalar_summary("discrimination_loss_on_generated", self.tdiscr_genimages)
            tf.scalar_summary("discrimination_loss_on_real", self.tdiscr_realimages)

        d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Discriminator")
        g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Autoencoder")

        with tf.name_scope("Optimizers"):
            self.gen_optim = tf.train.AdamOptimizer(learning_rate=hyp_param["learning_rate"],
                                                    beta1=hyp_param["beta1"]).minimize(self.gen_loss, var_list=g_vars)
            self.discr_optim = tf.train.AdamOptimizer(learning_rate=hyp_param["learning_rate"]).minimize(
                self.discr_loss, var_list=d_vars)

        self.imgsum_real = tf.image_summary("Real_image", self.x)
        self.imgsum_gen = tf.image_summary("Generated_image", self.gen_output)
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        self.train_writer = None
        self.merged = None

        self.feed_dict = dict()
        self.timestamp = time.strftime("%Y_%B_%d__%H:%M", time.localtime())
        self.tb_path = os.path.join(DataPath.base, "tf_logs", self.timestamp)

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        self.model_path = self.tb_path + ".tmp"

    @staticmethod
    def corrupt(x):
        """Take an input tensor and add uniform masking.

        Parameters
        ----------
        x : Tensor/Placeholder
            Input to corrupt.

        Returns
        -------
        x_corrupted : Tensor
            50 pct of values corrupted.
        """
        return tf.mul(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
                                                   minval=0,
                                                   maxval=2,
                                                   dtype=tf.int32), tf.float32))

    def discriminator(self, curr_input, filter_depths, filter_sizes, reuse=None):
        # batch normalization : deals with poor initialization helps gradient flow
        with tf.variable_scope("Discriminator", reuse=reuse):
            # self.d_bn1 = batch_norm(name='d_bn1')
            # self.d_bn2 = batch_norm(name='d_bn2')
            # self.d_bn3 = batch_norm(name='d_bn3')
            self.gf_dim = 64
            self.df_dim = 64

            h0 = lrelu(conv2d(curr_input, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(conv2d(h0, self.df_dim * 2, name='d_h1_conv'))
            h2 = lrelu(conv2d(h1, self.df_dim * 4, name='d_h2_conv'))
            h3 = lrelu(conv2d(h2, self.df_dim * 16, name='d_h3_conv'))
            h4 = linear(tf.reshape(h3, [self.hyp_param["batch_size"], -1]), 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    # def discriminator(self, curr_input, filter_depths, filter_sizes, reuse=None):
    #     assert len(filter_depths) == len(filter_sizes)
    #
    #     with tf.variable_scope("Discriminator", reuse=reuse):
    #         for i, filter_size in enumerate(filter_sizes):
    #             batch_norm_flag = True
    #             if i == 0:
    #                 batch_norm_flag = False
    #             n_input = curr_input.get_shape().as_list()[3]
    #             n_output = filter_depths[i]
    #             output, w = self.conv_layer(curr_input, filter_size, n_input, n_output,
    #                                         name="Convolution_layer" + str(i), stride=2,
    #                                         batch_norm_flag=batch_norm_flag)
    #             curr_input = output
    #
    #         with tf.variable_scope("Linear"):
    #             n_input = curr_input.get_shape().as_list()[3]
    #             w = tf.get_variable("Matrix", [n_input, 1], tf.float32,
    #                                 tf.truncated_normal_initializer(stddev=0.001))
    #             b = tf.get_variable("Bias", [1], initializer=tf.constant_initializer())
    #             curr_input = tf.reshape(curr_input, shape=[-1, filter_depths[-1]])
    #             lin_output = tf.matmul(curr_input, w) + b
    #             output = tf.nn.sigmoid(lin_output)
    #
    #         return output, lin_output

    def autoencoder(self, curr_input, filter_depths, filter_sizes):
        encoder = []
        shapes = []
        with tf.variable_scope("Autoencoder"):
            with tf.name_scope("Encoder"):
                for i, filter_size in enumerate(filter_sizes):
                    n_input = curr_input.get_shape().as_list()[3]
                    n_output = filter_depths[i]
                    activation = tf.nn.relu
                    # if i == len(filter_sizes) - 1:
                    #     activation = tf.tanh
                    output, w = self.conv_layer(curr_input, filter_size, n_input, n_output,
                                                name="Convolution_layer" + str(i), activation=activation)
                    encoder.append(w)
                    shapes.append(curr_input.get_shape().as_list())
                    curr_input = output

            with tf.name_scope("Feature_space"):
                z = curr_input
            encoder.reverse()
            shapes.reverse()
            with tf.name_scope("Decoder"):
                for i, filter_size in enumerate(shapes):
                    w = encoder[i]
                    shape = shapes[i]
                    activation = tf.nn.relu
                    batch_norm_flag = True
                    if i == len(shapes) - 1:
                        activation = tf.tanh
                        batch_norm_flag = False
                    output = self.deconv_layer(curr_input, w, shape, name="Deconvolution_layer" + str(i),
                                               activation=activation, batch_norm_flag=batch_norm_flag)
                    curr_input = output
        return curr_input, z

    @staticmethod
    def conv_layer(_inp, filter_size, n_input, n_output, name="Convolution_layer", stride=2, activation=lrelu,
                   batch_norm_flag=True):
        with tf.variable_scope(name):
            w = tf.get_variable("weights", [filter_size, filter_size, n_input, n_output],
                                initializer=tf.truncated_normal_initializer(stddev=0.001))
            b = tf.Variable(tf.constant(0.0, shape=[n_output], dtype=tf.float32),
                            trainable=True, name='biases')
            conv = tf.nn.conv2d(_inp, w, strides=[1, stride, stride, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, b)
            conv = activation(conv)
        return conv, w

    @staticmethod
    def deconv_layer(_inp, w, shape, name="Deconvolution_layer", stride=2, activation=tf.nn.relu, batch_norm_flag=True):
        with tf.name_scope(name):
            b = tf.Variable(tf.zeros([w.get_shape().as_list()[2]]), name="biass")
            # b = tf.get_variable("biases", [w.get_shape().as_list()[2]], initializer=tf.constant_initializer(0.0))
            output_shape = tf.pack([tf.shape(_inp)[0], shape[1], shape[2], shape[3]])
            deconv = tf.nn.conv2d_transpose(_inp, w, output_shape=output_shape, strides=[1, stride, stride, 1],
                                            padding="SAME")
            deconv = tf.nn.bias_add(deconv, b)
            deconv = activation(deconv)
        return deconv

    def train(self, data, batch_size, training_iters, display_step=5, old_model_path=None):
        assert training_iters % batch_size == 0

        with tf.Session() as sess:
            self.merged = tf.merge_all_summaries()
            self.train_writer = tf.train.SummaryWriter(os.path.join(self.tb_path, "train"), sess.graph)
            if not old_model_path:
                sess.run(tf.initialize_all_variables())
            else:
                self.saver.restore(sess, old_model_path)
                print("Old model restored!")

            for step in range(int(training_iters / batch_size)):
                batch_x, batch_y = data.next_batch(batch_size)
                self.feed_dict = {self.x_reshaped: batch_x, self.step: step}
                # discr_loss, gen_loss = sess.run([self.discr_loss, self.gen_loss], feed_dict=self.feed_dict)
                sess.run(self.discr_optim, feed_dict=self.feed_dict)
                sess.run(self.gen_optim, feed_dict=self.feed_dict)
                sess.run(self.gen_optim, feed_dict=self.feed_dict)

                if step % display_step == 0 and self.observers is not None:
                    self.observers.notify(self, sess=sess, data=data, step=step)

                if step % 50 == 0:
                    self.saver.save(sess, self.model_path)

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
        feed_dict = {self.x_reshaped: test_data}
        test_acc = self.run_function(model_path, feed_dict)
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
