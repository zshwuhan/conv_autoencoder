import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import tensorflow as tf
import time

from tensorflow.contrib.layers.python import layers
from src.model.utils import *
from src.data.data import *


class DCGAAE:
    def __init__(self, hyperparameters, observers=None, corrupt=None):
        self.observers = observers

        self.hp = hyperparameters
        self.corrupt = corrupt
        self.is_training = True

        assert len(self.hp.autoenc_filter_depths) == len(self.hp.autoenc_filter_sizes)

        # self.discr_bn = [BatchNorm() for _ in range(len(self.hp.discr_filter_depths))]
        # self.gen_bn = [BatchNorm() for _ in range(len(self.hp.autoenc_filter_depths))]

        with tf.name_scope("X"):
            self.x = tf.placeholder(tf.float32,
                                    [None, self.hp.input_shape[0], self.hp.input_shape[1], self.hp.input_shape[2]],
                                    name="X")

        with tf.name_scope("Reshaping_and_transposing"):
            if self.corrupt is not None:
                self.input = self.corrupt(self.x)
            else:
                self.input = self.x

        self.shapes = []  # used in autoencoder
        curr_input = self.input
        self.gen_output, self.z = self.autoencoder(curr_input)

        with tf.name_scope("Autoencoder_only_output"):
            self.z_input = tf.placeholder(tf.float32, [None, 1, 1, self.hp.autoenc_filter_depths[-1]])
            with tf.variable_scope("Autoencoder", reuse=True):
                self.decoded_image = self.decoder(self.z_input)

        self.discr_on_gen, self.discr_on_gen_lin = self.discriminator(self.gen_output)

        self.discr_on_real, self.discr_on_real_lin = self.discriminator(self.x, reuse=True)

        # Real = 1, Fake = 0
        with tf.name_scope("Generator_loss"):
            self.reconstr_loss = tf.reduce_mean(tf.square(self.gen_output - self.x),
                                                name="Reconstruction_loss")
            # training the generator on generated images
            self.tgen_genimages = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.discr_on_gen_lin,
                                                                                         tf.ones_like(
                                                                                             self.discr_on_gen)))

            # total loss for the generator training
            self.gen_loss = self.tgen_genimages
            tf.scalar_summary("generator_loss_TOTAL", self.gen_loss)
            tf.scalar_summary("generator_loss_reconstruction", self.reconstr_loss)
            tf.scalar_summary("generator_loss_on_generated", self.tgen_genimages)

        with tf.name_scope("Discriminator_loss"):
            # training the discriminator on real mages
            self.tdiscr_realimages = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(self.discr_on_real_lin, tf.ones_like(self.discr_on_real)))
            # training the discriminator on generated images, we want the discriminator to recognize the generated
            # images
            self.tdiscr_genimages = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(self.discr_on_gen_lin, tf.zeros_like(self.discr_on_gen)))
            # total loss for the discriminator training
            self.discr_loss = self.tdiscr_genimages + self.tdiscr_realimages
            tf.scalar_summary("discrimination_loss_TOTAL", self.discr_loss)
            tf.scalar_summary("discrimination_loss_on_generated", self.tdiscr_genimages)
            tf.scalar_summary("discrimination_loss_on_real", self.tdiscr_realimages)

        with tf.name_scope("Optimizers"):
            d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Discriminator")
            g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Autoencoder")
            adam = tf.train.AdamOptimizer
            self.gen_optim = adam(learning_rate=self.hp.lr).minimize(self.gen_loss, var_list=g_vars)
            self.discr_optim = adam(learning_rate=self.hp.lr).minimize(self.discr_loss, var_list=d_vars)
            self.reconstr_optim = adam(learning_rate=self.hp.lr).minimize(self.reconstr_loss, var_list=g_vars)

        self.imgsum_real = tf.image_summary("Real_image", self.x)
        self.imgsum_gen = tf.image_summary("Generated_image", self.gen_output)
        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

        self.train_writer = None
        self.test_writer = None
        self.merged = tf.merge_all_summaries()

        self.feed_dict = dict()
        self.timestamp = time.strftime("%Y_%B_%d__%H:%M", time.localtime())
        self.tb_path = os.path.join(DataPath.base, "models", self.timestamp)

        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        self.model_path = self.tb_path + ".tmp"

    def discriminator(self, discr_input, reuse=None):
        d0 = discr_input
        f = self.hp.discr_filter_depths
        s = self.hp.discr_filter_sizes
        with tf.variable_scope("Discriminator", reuse=reuse):
            d1 = layers.conv2d(d0, f[0], s[0], stride=2, activation_fn=lrelu)
            d2 = layers.conv2d(d1, f[1], s[1], stride=2, activation_fn=lrelu, normalizer_fn=layers.batch_norm)
            d3 = layers.conv2d(d2, f[2], s[2], stride=2, activation_fn=lrelu, normalizer_fn=layers.batch_norm)
            d4 = layers.conv2d(d3, f[3], s[3], stride=2, activation_fn=lrelu, normalizer_fn=layers.batch_norm)

            layer_size = int((64 / 2 ** 4) ** 2 * f[-1])  # a better way of calculating size of the last layer?
            d4 = tf.reshape(d4, shape=[self.hp.batch_size, layer_size])
            fcn = layers.fully_connected(d4, 1, normalizer_fn=layers.batch_norm)
            output = tf.nn.sigmoid(fcn)

        return output, fcn

    def discriminator_old(self, discr_input, reuse=None):
        curr_input = discr_input
        with tf.variable_scope("Discriminator", reuse=reuse):
            for i, (filter_size, n_output) in enumerate(zip(self.hp.discr_filter_sizes, self.hp.discr_filter_depths)):
                batch_norm = self.discr_bn
                activation = lrelu
                if i == 0:
                    batch_norm = None

                output, w = self.conv_layer(curr_input, filter_size, n_output, i, batch_norm=batch_norm,
                                            activation=activation)
                curr_input = output

            with tf.variable_scope("Linear"):
                curr_input_shape = curr_input.get_shape().as_list()
                n_input = (self.hp.input_shape[1] / 2 ** (len(self.hp.discr_filter_depths))) ** 2 * curr_input_shape[3]
                w = tf.get_variable("Matrix", [n_input, 1], tf.float32, tf.contrib.layers.xavier_initializer())
                b = tf.get_variable("Bias", [1], initializer=tf.constant_initializer())
                curr_input = tf.reshape(curr_input, shape=[self.hp.batch_size, -1])
                lin_output = tf.matmul(curr_input, w) + b
                output = tf.nn.sigmoid(lin_output)
            return output, lin_output

    def autoencoder(self, curr_input):
        with tf.variable_scope("Autoencoder"):
            z = self.encoder(curr_input)
            output = self.decoder(z)
        return output, z

    def decoder_old(self, z):
        curr_input = z
        for ind, shape in reversed(list(enumerate(self.shapes))):
            activation = tf.nn.relu
            batch_norm = self.gen_bn
            if ind == 0:
                activation = tf.tanh
                batch_norm = None

            output = self.deconv_layer(curr_input, shape, ind, activation=activation, batch_norm=batch_norm)

            curr_input = output

        return curr_input

    def decoder(self, z):
        h6 = z
        f = self.hp.autoenc_filter_depths
        s = self.hp.autoenc_filter_sizes
        activation = tf.nn.relu
        with tf.name_scope("Decoder"):
            h5 = layers.conv2d_transpose(h6, f[5], s[6], stride=2, activation_fn=None, biases_initializer=None,
                                         scope="Layer_5", reuse=True)
            h5 = activation(layers.batch_norm(h5))

            h4 = layers.conv2d_transpose(h5, f[4], s[5], stride=2, activation_fn=None, biases_initializer=None,
                                         scope="Layer_4", reuse=True)
            h4 = activation(layers.batch_norm(h4))

            h3 = layers.conv2d_transpose(h4, f[3], s[4], stride=2, activation_fn=None, biases_initializer=None,
                                         scope="Layer_3", reuse=True)
            h3 = activation(layers.batch_norm(h3))

            h2 = layers.conv2d_transpose(h3, f[2], s[3], stride=2, activation_fn=None, biases_initializer=None,
                                         scope="Layer_2", reuse=True)
            h2 = activation(layers.batch_norm(h2))

            h1 = layers.conv2d_transpose(h2, f[1], s[2], stride=2, activation_fn=None, biases_initializer=None,
                                         scope="Layer_1", reuse=True)
            h1 = activation(layers.batch_norm(h1))

            h0 = layers.conv2d_transpose(h1, f[0], s[1], stride=2, activation_fn=tf.nn.tanh)

        return h0

    def encoder(self, image):
        h0 = image
        f = self.hp.autoenc_filter_depths
        s = self.hp.autoenc_filter_sizes
        with tf.name_scope("Encoder"):
            h1 = layers.conv2d(h0, f[1], s[1], stride=2, normalizer_fn=layers.batch_norm, scope="Layer_0")
            h2 = layers.conv2d(h1, f[2], s[2], stride=2, normalizer_fn=layers.batch_norm, scope="Layer_1")
            h3 = layers.conv2d(h2, f[3], s[3], stride=2, normalizer_fn=layers.batch_norm, scope="Layer_2")
            h4 = layers.conv2d(h3, f[4], s[4], stride=2, normalizer_fn=layers.batch_norm, scope="Layer_3")
            h5 = layers.conv2d(h4, f[5], s[5], stride=2, normalizer_fn=layers.batch_norm, scope="Layer_4")
            h6 = layers.conv2d(h5, f[6], s[6], stride=2, normalizer_fn=layers.batch_norm, scope="Layer_5")

        return h6

    def encoder_old(self, image):
        curr_input = image
        for i, (filter_size, n_output) in enumerate(zip(self.autoenc_filter_sizes, self.autoenc_filter_depths)):

            batch_norm = self.gen_bn
            if i == len(self.autoenc_filter_depths) - 1:
                batch_norm = None

            output, w = self.conv_layer(curr_input, filter_size, n_output, i, batch_norm=batch_norm)

            if len(self.shapes) < len(self.autoenc_filter_depths):
                self.shapes.append(curr_input.get_shape().as_list())
            curr_input = output

        return curr_input

    @staticmethod
    def conv_layer(_inp, filter_size, n_output, layer_index, stride=2, activation=tf.nn.relu,
                   batch_norm=None):
        layer_name = "Layer" + str(layer_index)
        with tf.variable_scope(layer_name):
            n_input = _inp.get_shape().as_list()[3]
            w = tf.get_variable("weights", [filter_size, filter_size, n_input, n_output],
                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            # b = tf.Variable(tf.constant(0.0, shape=[n_output], dtype=tf.float32),
            #                 trainable=True, name='biases')
            b = tf.get_variable('biases', [n_output], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(_inp, w, strides=[1, stride, stride, 1], padding='SAME')
            conv = tf.nn.bias_add(conv, b)
            if batch_norm is not None:
                conv = batch_norm[layer_index](conv)
            conv = activation(conv)
        return conv, w

    @staticmethod
    def deconv_layer(_inp, shape, layer_index, stride=2, activation=tf.nn.relu, batch_norm=None, w=None):
        layer_name = "Layer" + str(layer_index)
        with tf.variable_scope(layer_name, reuse=True):
            if w is None:
                w = tf.get_variable("weights")
            b = tf.Variable(tf.zeros([w.get_shape().as_list()[2]]), name="biass")
            output_shape = tf.pack([tf.shape(_inp)[0], shape[1], shape[2], shape[3]])
            deconv = tf.nn.conv2d_transpose(_inp, w, output_shape=output_shape, strides=[1, stride, stride, 1],
                                            padding="SAME")
            deconv = tf.nn.bias_add(deconv, b)
        # dirty hack by changing the variable_scope? or not? you decide
        layer_index_next = layer_index - 1
        layer_name_next = "Layer" + str(layer_index_next)
        with tf.variable_scope(layer_name_next, reuse=True):
            if batch_norm is not None:
                deconv = batch_norm[layer_index_next](deconv)
            deconv = activation(deconv)
        return deconv

    def train(self, data, display_step=20, save_step=500, old_model_path=None):
        assert self.hp.training_iters % self.hp.batch_size == 0

        with tf.Session() as sess:
            self.train_writer = tf.train.SummaryWriter(os.path.join(self.tb_path, "train"), sess.graph)
            self.test_writer = tf.train.SummaryWriter(os.path.join(self.tb_path, "test"), sess.graph)
            if old_model_path is not None:
                self.saver.restore(sess, old_model_path)
                print("Old model restored!")
            else:
                sess.run(tf.initialize_all_variables())

            for step in range(int(self.hp.training_iters / self.hp.batch_size)):
                batch_x = data.train.next_batch(self.hp.batch_size)
                self.feed_dict = {self.x: batch_x}

                if step % display_step == 0 and self.observers is not None:
                    self.observers.notify(self, sess=sess, data=data, step=step)
                if step % save_step == 0 and step > 0:
                    self.saver.save(sess, self.model_path)

                # discr_loss, gen_loss = sess.run([self.discr_loss, self.gen_loss], feed_dict=self.feed_dict)
                # if step < 500:
                #     sess.run(self.reconstr_optim, feed_dict=self.feed_dict)
                # else:
                #     sess.run(self.gen_optim, feed_dict=self.feed_dict)
                sess.run(self.gen_optim, feed_dict=self.feed_dict)
                sess.run(self.discr_optim, feed_dict=self.feed_dict)

            self.saver.save(sess, self.model_path)

    def run_function(self, model_path, fn, feed_dict=None):
        if feed_dict is None:
            feed_dict = dict()
        with tf.Session() as sess:
            self.saver.restore(sess, model_path)
            return sess.run(fn, feed_dict=feed_dict)

    def get_feature_representation(self, input_image, model_path=None):
        return self.run_function(model_path, self.z, feed_dict={self.x: input_image})[0, 0, 0, :]
