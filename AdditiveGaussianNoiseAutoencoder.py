# 去噪自编码器
import tensorflow as tf


class AdditiveGaussianNoiseAutoencoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        self.x = tf.placeholder(tf.float32,[None,self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(
                        self.x + scale * tf.random_normal((n_input,)),
                        self.weights['W1']),
                        self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(
                        self.hidden,
                        self.weights['W2']),
                        self.weights['b2'])

        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
                            self.reconstruction,self.x),2.0))

    def _initialize_weights(self):
        all_weights = dict()
