
import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs, Weights, biases

# def create_network(num_neurons, type_neurons, inputs, in_size):
#     '''
#
#     :param num_neurons: list of number of neurons for each layer
#     :param type_neurons: type of neurons for each layer
#     :return:
#     '''
