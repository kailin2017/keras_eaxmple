import tensorflow as tf


class api_tensorflow:
    def layer(dim_out, dim_in, inputs, activation=None):
        w = tf.Variable(tf.random_normal([dim_in, dim_out]))
        b = tf.Variable(tf.random_normal([1, dim_out]))
        xwb = tf.matmul(inputs, w) + b
        if activation is None:
            output = xwb
        else:
            output = activation(xwb)
        return output


