import tensorflow as tf


class NnModelFactory:

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name="weight")

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name="bias")

    @staticmethod
    def _conv2d(x, w):
        return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="VALID", name="conv2d")

    @staticmethod
    def _max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID", name="max_pool")

    def create(self, x, tf_keep_prob):
        with tf.name_scope("conv1"):
            w_conv1 = self._weight_variable([5, 5, 3, 15])
            b_conv1 = self._bias_variable([15])
            conv1_wx_b = self._conv2d(x, w_conv1) + b_conv1

            conv1 = tf.nn.relu(conv1_wx_b, name="relu")
            # Convolution output is 28x28x15
            pool1 = self._max_pool_2x2(conv1)
            # Pooling output is 14x14x15

        with tf.name_scope("conv2"):
            w_conv2 = self._weight_variable([5, 5, 15, 75])
            b_conv2 = self._bias_variable([75])
            conv2_wx_b = self._conv2d(pool1, w_conv2) + b_conv2

            conv2 = tf.nn.relu(conv2_wx_b, name="relu")
            # Convolution output is 10x10x75
            pool2 = self._max_pool_2x2(conv2)
            # Pooling output is 5x5x75

        with tf.name_scope("fully_connected"):
            w_fc1 = self._weight_variable([5 * 5 * 75, 600])
            b_fc1 = self._bias_variable([600])

            pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 75], name="pool")
            fc1 = tf.nn.relu(tf.matmul(pool2_flat, w_fc1) + b_fc1, name="relu")

        # dropout
        fc1_drop = tf.nn.dropout(fc1, keep_prob=tf_keep_prob, name="dropout")

        # readout
        with tf.name_scope("read_out"):
            w_fc2 = self._weight_variable([600, 42])
            b_fc2 = self._bias_variable([42])

        # model
        return tf.add(tf.matmul(fc1_drop, w_fc2), b_fc2, name="model")