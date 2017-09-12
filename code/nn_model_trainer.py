import tensorflow as tf
from nn_model_factory import *


class NnModelTrainer:

    def __init__(self, im_shape, n_classes, learning_rate=0.0001):

        with tf.name_scope("traffic_signs_input"):
            self.t_input = Placeholders(im_shape)
            t_one_hot_labels = tf.one_hot(self.input.labels_tensor, n_classes, name="one_hot_labels")

        with tf.name_scope("input"):
            tf.summary.image("images", self.input.images_tensor, 5)

        t_preprocessed_images = tf.image.convert_image_dtype(self.input.images_tensor, dtype=tf.float32)
        logits = NnModelFactory().create(t_preprocessed_images, self.input.keep_prob_tensor, n_classes)

        with tf.name_scope("cross_entropy"):
            self.t_softmax = tf.nn.softmax_cross_entropy_with_logits(labels=t_one_hot_labels, logits=logits)
            self.t_cross_entropy = tf.reduce_mean(self.t_softmax)

        with tf.name_scope("accuracy"):
            self.prediction_is_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(t_one_hot_labels, 1))
            self.t_accuracy = tf.reduce_mean(tf.cast(self.prediction_is_correct, tf.float32))

        with tf.name_scope("loss_optimizer"):
            self.t_train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.t_cross_entropy)

    @property
    def input(self):
        return self.t_input

    @property
    def accuracy(self):
        return self.t_accuracy

    @property
    def train_step(self):
        return self.t_train_step

    @property
    def cross_entropy(self):
        return self.t_cross_entropy

    @property
    def prediction(self):
        return self.prediction_is_correct

    @property
    def softmax(self):
        return self.t_softmax


class Placeholders:

    def __init__(self, im_shape):
        t_images_shape = list(im_shape)
        t_images_shape.insert(0, None)
        self.t_images = tf.placeholder(tf.uint8, t_images_shape, name="input_images")
        self.t_labels = tf.placeholder(tf.int32, (None,), name="input_labels")
        self.t_keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    @property
    def labels_tensor(self):
        return self.t_labels

    @property
    def images_tensor(self):
        return self.t_images

    @property
    def keep_prob_tensor(self):
        return self.t_keep_prob

    def create_feed(self, labels_data, images_data, keep_prob=1.0):
        return {
            self.t_labels: labels_data,
            self.t_images: images_data,
            self.t_keep_prob: keep_prob
        }
