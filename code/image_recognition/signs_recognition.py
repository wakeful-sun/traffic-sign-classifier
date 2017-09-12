import sys
sys.path.append("../")

import tensorflow as tf
from traffic_signs_data import TrafficSignNames
from nn_model_trainer import NnModelTrainer
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class SignReader:

    def __init__(self):
        self.images_shape = [32, 32, 3]
        self.model_path = "./model/model.ckpt"

        #tf.reset_default_graph()
        self.sign_names = TrafficSignNames("../../traffic-signs-data/signnames.csv").sign_names
        self.trainer = NnModelTrainer(self.images_shape, len(self.sign_names))
        self.saver = tf.train.Saver()

    def test(self, input_image):

        for i, dimension in enumerate(self.images_shape):
            if input_image.shape[i] != dimension:
                err_msg = "Input image shape expected to be {}, but was {}".format(self.images_shape, input_image.shape)
                raise Exception(err_msg)

        n_classes = len(self.sign_names)
        #labels = list(self.sign_names.keys())
        labels = [16]
        #images = [input_image] * n_classes
        images = [input_image]

        with tf.Session() as session:
            self.saver.restore(session, self.model_path)
            feed_dict = self.trainer.input.create_feed(labels, images)

            #nn_output = session.run(self.trainer.softmax, feed_dict=feed_dict)
            calculated_accuracy = session.run(self.trainer.accuracy, feed_dict=feed_dict)
            return calculated_accuracy
            #return nn_output


def load_image(file_path):
    current_directory = os.path.dirname(__file__)
    image_path = os.path.join(current_directory, file_path)
    return mpimg.imread(image_path)


img = load_image("./test_signs/00000_00010.ppm")
image = cv2.resize(img, (32,32))

reader = SignReader()
result = reader.test(image)
print(result)