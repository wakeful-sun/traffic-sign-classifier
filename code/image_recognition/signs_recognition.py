import sys
sys.path.append("../")

import tensorflow as tf
from traffic_signs_data import TrafficSignNames
from nn_model_trainer import NnModelTrainer
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import operator

class SignReader:

    def __init__(self):
        self.images_shape = [32, 32, 3]
        self.model_path = "./model/model.ckpt"

        #tf.reset_default_graph()
        self.s_names = TrafficSignNames("../../traffic-signs-data/signnames.csv").sign_names
        self.trainer = NnModelTrainer(self.images_shape, len(self.s_names))
        self.saver = tf.train.Saver()

    @property
    def sign_names(self):
        return self.s_names

    def test(self, input_image):

        for i, dimension in enumerate(self.images_shape):
            if input_image.shape[i] != dimension:
                err_msg = "Input image shape expected to be {}, but was {}".format(self.images_shape, input_image.shape)
                raise Exception(err_msg)

        labels = list(self.s_names.keys())
        images = [input_image]

        with tf.Session() as session:
            self.saver.restore(session, self.model_path)
            feed_dict = self.trainer.input.create_feed(labels, images)

            output = session.run(tf.nn.softmax(self.trainer.logits), feed_dict=feed_dict)
            return output[0]


def load_image(file_path):
    current_directory = os.path.dirname(__file__)
    image_path = os.path.join(current_directory, file_path)
    return mpimg.imread(image_path)


raw_image = load_image("./test_signs/00001_00025.ppm")
resized_image = cv2.resize(raw_image, (32, 32))

reader = SignReader()
nn_output = reader.test(resized_image)

sign_names_with_probabilities = [(s_name, nn_output[key]) for key, s_name in reader.sign_names.items()]

top_items = sorted(sign_names_with_probabilities, key=lambda x: x[1], reverse=True)[:5]

probabilities = ["{}: {:.3f}".format(name, prob*100) for name, prob in top_items]
print("\n".join(probabilities))


def plot_image(image, caption="no caption"):
    fig, ax = plt.subplots()
    fig.suptitle(caption, fontweight="bold")
    plt.imshow(image)
    plt.show()


plot_image(raw_image, probabilities[0])
