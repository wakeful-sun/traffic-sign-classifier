import tensorflow as tf
import sys
sys.path.append("../")

from traffic_signs_data import TrafficSignsData
from nn_model_trainer import NnModelTrainer


images_shape = [32, 32, 3]
traffic_sign_names_path = "../../signnames.csv"
data_folder_path = "../../traffic-signs-data/"
trained_model_path = "../../tb_logs/E90_B200_R0.0001_D0.3_all-data/model.ckpt"

data = TrafficSignsData(data_folder_path, traffic_sign_names_path)

n_classes = len(data.sign_names)
print("Traffic sing classes: ", n_classes)
trainer = NnModelTrainer(images_shape, n_classes)

saver = tf.train.Saver()


def evaluate_accuracy_for(test_data):
    total_accuracy = 0

    with tf.Session() as session:
        saver.restore(session, trained_model_path)

        while test_data.move_next(200):
            labels, images = test_data.current
            feed_dict = trainer.input.create_feed(labels, images)

            calculated_accuracy = session.run(trainer.accuracy, feed_dict=feed_dict)
            total_accuracy += (calculated_accuracy * len(images))

        return total_accuracy / test_data.length


v_accuracy = evaluate_accuracy_for(data.train)
print("Training accuracy is: {:.3f}%".format(v_accuracy*100))

v_accuracy = evaluate_accuracy_for(data.validation)
print("Validation accuracy is: {:.3f}%".format(v_accuracy*100))

t_accuracy = evaluate_accuracy_for(data.test)
print("Test accuracy is: {:.3f}%".format(t_accuracy*100))
