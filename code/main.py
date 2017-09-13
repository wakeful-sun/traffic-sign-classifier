import tensorflow as tf
import time
import os
from nn_model_factory import *
from traffic_signs_data import TrafficSignsData
from nn_model_trainer import NnModelTrainer


learning_rate = 1e-4
EPOCHS = 90
BATCH_SIZE = 200
drop = 0.3
tb_log_path = "../tb_logs/E{}_B{}_R{}_D{}_all-data/".format(EPOCHS, BATCH_SIZE, learning_rate, drop)
model_path = tb_log_path + "model.ckpt"

data = TrafficSignsData()
data.print_info()

trainer = NnModelTrainer([32, 32, 3], data.train.n_classes, learning_rate)

tf.summary.scalar("loss", trainer.loss)
tf.summary.scalar("training_accuracy", trainer.accuracy)
summarize_all = tf.summary.merge_all()

saver = tf.train.Saver()

start_time = time.time()

with tf.Session() as session:

    session.run(tf.global_variables_initializer())
    tb_writer = tf.summary.FileWriter(tb_log_path, session.graph)

    def evaluate_accuracy(dataset):
        total_accuracy = 0
        while dataset.move_next(BATCH_SIZE):
            labels, images = dataset.current
            a_feed_dict = trainer.input.create_feed(labels, images)

            calculated_accuracy = session.run(trainer.accuracy, feed_dict=a_feed_dict)
            total_accuracy += (calculated_accuracy * len(images))

        return total_accuracy / dataset.length


    index = 0
    accuracy_calc_period = 10000
    threshold = accuracy_calc_period
    validation_accuracy = 0

    for epoch in range(EPOCHS):
        if validation_accuracy > 96:
            break

        print("Epoch: ", epoch)
        data.train.shuffle()

        while data.train.move_next(BATCH_SIZE):
            if validation_accuracy > 96:
                break

            labels_batch, images_batch = data.train.current
            t_feed_dict = trainer.input.create_feed(labels_batch, images_batch, drop)

            _, summary = session.run([trainer.train_step, summarize_all], feed_dict=t_feed_dict)

            if index/threshold > 1:
                threshold += accuracy_calc_period
                validation_accuracy = evaluate_accuracy(data.validation)
                print("validation accuracy {:.3f}".format(validation_accuracy * 100))

            index += len(images_batch)
            tb_writer.add_summary(summary, index)


    final_accuracy = evaluate_accuracy(data.validation)

    saver.save(session, model_path)

elapsed = (time.time() - start_time)/60
final_accuracy_msg = "*** final validation accuracy {:.3f}".format(final_accuracy * 100)
spent_time_msg = "Total training time for {} samples: {:.2f}m".format(data.train.length * EPOCHS, elapsed)

with open("{}_summary_{:.3f}.txt".format(tb_log_path, final_accuracy * 100), "w") as f:
    f.write(final_accuracy_msg)
    f.write("\n")
    f.write(spent_time_msg)

print(final_accuracy_msg)
print(spent_time_msg)
print("Location is: ", os.path.join(os.path.dirname(__file__), tb_log_path))
