import tensorflow as tf
from nn_model_factory import *
from time_tracker import *
from input_normalizer import *
from traffic_signs_data import TrafficSignsData


learning_rate = 1e-4
EPOCHS = 66
BATCH_SIZE = 200
drop = 0.17
tb_log_path = "../tb_logs/E{}_B{}_R{}_D{}_all-data/".format(EPOCHS, BATCH_SIZE, learning_rate, drop)
model_path = tb_log_path + "model.ckpt"


data = TrafficSignsData()
data.print_info()

session = tf.InteractiveSession()

def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
    with tf.name_scope("stddev"):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar("stddev", stddev)
    tf.summary.scalar("max", tf.reduce_max(var))
    tf.summary.scalar("min", tf.reduce_min(var))
    tf.summary.histogram("histogram", var)


with tf.name_scope("traffic_signs_input"):
    im_input = tf.placeholder(tf.uint8, [None, 32, 32, 3], name="im_input")
    y_ = tf.placeholder(tf.int32, (None,), name="y_")
    one_hot_y = tf.one_hot(y_, 42, name="one_hot_y")
    x_ = tf.image.convert_image_dtype(im_input, dtype=tf.float32)

with tf.name_scope("input"):
    tf.summary.image("images", im_input, 5)

keep_prob = tf.placeholder(tf.float32, name="keep_prob")

model = NnModelFactory().create(x_, keep_prob)

# loss measurement
with tf.name_scope("cross_entropy"):
    softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=model)
    cross_entropy = tf.reduce_mean(softmax_cross_entropy)

# I'm using Adam optimizer instead of gradient descent
# Adam is the variant of gradient descent optimizer, that varies step size to prevent
# overshooting a good solution and becoming unstable
with tf.name_scope("loss_optimizer"):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# accuracy calculation
with tf.name_scope("acuracy"):
    prediction_is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(one_hot_y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction_is_correct, tf.float32))

tf.summary.scalar("cross_entropy_scl", cross_entropy)
tf.summary.scalar("training_acuracy", accuracy)

summarize_all = tf.summary.merge_all()


timer = TimeTracker(data.train.length)
saver = tf.train.Saver()

session.run(tf.global_variables_initializer())
timer.log("Global variable initialization")

tb_writer = tf.summary.FileWriter(tb_log_path, session.graph)

input_normalizer = InputNormalizer()


def get_feed_dict(labels, images, dropout):
    return {y_: labels, im_input: images, keep_prob: dropout}


def evaluate_accuracy(dataset):
    total_accuracy = 0
    while dataset.move_next(BATCH_SIZE):
        labels, images = dataset.current
        calculated_accuracy = session.run(accuracy, feed_dict=get_feed_dict(labels, images, 1.0))
        total_accuracy += (calculated_accuracy * len(images))

    return total_accuracy / dataset.length


index = 0
accuracy_calc_period = 10000
threshold = accuracy_calc_period

for epoch in range(EPOCHS):
    print("Epoch: ", epoch)
    #data.train.apply_map(input_normalizer.normalize_by_amount)
    data.train.shuffle()

    while data.train.move_next(BATCH_SIZE):
        labels_batch, images_batch = data.train.current
        _, summary = session.run([train_step, summarize_all], feed_dict=get_feed_dict(labels_batch, images_batch, drop))

        #train_accuracy = session.run(accuracy, feed_dict=get_feed_dict(labels_batch, images_batch, 1.0))
        if index/threshold > 1:
            threshold += accuracy_calc_period
            validation_accuracy = evaluate_accuracy(data.validation)
            print("validation accuracy {:.3f}".format(validation_accuracy * 100))

        index += len(images_batch)
        # write summary to log
        tb_writer.add_summary(summary, index)

validation_accuracy = evaluate_accuracy(data.validation)
print("*validation accuracy {:.3f}".format(validation_accuracy * 100))

saver.save(session, model_path)
timer.log("Session save")

timer.output_summary()
session.close()
