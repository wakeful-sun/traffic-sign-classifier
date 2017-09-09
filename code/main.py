import tensorflow as tf
from nn_model_factory import *
from time_tracker import *
from input_normalizer import *
from traffic_signs_data import TrafficSignsData


data = TrafficSignsData()
data.print_info()

input_normalizer = InputNormalizer()
n_labels, n_data = data.train.apply_map(input_normalizer.normalize_by_amount)
# n_labels, n_data = y_train, X_train

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
learning_rate = 1e-4
with tf.name_scope("loss_optimizer"):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# accuracy calculation
with tf.name_scope("acuracy"):
    prediction_is_correct = tf.equal(tf.arg_max(model, 1), tf.argmax(one_hot_y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction_is_correct, tf.float32))

tf.summary.scalar("cross_entropy_scl", cross_entropy)
tf.summary.scalar("training_acuracy", accuracy)

summarize_all = tf.summary.merge_all()


num_steps = len(n_data)
timer = TimeTracker(num_steps)

# x_data = input_normalizer.normalize_color(n_data)
# x_data = input_normalizer.image_to_float(n_data)

EPOCHS = 25
BATCH_SIZE = 20

saver = tf.train.Saver()

session.run(tf.global_variables_initializer())
timer.log("Global variable initialization")

# ----
tb_log_path = "./tb_logs/{}-ep_{}-batch_rate-{}_all_data/".format(EPOCHS, BATCH_SIZE, learning_rate)
model_path = tb_log_path + "model/"
tb_writer = tf.summary.FileWriter(tb_log_path, session.graph)
# ----
index = 0

for epoch in range(EPOCHS):
    print("Epoch: ", epoch)
    data.train.shaffle()
    n_labels, n_data = data.train.labels, data.train.features
    #n_labels, n_data = shuffle(n_labels, n_data)
    #n_labels, n_data = input_normalizer.normalize_by_amount(y_train, X_train)

    for offset in range(0, num_steps, BATCH_SIZE):
        x_batch, y_batch = n_data[offset:offset + BATCH_SIZE], n_labels[offset:offset + BATCH_SIZE]
        _, summary = session.run([train_step, summarize_all],feed_dict={im_input: x_batch, y_: y_batch, keep_prob: 0.5})

        train_accuracy = session.run(accuracy, feed_dict={im_input: x_batch, y_: y_batch, keep_prob: 1.0})
        if offset%1000 == 0:
            print("training accuracy {:.3f}".format(train_accuracy*100))

        index += len(x_batch)
        # write summary to log
        tb_writer.add_summary(summary, index)


saver.save(session, model_path)
timer.log("Session save")

timer.output_summary()
session.close()
