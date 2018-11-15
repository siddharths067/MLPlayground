import tensorflow as tf 
import numpy as np 

raw_data = np.random.normal(10, 1, 100)
const_val = np.mean(raw_data, dtype=np.float32)
true_avg = tf.constant(const_val)

alpha = tf.constant(0.05)
curr_value = tf.placeholder(tf.float32)
prev_avg = tf.Variable(0.)
update_avg = alpha * curr_value + (1 - alpha) * prev_avg
curr_error = tf.abs(update_avg - true_avg)

avg_hist = tf.summary.scalar("running average", update_avg)
value_hist = tf.summary.scalar("incoming values", curr_value)
error_hist = tf.summary.scalar("Error History", curr_error)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./newlog1")

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #sess.add_graph(sess.graph)
    for i in range(len(raw_data)):
        summary_str, curr_avg, curr_err = sess.run([merged, update_avg, curr_error], feed_dict={curr_value: raw_data[i]})
        sess.run(tf.assign(prev_avg, curr_avg))
        print(raw_data[i], curr_avg, curr_err)
        writer.add_summary(summary_str, i)
        