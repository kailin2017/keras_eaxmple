import tensorflow as tf

tf_x = tf.Variable([[0.4, 0.2, 0.4]])
tf_w = tf.Variable([[-0.5, -0.2],
                    [-0.3, 0.4],
                    [-0.5, 0.2]])
tf_b = tf.Variable([[0.1, 0.2]])
xwb = tf.matmul(tf_x, tf_w) + tf_b
xwb_relu = tf.nn.relu(xwb)
xwb_sigmod = tf.nn.sigmoid(xwb)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("relu", sess.run(xwb_relu))
    print("sigmod", sess.run(xwb_sigmod))
    tf.summary.merge_all()
    tf.summary.FileWriter('log/t02', sess.graph)
