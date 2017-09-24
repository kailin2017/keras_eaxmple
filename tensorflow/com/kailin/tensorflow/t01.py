import tensorflow as tf

width = tf.placeholder("int32", name="width")
height = tf.placeholder("int32", name="height")
area = tf.multiply(width, height)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(area, feed_dict={height: 7, width: 8}))
    tf.summary.merge_all()
    tf.summary.FileWriter('log/t01', sess.graph)

