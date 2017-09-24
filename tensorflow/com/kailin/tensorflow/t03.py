import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from com.kailin.api_image import api_image
from com.kailin.tensorflow.api_tensorflow import api_tensorflow as api_tf
from time import time

mnist = input_data.read_data_sets("mnist_data/", one_hot=True)
# print("train", mnist.train.num_examples)
# print("validation", mnist.validation.num_examples)
# print("test", mnist.test.num_examples)
# api_image.showImageLabelPredictionHot(mnist.train.images, mnist.train.labels, [], 0)
# api_image.showImageLabelPredictionHot(mnist.validation.images, mnist.validation.labels, [], 0)
# api_image.showImageLabelPredictionHot(mnist.test.images, mnist.test.labels, [], 0)

# 建立神經網路
x = tf.placeholder("float", [None, 784])
h1 = api_tf.layer(dim_in=784, dim_out=256, inputs=x, activation=tf.nn.relu)
h2 = api_tf.layer(dim_in=256, dim_out=64, inputs=h1)
y_predict = api_tf.layer(dim_in=64, dim_out=10, inputs=h2)
# 定義訓練方式
y_label = tf.placeholder("float", [None, 10])
loss_function = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_predict, labels=y_label))
optimizer_function = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)
# 評估模型準確率
correct_predcit = tf.equal(tf.argmax(y_label, 1), tf.argmax(y_predict, 1))
# 將預測結果進行平均
accuracy = tf.reduce_mean(tf.cast(correct_predcit, "float"))

trainEpochs = 15
batchSize = 100
loss_list = []
epoch_list = []
accuracy_list = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(trainEpochs):
        startTime = time()
        # 分批訓練
        for i in range(int(mnist.train.num_examples / batchSize)):
            batch_x, batch_y = mnist.train.next_batch(batchSize)
            sess.run(optimizer_function, feed_dict={x: batch_x, y_label: batch_y})
        loss, acc = sess.run([loss_function, accuracy],
                             feed_dict={x: mnist.validation.images, y_label: mnist.validation.labels})
        epoch_list.append(epoch)
        loss_list.append(loss)
        accuracy_list.append(acc)
        print("Train Epoch=", "%02d" % (epoch + 1),
              "Train Time=", time() - startTime,
              "Loss=", "{:.9f}".format(loss),
              "Accuracy=", acc)
    print("Accuracy", sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels}))
    prediction = sess.run(tf.argmax(y_predict, 1), feed_dict={x: mnist.test.images})
    api_image.showImageLabelPredictionHot(mnist.test.images, mnist.test.labels, prediction, 0)
