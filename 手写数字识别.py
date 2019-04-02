import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

# 载入数据
# one_hot格式就是其它位数都是0，只有其中一个状态是1
mnist = input_data.read_data_sets('/Users/mac/Downloads/MNIST_DATA', one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少批次
n_batch = mnist.train.num_examples // batch_size
# 定义两个占位符
x = tf.placeholder(tf.float32, [None, 784])  # 代表有n张，每张有784个像素的图片
y = tf.placeholder(tf.float32, [None, 10])

# 定义一个简单的神经网络
Weighits = tf.Variable(tf.zeros([784, 10]))  # 权重值
b = tf.Variable(tf.zeros([1, 10]))  # 偏置值
prediction = tf.nn.softmax(tf.matmul(x, Weighits) + b)  # softmax函数把输出值转为概率值，方便对比

# 交叉熵方式
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))

# 二次代价函数
loss = tf.reduce_mean(tf.square(y - prediction))  # 先平方再均值
# 梯度下降法
# train_setp = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
# 梯度下降法
train_setp = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 真实标签跟预测标签对比一下，看预测函数的预测效果 arg_max函数是返回一维张量中最大值所在位置
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))  # 返回布尔型数组

# 求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 把布尔型值转换成整型,再取平均值

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # batch_xs图片像素值，batch_ys标签值,比如:2(类似y的值)
            sess.run(train_setp, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter" + str(epoch) + "准确率" + str(acc))

class Ssss(object):

    def das(fff):
        print(fff)
