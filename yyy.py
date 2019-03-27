import tensorflow as tf
import numpy as np

x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

b = tf.Variable(0.)
k = tf.Variable(0.)

y = k*x_data+b

#二次代价函数
loss = tf.reduce_mean(tf.square(y_data-y)) # 真实值减去预测值然后平方 然后取均值
#定义一个梯度下降发 来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2) #学习率 0.2
#最小化代价函数上面的loss
train = optimizer.minimize(loss)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for setp in range(221):
        sess.run(train)
        if setp%20 == 0:
            print(setp,sess.run([k,b]))

