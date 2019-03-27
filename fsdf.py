import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#非线性回归
#生成200随机点
x_data = np.linspace(-0.5, 0.5, 400)[:, np.newaxis] # 样本值
#符合正太分布的比列0。02的随机值 np.random.normal
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data)+noise  #x的平方加上一个实数。弧形函数图形 ，样本y的值

#定义2个占位符
x = tf.placeholder(tf.float32,[None, 1])
y = tf.placeholder(tf.float32,[None, 1])

#定义神经网络中间层
#形状为一行十列的随机二维数组tf.random.normal([1, 10])
Weights_L1 = tf.Variable(tf.random.normal([1, 10]))#权重
biases_L1 = tf.Variable(tf.zeros([1, 10]))# 偏置值
#信号的总和
Wx_plus_L1 = tf.matmul(x,Weights_L1) + biases_L1 #矩阵x乘以矩阵Weights_L1加上偏置值
L1 = tf.nn.tanh(Wx_plus_L1) #激活函数 得到中间层

#定义神经网络输出层 输出层只有一个数值 所以10行一列
Weights_L2 = tf.Variable(tf.random.normal([10, 1]))#权重
biases_L2 = tf.Variable(tf.zeros([1, 1]))# 偏置值
#信号的总和
Wx_plus_L2 = tf.matmul(L1,Weights_L2) + biases_L2 #中间层L1乘以矩阵Weights_L2加上偏置值
prediction = tf.nn.tanh(Wx_plus_L2) #输出层结果

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction)) # 真实值减去预测值然后平方 然后取均值
#定义一个梯度下降发 来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2) #学习率 0.2
#最小化代价函数上面的loss
train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sss = sess.run(init)
    for _ in range(7000):
        sess.run(train_step,feed_dict={x:x_data,y:y_data})


    #获得预测值
    prediction_value = sess.run(prediction,feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r',lw=3)
    plt.show()
