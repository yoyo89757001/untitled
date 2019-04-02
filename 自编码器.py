import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
import 手写数字识别


#定义初始化器，让权重初始化的刚好合适，避免权重过大发散失效，权重过小起不到作用
def xavier_init(fan_in,fan_out,constant=1):

    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    hight = constant * np.sqrt(6.0/(fan_in+fan_out))
    print(fan_in, fan_out,low,hight)
    return tf.random_uniform((fan_in, fan_out),minval = low ,maxval= hight , dtype=tf.float32)


ddd = 手写数字识别.Ssss.das('dded')
#das= tf.Variable(0.)
initV = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(initV)

print(ddd)