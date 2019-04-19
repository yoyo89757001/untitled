import tensorflow as tf
import turtle
import pickle
import seaborn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d

a= tf.placeholder(tf.float32,shape=(1,4))
w = tf.Variable(tf.random_normal(shape=(1,4),mean=100,stddev=0.35))
b = tf.Variable(tf.zeros([4]),name='b')
y = w * a +b
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    fff= y.eval(feed_dict={a:[[1,2,3,4]]})
    #print(fff)
    optimizer = tf.train.AdadeltaOptimizer()
   # print(tf.device('/cpu:12'))
    tf.get_variable('dsd',(2,3),initializer=tf.constant_initializer())

da =(2,3,4,5)

seaborn.set(context='notebook',style='whitegrid',palette='dark')
da1=pd.read_csv('food_info2.csv')
seaborn.lmplot('Shrt','u',da1,height=6,fit_reg=False)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.scatter3D(da1.get('y'),da1.get('Shrt'),da1.get('u'),c=da1.get('u'),cmap='Greens')
print(da1.get('y'))
plt.show()


#归一化方法
def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())
#归一化后的数据
df = normalize_feature(da1)
ones = pd.DataFrame({'ones':np.ones(len(df))}) # 构造第跟df行数相同的数据为1的DataFrame数据
#合并到df 中
df = pd.concat([ones,df],axis=1) #0是行，1是列

print(df)