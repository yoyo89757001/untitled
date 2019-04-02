import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
import pandas.plotting as pt
import mglearn
import sklearn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



iris_dataset = load_iris()

# 拆分数据train_test_split 函数利用伪随机数生成器将数据集打乱
#为了确保多次运行同一函数能够得到相同的输出，我们利用 random_state
# 参数指定了随机 数生成器的种子。这样函数输出就是固定不变的，所以这行代码的输出始终相同
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# 利用X_train中的数据创建DataFrame
# 利用iris_dataset.feature_names中的字符串对数据列进行标记
#iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# 利用DataFrame创建散点图矩阵，按y_train着色
#grr = pt.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',hist_kwds={'bins': 20}, s=160, alpha=.8, cmap=mglearn.cm3)
#plt.show()

knn = KNeighborsClassifier(n_neighbors=1)#临近个数
#基于训练集来构建模型
knn.fit(X_train, y_train)

#预测新样本 新建一个样本
X_new = np.array([[5, 2.9, 1, 0.2]])
#调用 knn 对象的 predict 方法来进行预测
prediction = knn.predict(X_new)
print("预测值: {}".format(prediction))
print("对应的标签名: {}".format(iris_dataset['target_names'][prediction]))
#预测测试数据集，来衡量模型的精度
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


# 生成数据集 列子数据集
X, y = mglearn.datasets.make_forge()
# 数据集绘图
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape: {}".format(y.shape))
plt.show()
