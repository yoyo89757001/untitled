
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd
import pandas.plotting as pt
import mglearn
import sklearn
import cv2


from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


# 生成数据集 列子数据集
# X, y = mglearn.datasets.make_forge()
# 数据集绘图
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.legend(["Class 0", "Class 1"], loc=4)
# plt.xlabel("First feature")
# plt.ylabel("Second feature")
# print("X.shape: {}".format(y.shape))
# plt.show()

X, y = sklearn.datasets.make_friedman3()
# plt.plot(X, y, 'o')
# plt.ylim(-3, 3)
# plt.xlabel("Feature")
# plt.ylabel("Target")
# plt.show()
print(X.shape)

#肿瘤数据集
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
#总数
print(cancer.data.shape)
#取后面2个n和v赋值给前面两个n个v  zip是遍历字典的其中一种方法 np.bincount是取所有相同标签的累计的和。
#因为标签只有2个 所以循环2次
for n, v in zip(cancer.target_names, np.bincount(cancer.target)):
    print(n)
    print(v)
#字典形式输出
print( {n : v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})

X, y = mglearn.datasets.make_forge()
# print("X.shape: {}".format(X.shape))
# #随机分离数据集 训练占75% 测试占25%
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# #k邻近算法
# clf = KNeighborsClassifier(n_neighbors=3)
# #利用训练集对这个分类器进行拟合。对于 KNeighborsClassifier 来说就是保存数据集，
# # 以便在预测时计算与邻居之间的距离
# clf.fit(X_train, y_train)
# print("Test set predictions: {}".format(clf.predict(X_test)))
# print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
print(axes[0])
i = -1
for n_neighbors, ax in zip([1, 3, 9], axes):

    #fit方法返回对象本身，所以我们可以将实例化和拟合放在一行代码中
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
    axes[0].legend(loc=3)

plt.show()

