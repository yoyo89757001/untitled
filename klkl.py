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


#用于回归的 k 近邻算法在 scikit-learn 的 KNeighborsRegressor 类中实现。其用法与 KNeighborsClassifier 类似:
from sklearn.neighbors import KNeighborsRegressor
X, y = mglearn.datasets.make_wave(n_samples=40)
# 将wave数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# 模型实例化，并将邻居个数设为3
reg = KNeighborsRegressor(n_neighbors=3) # 利用训练数据和训练目标值来拟合模型 reg.fit(X_train, y_train)
reg.fit(X_train, y_train)
#现在可以对测试集进行预测:
print("Test set predictions:\n{}".format(reg.predict(X_test)))
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))

#线性回归
from sklearn.linear_model import LinearRegression
#岭回归在 linear_model.Ridge 中实现（也可以叫正则化）
from sklearn.linear_model import Ridge
X, y = mglearn.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)
#没有用正则化的时候 训练数据表现的很好，测试数据变现的很差，过拟合了
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
#加入正则化
#ridge = Ridge().fit(X_train, y_train)
#增大 alpha 会使得系数更加趋向于 0，从而降低训练集性能， 但可能会提高泛化性能
ridge = Ridge(alpha=0.9).fit(X_train, y_train)#alpha自己调节
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
#除了 Ridge，还有一种正则化的线性回归是 Lasso
from sklearn.linear_model import Lasso
# 我们增大max_iter的值，否则模型会警告我们，说应该增大max_iter
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))
from sklearn.linear_model import ElasticNet
e = ElasticNet(alpha=0.001,max_iter=100000).fit(X_train,y_train)
print("Training set score: {:.2f}".format(e.score(X_train, y_train)))
print("Test set score: {:.2f}".format(e.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(e.coef_ != 0)))

#最常见的两种线性分类算法是 Logistic 回归(logistic regression)和
# 线性支持向量机(linear support vector machine， 线 性 SVM)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
# X, y = mglearn.datasets.make_forge()
# fig, axes = plt.subplots(1, 2, figsize=(10, 3))
# for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
#    clf = model.fit(X, y)
#    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,ax=ax, alpha=.7)
#    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
#    ax.set_title("{}".format(clf.__class__.__name__))
#    ax.set_xlabel("Feature 0")
#    ax.set_ylabel("Feature 1")
# axes[0].legend()
# plt.show()

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg100 = LogisticRegression(C=100).fit(X_train2, y_train2)
print("Training set score: {:.3f}".format(logreg100.score(X_train2, y_train2)))
print("Test set score: {:.3f}".format(logreg100.score(X_test2, y_test2)))

