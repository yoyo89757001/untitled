
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf


def sinplot(flip=1):
    x = np.linspace(0,14,100)
    for i in range(1,7):
        plt.plot(x,np.sin(x+i * 5) * (7-i *flip ))

    plt.show()

# sns.set_style("whitegrid")
# sinplot()

ss=tf.Session


sns.set()
# uniform_data = np.random.rand(3,3)
# heatmap = sns.heatmap(uniform_data)
# print(uniform_data)
# plt.show()

fiights = sns.load_dataset("flights")
print(fiights.head())
fiights=fiights.pivot("month","year","passengers")
print(fiights.head())
sss = sns.heatmap(fiights,cmap="Blues")
plt.show()

