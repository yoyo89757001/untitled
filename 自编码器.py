import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data


class AdditiveGaussianNoiseAutoencoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer=tf.train.AdamOptimizer(), scale=0.1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights

        self.x = tf.placeholder(tf.float32,[None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(
                        self.x + scale * tf.random_normal((n_input,)),
                        self.weights['w1']),
                        self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(
                        self.hidden,
                        self.weights['w2']),
                        self.weights['b2'])

        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(
                            self.reconstruction,self.x),2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input,self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden,self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost,self.optimizer),
                                  feed_dict={self.x: X, self.scale: self.training_scale})
        return cost

    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x: X, self.scale:self.training_scale})

    def transform(self, X):
        return self.sess.run(self.hidden,feed_dict={self.x: X, self.scale:self.training_scale})


    def generate(self,hidden = None):
        if hidden is None :
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction,feed_dict={self.hidden:hidden})

    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X, self.scale:self.training_scale})

   # 这里的getWeights函数作用是获取隐含_的权重w1。
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    #而getBiases函数则是获取隐含层的偏置系数b1。
    def getBiases(self):
        return self.sess.run(self.weights['b1'])

#定义初始化器，让权重初始化的刚好合适，避免权重过大发散失效，权重过小起不到作用
def xavier_init(fan_in,fan_out,constant=1):
    low = -constant * np.sqrt(6.0/(fan_in+fan_out))
    hight = constant * np.sqrt(6.0/(fan_in+fan_out))
    print(fan_in, fan_out,low,hight)
    return tf.random_uniform((fan_in, fan_out),minval = low ,maxval= hight , dtype=tf.float32)


# one_hot格式就是其它位数都是0，只有其中一个状态是1
mnist = input_data.read_data_sets('/Users/mac/Downloads/MNIST_DATA', one_hot=True)



# 先定义一个对训练、测试数据进行标准化处理的函数。标准化即让数据变成0均值，
# 且标准差为1的分布。方法就是先减去均值，再除以标准差。
# 我们直接使用 skleam.preprossing的StandardScaler这个类，
# 先在训练集上进行fit,再将这个Scaler用到 训练数据和测试数据上。
# 这里需要注意的是，必须保证训练、测试数据都使用完全相同的 Scaler,
# 这样才能保证后面模型处理数据时的一致性，这也就是为什么先在训练数据上fit 出一个共用的Scaler的原因。
def standard_scale(X_train,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

# 再定义一个获取随机block数据的函数：
# 取一个从0到len(data) - batch_size之间的 随机整数,
# 再以这个随机数作为block的起始位置，然后顺序取到一个batch size的数据。
# 需要注意的是，这属于不放回抽样，可以提高数据的利用效率。
def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

#使用之前定义的standard_scale函数对训练集、测试集进行标准化变换。
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

# 接下来定义几个常用参数，总训练样本数，
# 最大训练的轮数(epoch设为20,batch_size 设为128，并设置每隔一轮(epoch)就显示一次损失cost。
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1

# 创建一个AGN自编码器的实例，定义模型输入节点数njnput为784,
# 自编码器的隐 含层节点数n_hidden为200,隐含层的激活函数transfer_function为softplus,
# 优化器 optimizer为Adam且学习速率为0.001,同时将噪声的系数scale设为0.01。
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784, n_hidden = 200,
                transfer_function = tf.nn.softplus,
                optimizer = tf.train.AdamOptimizer(learning_rate = 0.001), scale = 0.01)

# 下面开始训练过程，在每一轮(epoch)循环开始时，我们将平均损失avg_C0St设为 0,
# 并计算总共需要的batch数(通过样本总数除以batch大小)，注意这里使用的是不放 回抽样，
# 所以并不能保证每个样本都被抽到并参与训练。然后在每一个batch的循环中，
# 先使用get_random_block_from_data函数随机抽取一个block的数据，
# 然后使用成员函数 partial_fit训练这个batch的数据并计算当前的cost,最后将当前的cost整合到avg_cost 中。
# 在每一轮迭代后，显示当前的迭代数和这一轮迭代的平均cost。我们在第一轮迭代时， cost大约为19000,
# 在最后一轮迭代时，cost大约为7000,再接着训练cost也很难继续降 低了。读者如果感兴趣，
# 可以通过调整batch_size, epoch数、优化器、自编码器的隐含 层数、隐含节点数等，来尝试获得更低的cost。
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size
        if epoch % display_step == 0:
          print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))


print("Total cost: " + str(autoencoder)+'dsadsad'+
      str(AdditiveGaussianNoiseAutoencoder.calc_total_cost(autoencoder,X_test)))



