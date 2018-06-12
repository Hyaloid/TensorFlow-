"""自编码器是一种神经网络，借助系数编码的思想，目标是使用稀疏的一些高阶特征编码自己。
１．期望输入输出一致；２．希望使用高阶特征来重构自己，而不只是复制像素点

可以根据中间隐含层的惩罚系数控制隐含层的节点的稀疏程度，惩罚系数越大，学到的特征组合月系数，实际使用的特征数量越少。"""

import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#实现标准的均匀分布，fan_in是输入节点的数量，fan_out是输出节点的数量。
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0/(fan_in + fan_out))
    high = constant * np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out),minval=low,maxval=high,dtype=tf.float32)

#定义一个去噪自编码的class，方便之后使用。
class AdditiveGaussianNoiseAutoencoder(object):
    def_init_(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), scale=0.1)
    self.n_input = n_input
    self.n_hidden = n_hidden
    self.transfer = transfer_function
    self.scale = tf.placeholder(tf.float32)
    self.traning_scale = scale
    network_weigths = self.initialize_weight()
    self.weights = network_weights
    #定义网络结构，为输入ｘ创建一个维度为n_input的placeholder，然后建立一个能提取特征的隐含层。
    self.x = tf.placeholder(tf.float32, [None, self.n_input])
    self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_inputs,)), self.weights['w1']),self.weigths['b1']))
    self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']),self.weights['b2'])
    #定义自编码器的损失函数，直接使用平方误差squared Error作为cost
    self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
    self.optimizer = optimizer.minimize(self.cost)
    #创建Session，并初始化自编码器的全部模型.
    init = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(init)
    #创建一个名为all_weights的字典dict,然后将w1,b1,w2,b2全部存入其中，最后返回all_weights.
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input), self.n_hidden)
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input],dtype = tf.float32))
        return all_weights
    #定义计算损失cost及执行一步训练的函数partial_fit
    def partial_fit(self,X):
        cost, opt = self.sess.run((self.cost,self.optimizer),feed_dict = {self.x: X, self.scale: self.training_scale})
        return cost
    #定义一个只求损失cost的函数calc_total_cost
    def calc_total_cost(self,X):
        return self.sess.run(self.cost, feed_dict = {self.x: X,slef.scale: self.traning_scale})
    #定义transform函数，它返回自编码器隐含层的输出结果，目的是提供一个接口来获取抽象后的特征，自编码器的隐含层的最主要功能就是学习出数据中的高阶特征
    def transform(self,X):
        return self.sess.run(self.hidden, feed_dict = {self.x: X,self.scale: self.traning_scale})
    #定义generate函数，它将隐含层的输出结果作为输入，通过之后的重建层将提取到的高阶特征复原为原始数据。
    def generate(selfs,hidden = None):
        if hidden is None:
            hidden = np.random.normal(size = self.weighs["b1"])
            return self.sess.run(self.reconstruction,feed_dict = {self.hidden: hidden})
    #定义reconstruct函数，它将整体运行一遍复原过程，包括提取高阶特征和通过高阶特征复原数据，包括transform和generate两块，输入是原数据，输出是复原后的数据
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict = {self.x: X, self.scale: self.training_scale})
    #getWeights函数获取隐含层的权重w1
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    #getBiases获取隐含层的偏执系数b1
    def getBiases(self):
        return self.sess.run(self.weights['b1'])


mnist = input_data.read_data_sets('MNIST_data', one_hot = True)
#定义一个对训练、测试数据进行标准化处理的函数。
def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train,X_test
#定义一个获取随机block数据的函数，取一个从０到len(data)-batch_size之间的随机整数，再以这个随机数作为block的起始位置，然后顺序取到一个batch size 的数据。
def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0,len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]
#使用之前定义的standard_scale函数对训练集、测试集进行标准化变换.
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
#定义几个常用的参数，总训练样本数，最大训练的轮数(epoch)设为20，batch_size设为128,并设置每隔一轮(epoch)就显示一次损失cost
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1
#创建一个AGN自编码器的实例，定义模型输入节点数n_input为784,自编码器的隐含层节点数为n_hidden为200，隐含层的激活函数transfer_function为softplus
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input = 784, n_hidden = 200, transfer_function = tf.nn.softplus, \
              optimizer = tf.train.AdamOptimizer(learning_rate=0.001),scale = 0.01)
#开始训练，在每一轮(epoch)循环开始时，我们将平均损失avg_cost设为0，并计算总共需要的batch数
for epoch in range(traning_epochs):
    avg_cost = 0.
    total_batch = ini(n_samples / batch_size)
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        cost = autoencoder.partial_fit(batch_xs)
        avg_cost +=cost / n_samples * batch_size

    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
#对训练完的模型进行性能测试，使用之前定义的成员函数calc_total_cost对测试集X_test进行测试，评价指标依然是平方误差，如果使用示例中的参数，损失值约为60w
print("Total cost:" + str(autoencoder.calc_total_cost(X_test)))

"""实现自编码器和实现一个单隐含层的神经网络差不多，只不过是在数据输入时做了标准化，并加上了一个高斯噪声，同时我们的输出结果不是数字部分分类结果，而是复原的数据
因此不需要用标注过的数据进行监督训练。自编码器作为一种无监督学习的方法，它与其他无监督学习的主要不同在于，它不是对数据进行聚类，而是提取其中最有用、最频繁出现的
高阶特征，根据这些高阶特征重构数据，提取到一些有用的特征，将神经网络权重初始化到一个较好的分布，然后再使用有标注的数据进行监督训练，即对权重进行fine_tune"""