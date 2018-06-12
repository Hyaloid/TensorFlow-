"""softmax最大的特点是简单易用，但是拟合能力不强。　　层数越多，审计网路哦所需要的隐含层节点可以越少，层数越深，概念越抽象。
实际使用中，使用层数较深的神经网络容易过拟合、参数难以调试、梯度弥散等，对于这些问题我们需要很多Ｔrick来解决。
隐含层的一个代表性的功能是可以解决XOR问题。"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)
sess = tf.InteractiveSession()

in_units = 784
h1_inits = 300
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev = 0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

#定义输入x的placeholder。在训练和预测时,Dropout的比率keep_prob是不一样的，通常在训练时小于１，而预测时则等于１，所以也把Dropout的比率作为计算图的输入，并定义成一个placeholder
x = tf.placeholder(tf.float, [None, in_units])
keep_prob = tf.placeholder(tf.float32)

#定义模型结构
hidden = tf.nn.relu(tf.matmul(x,W1) + b1)
hidden_drop = tf.nn.dropout(hidden, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden_drop.W2) + b2)

#定义损失函数和选择优化器来优化loss
y_ = tf.placeholder(tf.float32,[None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))
train_step = tf.train.AdagradDAOptimizer(0.3).minimize(cross_entropy)

#训练，与之前的训练不同的是加入了keep_prob作为计算图的输入，并且在训练时设为0.75,即保留75%的节点，其余25%置为0
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: mnist.test.images, y_: batch_ys, keep_prob: 0.75})

#对模型进行准确率评测
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

"""没有隐含层的Softmax Regression只能直接从图像的像素点推断是哪个数字，而没有特征抽象的过程。多层神经网络依靠隐含层
则可以组合出高阶特征，比如横线、竖线、圆圈等，之后可以将这些高阶特征或者说组件再组合成数字，就能实现精准的匹配和分类。"""
"""使用全连接神经网络也是有局限的，即使使用很深的网络，很多的隐藏节点、很大的迭代轮数，也很难在MNIST数据集上达到99%以上的准确率。"""
