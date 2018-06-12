from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("mnist_data/", one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

#载入TensorFlow库
import tensorflow as tf
#创建一个新的InteractiveSession，使用这个命令将这个Session注册为默认的session，之后的运算也会默认跑在这个Session里面
sess = tf.InteractiveSession()
#创建一个placeholder，即输入数据的地方。Placeholder的第一个参数是数据类型。
#第二个参数【None，784】代表tensor的shape，也是数据的尺寸，None代表不限条数的输入，784代表每输入是一个784维的向量
x = tf.placeholder(tf.float32, [None, 784])
#给Softmax Regression模型中的Weights和biases创建Variable对象。
#Variable是用来存储模型参数的，与tensor不同的是，tensor一旦使用掉就会消失，而Variable在模型训练迭代中是持久的。可以长期存在并且在每轮迭代中被更新。
#把weights和biases全部初始化为0，因为模型训练的时候会自动学习合适的值，所以对这个简单的模型来说，初始值不太重要。
#不过对于复杂的卷及神经网络，循环网络或者比较深的全连接网络，初始化的方法就比较重要了。
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
#公式：y=softmax（Wx+b）   tf.matmul是Tensorflow中的矩阵乘法函数。
y = tf.nn.softmax(tf.matmul(x,W) + b)

#定义cross—entropy，通常使用cross—entropy作为loss
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),reduction_indices=[1]))

#使用随机梯度下降来训练，这个函数是封装好的，只需要调用这个函数就行，Tensorflow会自动进行后续的反向传播和梯度下降。
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#使用Tensorflow的全局参数初始化器tf.global_variables_initializer,并直接执行它的run方法。
tf.global_variables_initializer().run()
#开始迭代执行训练的操作train_step，每次都随机从训练集中取100条样本构成一个mini_batch，并feed给Placeholder
#只用了一小部分数据进行随机梯度下降，这种做法绝大多数时候会比全样本训练的收敛速度快很多。
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs, y_:batch_ys})

#对模型的准确率进行验证。tf.argmax(y,1)是求各个预测的数字中概率最大的那一个，而tf.argman(y_,1)是找样本的真实数字类别。
#tf.equal方法是用来判断预测的数字类别是否就是正确的类别，最后返回计算分类是否正确的操作是correct_prediction
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#用tf.cast将之前的correct_prediction输出的bool值转换为float32，再求平均
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))


#总结：
#写一个没有隐含层的最浅的神经网络的步骤：
#1.定义算法公式，也就是神经网络forward时的计算
#2.定义loss，选定优化器，并制定优化器优化loss
#3.迭代地对数据进行训练。
#4.在测试集或验证集上对准确率进行评测。


###Tensorflow和spark类似，我们定义的各个公式其实只是computation graph，在执行这行代码时，计算还没有实际发生
###只有等调用run方法并feed数据时计算才真正执行。