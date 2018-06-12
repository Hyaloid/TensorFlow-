"""VGGNet是牛津大学视觉自和Google DeepMind公司的研究员一起研发的深度卷积神经网络。探索了卷积神经网络的深度与其性能之间的关系，通过反复堆叠3x3的小型卷积核
和2x2的最大池化层，VGGNet成功地构筑了16~19层深的卷积神经网络。　　　1x1的卷积的意义主要在于线性变换，而输入通道数和输出通道数不变，没有发生降维。"""
# VGGNet拥有5段卷积，每一段内有2~3个娟姐才能够，同时每段尾部会连接一个最大池化层用来缩小图片尺寸。
"""1.LRN层作用不大；2.月神的网络效果越好；3.1x1的卷积也是很有效的，但是没有3x3的卷积好，大一些的卷积核可以学习更大的空间特征。"""
from datetime import datetime
import math
import time
import tensorflow as tf


# 先写一个函数conv_op，用来创建卷积层并把本层的参数存入参数列表。

def conv_op(input_op, name, kh, kw, n_out, dh, dw, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(
            scope + 'w',
            shape=[kh, kw, n_in, n_out],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer_conv2d()
        )
        # 接着使用tf.nn.conv2d对input_op进行卷积处理，卷积核即为kernel,步长是dhxdw,padding模式设为SAME。biases使用tf,constant复制为0
        conv = tf.nn.conv2d(input_op, kernel, (1, dh, dw, 1), padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out], dtype=tf.float32)
        biases = tf.Variable(bias_init_val, trainable=True, name='b')
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        p += [kernel, biases]
        return activation


# 下面定义全连接层的创建函数fc_op。一样是先获取输入input_op的通道数，然后使用tf.get_variable创建全连接层的参数，只不过参数的维度只有两个，第一个维度为输入的通道数，
# n_in,第二个维度为输出的通道数n_out。同样，参数初始化方法也使用xavier_initializer。这里biases不再初始化为0，为啥赋予一个较小的值0.1以避免dead neuron.
# 然后使用tf.nn.relu_layer对输入变量input_op与kernel做矩阵乘法并加上biases，再做ReLu非线性变换得到activation。最后将这个全连接层用到参数kernel,biases添加到参数列表p,并将activation作为函数结果返回。
def fc_op(input_op, name, n_out, p):
    n_in = input_op.get_shape()[-1].value

    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + 'w',
                                 shape=[n_in, n_out], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.Variable(tf.constant(0.1, shape=[n_out], dtype=tf.float32), name='b')
        activation = tf.nn.relu_layer(input_op, kernel, biases, name=scope)
        p += [kernel, biases]
        return activation


# 再定义最大池化层的创建函数mpool_op。这里直接使用tf.nn.max_pool,输入即为input_op，池化尺寸为khxkw，步长是dhxdw,ｐａｄｄｉｎｇ模式设为SAME
def mpool_op(input_op, name, kh, kw, dh, dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1, kh, kw, 1],
                          strides=[1, dh, dw, 1],
                          padding='SAME',
                          name=name)


# 完成了卷积层、全连接层和最大池化层的创建函数，接下来就开始创建VGGNet-16的网络结构。
def inference_op(input_op, keep_prob):
    p = []

    conv1_1 = conv_op(input_op, name="conv1_1", kh=3, kw=3, n_out=64, dh=1, dw=1, p=p)
    conv1_2 = conv_op(conv1_1, name="pool1", kh=3, kw=3, n_out=63, dh=1, dw=1, p=p)
    pool1 = mpool_op(conv1_2, name="pool1", kh=2, kw=2, dh=2, dw=2)
    # 第二段卷及网络和第一段非常类似，同样是两个卷积层加一个最大池化层，两个卷积层的卷积核尺寸也是3x3,但是输出通道数变为128
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool2 = mpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)
    # 第三段卷及网络，这里有3个卷积层和1个最大池化层。
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)
    # 第四段卷积网络也是3个卷积层加1个最大池化层。
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)
    # 最后一段卷及网络有所变化，这里卷积输出的通道数不再增加，继续维持在512。最后一段卷积网络同样是3个卷积层加一个最大池化层，卷积核尺寸为3x3,步长为1x1,池化层尺寸为2x2，步长为2x2.
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3, name="pool5", kh=2, kw=2, dh=2, dw=2)
    # 将第5段卷积网络的输出结果进行扁平化，使用tf.reshape函数将每个样本化为长度为7x7x512=25088的一维向量。
    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5, [-1, flattened_shape], name="resh1")
    # 然后连接一个隐含层节点数为4096的全连接层，激活函数为ReLU。然后连接一个Dropout层，在训练时节点保留率为0.5,预测时为1.0
    fc6 = fc_op(resh1, name="fc6", n_out=4096, p=p)
    print("sb1")
    fc6_drop = tf.nn.dropout(fc6, keep_prob, name="fc_drop")
    # 接下来是一个和前面一样的全连接层，之后同样连接一个Dropout层
    fc7 = fc_op(fc6_drop, name="fc7", n_out=4096, p=p)
    print("sb2")
    fc7_drop = tf.nn.dropout(fc7, keep_prob, name="fc_drop")
        # 创建session并初始化全局参数。ut(fc7, keep_prob, name="fc7_drop")
    # 最后连接一个有1000个输出节点的全连接层，并使用Softmax进行处理得到分类输出概率。这里使用tf.argmax求输出概率最大的类别。
    fc8 = fc_op(fc7_drop, name="fc8", n_out=1000, p=p)
    print("sb3")
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax, 1)
    return predictions, softmax, fc8, p


# 评测函数time_tensorflow_run()和前面AlexNet中的非常想死，只有一点区别：我们在session.run()方法中引入了feed_dict,方便后面传入keep_prob来控制Dropout层的保留比率
def time_tensorflow_run(session, target, feed, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target, feed_dict=feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:
            if not i % 10:
                print('%s: step %d, duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
            total_duration += duration
            total_duration_squared += duration * duration
        mn = total_duration / num_batches
        vr = total_duration_squared / num_batches - mn * mn
        sd = math.sqrt(vr)
#       print('%s: %s across %d steps, %.3f +/- %.3f sec /batch' % (datetime.now(), info_string, num_batches, mn, sd))


# 定义评测的主函数run_benchmark,我们的目标依然是仅评测forward和backward的运算性能，并不进行实质的训练和预测。
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size, 3],
                                              dtype=tf.float32,
                                              stddev=1e-1))
        # 接下来，创建keep_prob的placeholder,并调用inference_op函数构建VGGNet-16的网络结构
        keep_prob = tf.placeholder(tf.float32)
        predictions, softmax, fc8, p = inference_op(images, keep_prob)
        # 创建session并初始化全局参数。
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        # 预测
        time_tensorflow_run(sess, predictions, {keep_prob: 1.0}, "Forward")
        objective = tf.nn.l2_loss(fc8)
        grad = tf.gradients(objective, p)
        time_tensorflow_run(sess, grad, {keep_prob: 0.5}, "Forward-backward")


batch_size = 1
num_batches = 100
run_benchmark()
