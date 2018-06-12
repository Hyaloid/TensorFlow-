"""AlexNet将LeNet的思想发扬光大，把CNN的基本原理应用到了很深很宽的网络中。AlexNet主要使用到的新技术点吐下：
１．成功使用ReLu函数作为CNN的激活函数，并验证其效果在较深的网络超过了Sigmoid，成功解决了Sigmoid在网络较深时的梯度弥散问题。
　　虽然ReLu激活函数在很久之前就被提出了，但是直到AlexNet的出现才将其发扬光大。
２．训练时使用Dropout随机忽略一部分神经元，以避免模型过拟合。Dropout虽有单独的论文论述，但是AlexNet将其实用化，通过实践证实了它的效果，
　　在AlexNet中主要是最后几个全连接层使用了Dropout.
３．在CNN中使用重叠的最大池化。此前CNN中普遍使用平均池化，AlexNet全部使用最大池化，避免平均池化的模糊化效果，并且AlexNet中提出让步长比池化层核的尺寸小，
　　这样池化层的输出之间会有重叠和覆盖，提升了特征的丰富性。
４．提出了LRN层，对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型泛化能力。
５．使用CUDA加速深度卷及网络的训练，利用GPU强大的并行计算能力，处理神经网络训练时大量的矩阵运算。AlexNet使用了凉快GTX 580 GPU进行训练，
　　单个GTX 580只有3GB显存，这限制了可训练的网络的最大规模，因此，作者将AlexNet分布在两个GPU上，在每个GPU的显存中储存一半的神经元的参数。
　　因为GPU也是非常高效的，同时，AlexNet的设计让GPU之间的通信只在网络的某些曾进行，控制了通信的性能损耗。
６．数据增强，随机地从256x256的原始图像中截取224x224大小的区域（以及水平翻转的镜像），相当于增加了(256-224)^2x2=2048倍的数据量。
　　如果没有数据增强，仅靠原始的数据量，参数众多的CNN会陷入过拟合中，使用了数据增强后可以大大减轻过拟合，提升泛化能力。进行预测时，
　　则是取图片的四个角加中间共5个位置，并进行左右翻转，一共获得10张图片，对他们进行预测并对10次结果求均值。同时，AlexNet论文中提到了会对图像的RGB
　　数据进行PCA处理，并对主成分做一个标准差为0.1的高斯扰动，增加一些噪声，这个Trick可以让错误率再下降1%."""

#导入会用到的几个系统库
from datetime import datetime
import math
import time
import tensorflow as tf
#设置batch_size为32,num_batches为100,即总共测试100个batch的数据
batch_size=32
num_batches=100
#定义一个用来显示网络每一层的结构的函数print_activations,展示每一个卷积层或池化层输出tensot的尺寸。
def print_activations(t):@
    print(t.op.name, ' ', t.get_shape().as_list())
#设计AlexNet的网络结构，先定义函数inference,它接受images作为输入，返回最后一层pool5(第5个池化层)及parameters(AlexNet中所有需要训练的模型参数)
def inference(images):
    parameters = []

    with tf.name_scope('cov1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,stddev=1e01),name='weights')
        conv = tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv1 = tf.nn.relu(bias,name=scope)
        print_activations(conv1)
        parameters +=[kernel,biases]
#在第1个卷积层后再添加LRN和最大池化层。
    lrn1 = tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1')
    pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool1')
    print_activations(pool1)
#设计第2个卷积层，大部分步骤和第1个卷积层相同，只有几个参数不同。
    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(pool1,kernel,[1,1,1,1,],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[192],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters +=[kernel,biases]
    print_activations(conv2)
#对第2个卷积层的输出conv2进行处理，同样是先做LRN处理，再进行最大池化处理
    lrn2 = tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn2')
    pool2 = tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID',name='pool2')
    print_activations(pool2)
#创建第3个卷积层，基本结构和前面两个类似，也只是参数不同。
    with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv3 = tf.nn.relu(bias, name=scope)
    parameters +=[kernel,biases]
    print_activations(conv3)
#创建第四个卷积层
    with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32),trainable=True,name='biases')
        bias = tf.nn.bias_add(conv,biases)
        conv4 = tf.nn.relu(bias,name=scope)
        parameters += [kernel,biases]
        print_activations(conv4)
#创建第５个卷积层
    with tf.name_scope('conv5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32, stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(conv4, kernel, [1,1,1,1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0,shape=[256],dtype=tf.float32), trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv5)
#在第５个卷积层之后，还有一个最大池化层，这个池化层和前两个卷积层后的池化层一致，最后我们返回这个池化层的输出pool5
    pool5 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1],padding='VALID',name='pool5')
    print_activations(pool5)
    return pool5,parameters
#接下来实现一个评估AlexNet每轮计算时间的函数time_tensorflow_run。这个函数的第一个输入是TensorFlow的Session,第二个变量是需要评测的运算算子，
#第三个变量是测试的名称。
def time_tensorflow_run(session, target, info_string):
    num_steps_burn_in = 10
    total_duration = 0.0
    total_duration_squared = 0.0
#进行num_batches+num_steps_burn_in次迭代计算，使用time.time()记录时间，每次迭代通过session.run(target)执行。
#在初始热身的num_steps_burn_in次迭代后，每10轮迭代显示当前迭代所需要的时间。同时每轮将total_duration和total_duration_squared累加，以便后面计算每轮耗时的均值和标准差。
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target)
        duration = time.time() - start_time
        if i >=num_steps_burn_in:
            if not i%10:
                print('%s:step %d, duration = %.3f' %(datetime.now(), i-num_steps_burn_in,duration))
            total_duration +=duration
            total_duration_squared +=duration * duration
    #在循环结束后，计算每轮迭代的平均耗时mn和标准差sd，最后将结果显示出来。这样就完成了计算每轮迭代耗时的评测函数time_tensorflow_run
    mn = total_duration / num_batches
    vr = total_duration_squared / num_batches - mn * mn
    sd = math.sqrt(vr)
    print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), info_string, num_batches, mn, sd))
#接下来是主函数run_benchmark。首先使用with tf.Graph().as_default()定义默认的Graph方便后面使用。如前面所说，我们并不使用ImageNet数据集来训练
#只使用随机图片数据测试前馈和反馈计算的耗时。
def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224
        images = tf.Variable(tf.random_normal([batch_size,
                                               image_size,
                                               image_size, 3],
                                              dtype=tf.float32,
                                              stddev=1e-1))
        pool5, parameters = inference(images)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        #下面进行AlexNet的forward计算的评测，这里直接使用time_tensorflow_run统计运算时间，传入的target就是pool5,即卷积网络最后一个池化层的输出。
        #然后进行backward，即训练过程的评测，这里和forward计算有些不同，我们需要给最后的输出pool5设置一个优化目标loss
        time_tensorflow_run(sess, pool5, "Forward")

        objective = tf.nn.l2_loss(pool5)
        grad = tf.gradients(objective, parameters)
        time_tensorflow_run(sess,grad,"Forward-backward")
        #最后执行主函数.
run_benchmark()
#程序显示的结果有三段，首先是AlexNet的网络结构，可以看到我们定义的5个卷积层中第1个，第2个和第5个卷积层后面还连接着池化层，另外每一层输出tensor的尺寸也显示出来了。
"""CNN的训练过程（即backward计算）通常都比较耗时，而且不像预测过程（即forward计算），训练通常需要过很多遍数据，进行大量的迭代，因此应用CNN的主要瓶颈还是在训练，
用CNN做预测问题不打。目前TensorFlow已经支出在iOS,Android系统中运行，所以在手机上使用CPU进行人脸识别的图片分类已经非常方便了，并且响应速度也很快。"""