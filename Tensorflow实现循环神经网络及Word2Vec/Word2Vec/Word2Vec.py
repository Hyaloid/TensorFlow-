"""Word2Vec是将语言中的词转化为计算机可以理解的稠密向量，进而可以做其他自然语言处理任务，比如文本分类，词性标注，机器翻译等。
Word2Vec是一个可以将语言中字词转化为向量形式表达的模型。自然语言处理在Word2Vec出现之前，通常将字词转成离散的单独的符号。
One-Hot Encoder是一个词对应一个向量，通常需要将一篇文章中每一个词都转成一个向量，而整篇文章则变为一个稀疏矩阵。但是One—Hot的问题是我们对
特征的编码往往是随机的，没有提供任何关联信息，没有考虑到字词间可能存在的关系。同时，字词存储为稀疏向量的话，我们通常需要更多的数据来训练，
因为稀疏数据训练的效率比较低，计算也非常麻烦。使用向量表达则可以有效地解决这个问题。
Word2Vec是一种计算非常高效的，可以从原始语料中学习字词空间向量的预测模型。它主要分为CBOW和Skip-Gram两种模式，其中CBOW是从原始语句推测目标字词；
而Skip-Gram则正好相反，它是从目标字词推测出原始语句，其中CBOW对小型数据比较合适，而Skip-Gram在大型语料中表现得更好。"""

#Skip-Gram模式的Word2Vec

import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib
import tensorflow as tf

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verifiled', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify' + filename + '. Can you get to it with a browser?')
    return filename
filename = maybe_download('text8.zip', 31344016)

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)
print('Data size', len(words))

#创建vocabulary词汇表，使用collec.Counter统计单词列表中单词的频数，然后使用most_common方法取top 50000频数的单词作为vocabulary。
#再创建一个dict，将top 50000词汇的vocabulary放入dictionary中，以便快速查询。

vocabulary_size = 50000
def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    return data, count, dictionary, reverse_dictionary
data, count, dictionary, reverse_dictionay,= build_dataset(words)
#删除原始单词列表，可以节约内存。
del words
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionay[i] for i in data[:10]])

data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <=2 * skip_window
    batch = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i*num_skips + j] = buffer[skip_window]
            labels[i*num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionay[batch[i]], '->', labels[i,0], reverse_dictionay[labels[i,0]])
batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2

valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()
with graph.as_default():

    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        embedding_size = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size], stddev = 1.0/math.aqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                         biases=nce_biases,
                                         labels=train_labels,
                                         inputs=embed,
                                         num_sampled=num_sampled,
                                         num_classes=vocabulary_size))

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()

num_steps = 100001

with tf.Session(graph=graph) as session:
    init.run()
    print("Initialized")

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

        _, loss_val = session.run([optimizer, loss], feed_dict=fed_dict)
        average_loss += loss_val

        if step%2000 == 0:
            if step > 0:
                average_loss /= 2000
            print("Average loss at step", step, ": ", average_loss)
            average_loss = 0
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionay[valid_examples[i]]
                top_k = 8
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log_str = "Nearest to %s:" % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionay[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
            final_embeddings = normalized_embeddings.eval()

#定义一个用来可视化Word2Vec效果的函数，这里low_dim_embs是降维到2维的单词的空间向量，将在图表中展示每个单词的位置
def plot_with_labels(low_dim_embs, labels, filename = 'tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize = (18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x,y)
        plt.annotate(label,
                     xy = (x,y),
                     xytest = (5,2),
                     textcoords = 'offset points',
                     ha = 'right',
                     va = 'bottom')

    plt.savefig(filename)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 100
low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
labels = [reverse_dictionay[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)

