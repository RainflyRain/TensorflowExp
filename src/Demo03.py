# coding=utf-8
import tensorflow as tf

from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# cross_entropy为交叉熵，刻画预测的概率 和 真实答案的概率 之间的距离
# 定义损失函数（y_ 代表正确结果 y为预测记过 tf.clip_by_value()函数可以
# 将张量限制在指定范围之内，小于取最小值，大于取最大值）
# tf.log()对张量所有元素取对数
# * 乘法，对张量元素一次相乘
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

# 反向传播算法
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataSet_size = 128
X = rdm.rand(dataSet_size, 2)

# 定义规则给出样本标签，x1 + x2 < 1为正样本，其他为负样本
Y = [[int(x1 + x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    '''
    训练之前神经网络参数的值：
    '''

    # 设定训练轮数
    STEPS = 5000
    for i in range(STEPS):
        # 每次选取batch_size 个样本进行训练
        start = (i * batch_size) % dataSet_size
        end = min(start + batch_size, dataSet_size)

        # 通过样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d trainning steps,cross entropy on all data is %g" % (i, total_cross_entropy))

    print sess.run(w1)
    print sess.run(w2)