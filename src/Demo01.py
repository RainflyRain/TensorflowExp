# coding=utf-8
# 一个简单的前向传播算法
import os

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 生命两个变量，设置seed随机种子，保证每次运行结果一致
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 输入特征向量
x = tf.constant([[0.7, 0.9]])

# 前向传播算法
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

sess = tf.Session()

# 变量需要初始化
sess.run(w1.initializer)
sess.run(w2.initializer)

# 输出
sess.run(a)
print(sess.run(y))
sess.close()
