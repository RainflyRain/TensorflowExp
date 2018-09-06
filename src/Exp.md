### TensorFlow

#### 1、简单模型

前向传播过程（向量 * 矩阵 * 矩阵 -- 输入向量、权值矩阵、权值矩阵-tf.matmul(w1,w2)） 

        a = tf.matmul(x,w1)
        y = tf.matmul(a,w2)

#### 2、申明矩阵变量-变量申明函数-tf.Variable() 

        weights = tf.Variable(tf.random_normal([2,3],stddev=2))

#### 3、生成2*3矩阵，均值为0，标准差为2的随机数矩阵
        
        tf.random_normal([2,3],stddev=2)

#### 4、偏置向（bias）通常为常数--值全为0，长度为3的变量

        biases = tf.Variable(tf.zeros([3]))

#### 5、其他初始化方法

        w2 = tf.Variable(weights.initialized_value())
        w3 = tf.Variable(weights.initialized_value()*2)

#### 6、变量必须要初始化

        init_op = tf.initialize_all_variables()
        sess.run(init_op)

#### 7、维度(shape) 和 类型(type)

        w2 = tf.Variable(tf.random_normal([2,3],dtype = tf.float64,stddev=1),name = "w2")


#### 8、placeholder用法

避免变量节点过多，优化节点复用

#### 9、激活函数和偏置项

激活函数作用是去线性化，解决更多非线性问题，偏置项为常量

#### 10、tf.select(v1,v2,v3) 和 tf.greater(v1,v2)

自定义一个损失函数：loss = tf.reduce_sum(tf.select(tf.greater(v1,v2),(v1-v2)*a,(v2-v1)*b))

#### 11、学习率

#### 12、过拟合