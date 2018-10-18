# coding=utf-8
import matplotlib

matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import tensorflow as tf

# 图像预处理

# 读取图片原始数据
image_raw_data = tf.gfile.FastGFile("/path/to/picture/mao.jpg", "r").read()

with tf.Session() as sess:
    # 从jpg中解码图片数据为矩阵
    img_data = tf.image.decode_jpeg(image_raw_data)

    # 打印图片数据
    print img_data.eval()

    # 显示图片数据
    plt.imshow(img_data.eval())
    plt.show()

    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.uint8)

    encoded_image = tf.image.encode_jpeg(img_data)

    with tf.gfile.GFile("/path/to/output/result", "wb") as f:
        f.write(encoded_image.eval())
