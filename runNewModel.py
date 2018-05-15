# -*- coding:utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np
from sys import path

path.append('../..')
#from common import extract_mnist


# 初始化单个卷积核上的参数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化单个卷积核上的偏置值
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 输入特征x，用卷积核W进行卷积运算，strides为卷积核移动步长，
# padding表示是否需要补齐边缘像素使输出图像大小不变
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 对x进行最大池化操作，ksize进行池化的范围，
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def main():
    # 定义会话
    sess = tf.InteractiveSession()

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    saver.restore(sess, 'D:\\tmp\\model_data\\model.ckpt')

    #init = tf.global_variables_initializer()
    #sess.run(y)

    im = cv2.imread('d:\\8.bmp', cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (28, 28), interpolation=cv2.INTER_CUBIC)
    # 图片预处理
    #img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) #.astype(np.float32)
    # 数据从0~255转为-0.5~0.5
    img_gray = (im - (255 / 2.0)) / 255
    # cv2.imshow('out',img_gray)
    # cv2.waitKey(0)
    x_img = np.reshape(im, [-1, 784])

    print(x_img)
    x_img
    output = sess.run(y, feed_dict={x: x_img})
    print(output)
    'the y_con :   ', '\n', output
    print
    'the predict is : ', np.argmax(output)
    print(np.argmax(output))

    # 关闭会话
    sess.close()


if __name__ == '__main__':
    main()

