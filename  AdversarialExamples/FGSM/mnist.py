# coding: utf-8
import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

tf.disable_v2_behavior()

#
# import os
# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# 初始化随机数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 初始化随机数
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def inference(x, keep_prob):
    # 将单张图片从784维向量重新还原为28x28的矩阵图片
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 第一层卷积层
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积层
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层，输出为1024维的向量
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 把1024维的向量转换成10维，对应10个类别
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return logits


def loss(logits, labels):
    return tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def evaluate(logits, y_):
    # 评估 argmax 1代表比较每行元素最大值
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    num = tf.argmax(logits, 1)
    num2 = tf.argmax(y_, 1)
    return correct_prediction, accuracy, num, num2


if __name__ == '__main__':

    # 读入数据
    # mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # 加载数据集
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    # x为训练图像的占位符、y_为训练图像标签的占位符
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

    # 使用Dropout，keep_prob是一个占位符，训练时为0.5，测试时为1
    keep_prob = tf.placeholder(tf.float32)
    # inference
    logits = inference(x, keep_prob=keep_prob)

    # crossentropy
    cross_entropy = loss(logits, y_)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 评测
    correct_prediction, accuracy , num= evaluate(logits, y_)

    # 创建Session和变量初始化
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # 训练1000步
    epochs = 1000
    for i in range(epochs):
        batch = mnist.train.next_batch(50)
        # 每100步报告一次在验证集上的准确
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    # 训练结束后报告在测试集上的准确度 这样写会爆显存
    # print("test accuracy %g" % accuracy.eval(feed_dict={
    #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    # 改成下面这种写法好点
    for i in range(10):
        testSet = mnist.test.next_batch(50)
        print("test accuracy %g" % accuracy.eval(feed_dict={x: testSet[0], y_: testSet[1], keep_prob: 1.0}))

    # adversarialMnist = input_data.read_data_sets("./adversiral_samples/", one_hot=True)
    # for i in range(10):
    #     testSet = adversarialMnist(5)
    #     print("test accuracy %g" % accuracy.eval(feed_dict={x: testSet[0], y_: testSet[1], keep_prob: 1.0}))

    saver = tf.train.Saver()
    saver.save(sess, "./net/model", global_step=epochs)
    saver.save(sess, './net/model.ckpt')

    # lable = [[0] * 10 for _ in range(10)]
    # images = []
    # i = 0
    # for file in os.listdir("./adversarial_image/"):
    #     image_raw = tf.gfile.FastGFile('./adversarial_image/' + file, 'rb').read()  # bytes
    #     img = tf.image.decode_jpeg(image_raw)
    #     images.append(img.eval().reshape(1, 784) / 255)
    #
    #     name = file.split(sep='.')
    #     name = name[0]
    #     pos = int(name[2])
    #     lable[pos][i] = 1
    #     i = i + 1
    # adv_image = np.array(images)
    # lable_np = np.array(lable)
    #
    # for i in range(9):
    #     print("adversiral sample accuracy 2 = ", sess.run(accuracy, feed_dict={
    #         x: adv_image[i].reshape(1, 784), y_: lable_np[i].reshape(1, 10), keep_prob: 1.0}))