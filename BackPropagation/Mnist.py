#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

class MutilClass:
    def __init__(self):
        # 加载数据集
        self.Mnsit = input_data.read_data_sets("./data/", one_hot=True)

        # 设置神经网络层参数
        self.n_hidden_1 = 256
        self.n_hidden_2 = 128
        self.n_input = 784
        self.n_classes = 10

        self.x = tf.placeholder(dtype=float, shape=[None, self.n_input], name="x")
        self.y = tf.placeholder(dtype=float, shape=[None, self.n_classes], name="y")
        # random_normal 高斯分布初始化权重
        # stddev是正态分布的标准差
        self.weights = {
            "w1": tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1], stddev=0.1)),
            "w2": tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2], stddev=0.1)),
            "out": tf.Variable(tf.random_normal([self.n_hidden_2, self.n_classes], stddev=0.1))
        }
        self.bias = {
            "b1": tf.Variable(tf.random_normal([self.n_hidden_1])),
            "b2": tf.Variable(tf.random_normal([self.n_hidden_2])),
            "out": tf.Variable(tf.random_normal([self.n_classes]))
        }

        print("参数初始化完成！")

    # 定义一个MLP，前向感知器
    def _multilayer_perceptron(self, _X, _weights, _bias):
        # tf.matmul（）将矩阵a乘以矩阵b，生成a * b
        # w1*X + b
        # sigmoid是 y = 1/(1 + exp (-x))  激活函数
        layer_1 = tf.sigmoid(tf.add(tf.matmul(_X, _weights["w1"]), _bias["b1"]))
        layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1, _weights["w2"]), _bias["b2"]))
        print(tf.matmul(layer_2, _weights["out"]) + _bias["out"])
        return (tf.matmul(layer_2, _weights["out"]) + _bias["out"])

    # 定义反向传播
    def _back_propagation(self):
        pred = self._multilayer_perceptron(self.x, self.weights, self.bias)
        # logits 未归一化的概率
        # softmax归一化，为了平衡概率分布，同时避免出现概率为0的情况
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)
        corr = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        accr = tf.reduce_mean(tf.cast(corr, dtype=float))

        init = tf.global_variables_initializer()
        return init, optimizer, cost, accr

    # 训练模型
    def _train_model(self, _init, _optimizer, _cost, _accr):
        epochs = 100
        batch_size = 100
        display_steps = 1
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        sess.run(_init)
        x = []
        y = []
        y2 = []

        for epoch in range(epochs):
            avg_cost = 0
            total_batch = int(self.Mnsit.train.num_examples / batch_size)
            for i in range(total_batch):
                batch_xs, batch_ys = self.Mnsit.train.next_batch(batch_size)
                feeds = {self.x: batch_xs, self.y: batch_ys}
                sess.run(_optimizer, feed_dict=feeds)
                avg_cost += sess.run(_cost, feed_dict=feeds)
            avg_cost = avg_cost / total_batch

            if (epoch + 1) % display_steps == 0:
                print("Epoch: {} / {}, cost: {}".format(epoch, epochs, avg_cost))
                feeds = {self.x: batch_xs, self.y: batch_ys}
                train_acc = sess.run(_accr, feed_dict=feeds)
                print("Train Accuracy: {}".format(train_acc))

                feeds = {self.x: self.Mnsit.test.images, self.y: self.Mnsit.test.labels}
                test_acc = sess.run(_accr, feed_dict=feeds)
                print("Test Accuracy: {}".format(test_acc))
                print("-" * 50)
                x.append(epoch)
                y.append(train_acc)
                y2.append(test_acc)

        plt.plot(x, y, label='train data accuracy')
        plt.plot(x, y2, label='test data accuracy')
        plt.xlabel('epoch') # xlabel 方法指定 x 轴显示的名字
        plt.ylabel('accuracy') # ylabel 方法指定 y 轴显示的名字
        plt.title('mnist')
        plt.legend() # legend 是在图区显示label，即上面 .plot()方法中label参数的值
        plt.show()





if __name__ == '__main__':

    network = MutilClass()
    init, optimizer, cost, accr = network._back_propagation()
    network._train_model(init, optimizer,cost, accr)