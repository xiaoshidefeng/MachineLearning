import os
import scipy.misc
import numpy as np

from mnist import *


if __name__ == '__main__':
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    mnist = input_data.read_data_sets("./data/", one_hot=True)

    # x为训练图像的占位符、y_为训练图像标签的占位符
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    y_ = tf.placeholder(tf.float32, [None, 10], name="y_")

    # 使用Dropout，keep_prob 是一个占位符，训练时为0.5，测试时为1
    keep_prob = tf.placeholder(tf.float32)
    # inference
    logits = inference(x, keep_prob=keep_prob)

    # crossentropy
    cross_entropy = loss(logits, y_)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # 准确度
    correct_prediction, accuracy , num, num2 = evaluate(logits, y_)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        # 重新获取已经训练好的模型结构
        module_file = "./net/model-1000"
        saver.restore(sess, module_file)

        # # test
        # print("test accuracy %g" % accuracy.eval(feed_dict={
        #     x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
        # 改成下面这种写法好点
        for i in range(10):
            testSet = mnist.test.next_batch(50)
            print("test accuracy %g" % accuracy.eval(feed_dict={x: testSet[0], y_: testSet[1], keep_prob: 1.0}))

        grad = tf.gradients(cross_entropy, x)
        adv_imgs = mnist.test.images.reshape((10000, 1, 784))  # 初始化样本

        n_sample = 10
        for i in range(n_sample):
            epsilon, prediction = 0.07, True
            img = adv_imgs[i]  # x_0 = x
            # 直到分类错误，说明是对抗样本
            while prediction:
                adv_img = tf.add(img, epsilon * tf.sign(grad))
                adv_imgs[i] = sess.run(adv_img, feed_dict={
                    x: img.reshape(1, 784), y_: mnist.test.labels[i].reshape(1, 10), keep_prob: 1.0})  # 计算样本

                prediction = sess.run(correct_prediction, feed_dict={
                    x: adv_imgs[i], y_: mnist.test.labels[i].reshape(1, 10), keep_prob: 1.0})

                epsilon += 0.07

            print("sample {}, eposion = {}".format(i, epsilon))

            image_array = adv_imgs[i]
            image_array = image_array.reshape(28, 28)
            save_dir = "adversiral_samples/"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            filename = save_dir + 'adv_img%d.jpg' % i
            scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)


        # print("adversiral sample accuracy = ", sess.run(accuracy, feed_dict={
        #     x: adv_imgs[0:n_sample], y_: mnist.test.labels[0 :n_sample], keep_prob: 1.0}))

        # 用下面的的写法 不然报错
        for i in range(10):

            print("adversiral sample accuracy = ", sess.run(accuracy,feed_dict={
                x: adv_imgs[i].reshape(1, 784), y_: mnist.test.labels[i] .reshape(1, 10), keep_prob: 1.0}))
            print("adversiral number = ", sess.run(num,feed_dict={x:adv_imgs[i].reshape(1, 784), keep_prob: 1.0}))
            print("real number = ", sess.run(num2, feed_dict={y_: mnist.test.labels[i] .reshape(1, 10), keep_prob: 1.0}))

        lable = [[0]*10 for _ in range(10)]
        images = []
        i = 0
        for file in os.listdir("./adversarial_image/"):
            image_raw = tf.gfile.FastGFile('./adversarial_image/' + file, 'rb').read()  # bytes
            img = tf.image.decode_jpeg(image_raw)
            images.append(img.eval().reshape(1, 784) / 255)

            name = file.split(sep='.')
            name = name[0]
            pos = int(name[2])
            lable[pos][i] = 1
            i = i + 1
        adv_image = np.array(images)
        lable_np = np.array(lable)

        for i in range(9):
            print("adversiral sample accuracy 2 = ", sess.run(accuracy,feed_dict={
                x: adv_image[i].reshape(1,784), y_: lable_np[i].reshape(1,10), keep_prob: 1.0}))
            print("adversiral number = ",sess.run(num, feed_dict={x: adv_image[i].reshape(1,784), keep_prob: 1.0}))
            print("real number = ",sess.run(num2, feed_dict={y_: lable_np[i].reshape(1,10), keep_prob: 1.0}))