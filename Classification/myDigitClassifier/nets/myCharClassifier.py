import tensorflow as tf
import numpy as np

class myCharClassifier(object):
    def __init__(self,num_classes,is_regular):
        self.num_classes=num_classes
        self.is_regular=is_regular

    #保证均值为0,范围在[-1,1]
    def preprocess(self,inputs):
        processed_inputs = tf.to_float(inputs)
        processed_inputs = tf.subtract(processed_inputs,128.0)
        processed_inputs = tf.div(processed_inputs, 128)
        return processed_inputs

    def get_variable_with_l2_loss(self, shape, stddev, wl, name):
        biases_num=shape[-1]
        var = tf.Variable(tf.truncated_normal(shape=shape, stddev=stddev), name=name+"weights")
        biases=tf.get_variable(name+"_biases",shape=[biases_num],dtype=tf.float32)
        if wl is not None:
            var_l2_loss = tf.multiply(tf.nn.l2_loss(var), wl, name=name+"_l2_loss")
            tf.add_to_collection("Loss", var_l2_loss)
        return var,biases

    def inference(self,preprocessed_inputs):
        shape = preprocessed_inputs.get_shape().as_list()
        height, width, num_channels = shape[1:]

        #卷积参数
        conv1_weights ,conv1_biases= self.get_variable_with_l2_loss([3, 3, num_channels, 32],5e-2, None, 'conv1')
        conv2_weights ,conv2_biases = self.get_variable_with_l2_loss([3, 3, 32, 32],5e-2, None, 'conv2')
        conv3_weights ,conv3_biases= self.get_variable_with_l2_loss([3, 3, 32, 64], 5e-2, None, 'conv3')
        conv4_weights ,conv4_biases = self.get_variable_with_l2_loss([3, 3, 64, 64],5e-2, None, 'conv4')
        conv5_weights ,conv5_biases = self.get_variable_with_l2_loss([3, 3, 64, 128],5e-2, None, 'conv5')
        conv6_weights,conv6_biases = self.get_variable_with_l2_loss([3, 3, 128, 128],5e-2, None, 'conv6_biases')

        #开始卷积
        net = preprocessed_inputs
        net = tf.nn.conv2d(net, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv1_biases))

        net = tf.nn.conv2d(net, conv2_weights, strides=[1, 1, 1, 1],padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv2_biases))
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

        net = tf.nn.conv2d(net, conv3_weights, strides=[1, 1, 1, 1],padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv3_biases))

        net = tf.nn.conv2d(net, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv4_biases))
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

        net = tf.nn.conv2d(net, conv5_weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv5_biases))

        net = tf.nn.conv2d(net, conv6_weights, strides=[1, 1, 1, 1],padding='SAME')
        net = tf.nn.relu(tf.nn.bias_add(net, conv6_biases))

        #展开成向量形式以供全连接
        flat_shape= net.get_shape().as_list()
        flat_height, flat_width,channals=flat_shape[1:]
        flat_size = flat_height * flat_width * channals
        net = tf.reshape(net, shape=[-1, flat_size])

        #全连接参数
        fc7_weights,fc7_biases = self.get_variable_with_l2_loss([flat_size, 512],5e-2, 0.002, 'fc7')
        fc8_weights,fc8_biases = self.get_variable_with_l2_loss([512, 512],5e-2, 0.002, 'fc8')
        fc9_weights,fc9_biases = self.get_variable_with_l2_loss([512, self.num_classes],5e-2, 0.002, 'fc9_weights')

        #全连接
        net = tf.nn.relu(tf.add(tf.matmul(net, fc7_weights), fc7_biases))
        net = tf.nn.relu(tf.add(tf.matmul(net, fc8_weights), fc8_biases))
        net = tf.add(tf.matmul(net, fc9_weights), fc9_biases)

        #返回最纯粹和原始的网络输出(不带softmax)
        return net

    def postprocess(self,logits):
        softmax=tf.nn.softmax(logits)
        classes=tf.cast(tf.argmax(softmax,axis=1),tf.int32)
        #softmax:  N*num_classes  ,classes:N*1
        #其中N为样本数
        return softmax,classes

    def loss(self,logits,labels):
        # tf.nn.sparse_softmax_cross_entropy_with_logits将元素值在[0,num_classes-1]范围的标签自动转为ont-hot形式，
        # 然后计算每个样本的损失,返回一个长度为N的向量,其中N为样本数
        # 交叉熵损失有log运算,为了防止出现0,则加一个微笑的正数(1e-8)
        # 其中logits是神经网络输出层的结果，而非softmax后的结果
        softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=logits+1e-8,labels=labels),name="softmax_loss")
        tf.add_to_collection("Loss",softmax_loss)
        loss_all=tf.add_n(tf.get_collection("Loss"),name="total_loss")

        return loss_all








