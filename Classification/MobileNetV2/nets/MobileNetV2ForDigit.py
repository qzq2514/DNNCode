#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np



class MobileNetV2ForDigit(object):
    def __init__(self, is_training, num_classes):
        self.num_classes = num_classes
        self._is_training = is_training

    def preprocess(self, inputs):
        preprocessed_inputs = tf.to_float(inputs)
        # preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        # preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return preprocessed_inputs


    def bootleneck(self,inputs,bottleneck_channel_upsample_rate,bottleneck_output_channels,stride,name):
        with tf.variable_scope(name):
            input_channels=inputs.get_shape().as_list()[-1]
            bottleneck=slim.convolution2d(inputs,int(input_channels*bottleneck_channel_upsample_rate),
                                          kernel_size=[1,1],stride=1,scope="pw_conv1")
            bottleneck=slim.batch_norm(bottleneck,scope="bn1")

            # num_outputs:pointwise 卷积的卷积核个数，如果为空，将跳过pointwise卷积的步骤,
            # 后面我们通过一般的1x1卷积自己实现pointwise
            bottleneck = slim.separable_convolution2d(bottleneck,num_outputs=None,stride=stride,
                                                      depth_multiplier=1,kernel_size=[3,3],scope="dw_conv")
            bottleneck = slim.batch_norm(bottleneck, scope="bn2")

            #bottleNeck的第二个pointwise_conv不使用激活函数:activation_fn=None
            bottleneck = slim.convolution2d(bottleneck, bottleneck_output_channels,
                                            kernel_size=[1, 1], stride=1, scope="pw_conv2")
            bottleneck = slim.batch_norm(bottleneck, scope="bn3",activation_fn=None)


            bottle_channels=bottleneck.get_shape().as_list()[-1]
            if bottle_channels==input_channels:
                bottleneck_output=tf.add(bottleneck,inputs)
            else:
                bottleneck_output=bottleneck
            return bottleneck_output

    # 宽度乘数在(0,1]之间,改变卷积核个数，进而改变输入和输出的通道数
    # MobileNetV1默认输入224*224
    def inference(self, inputs,scope='MobileNetV2'):
        with slim.arg_scope(self.mobilenet_arg_scope()):
            with tf.variable_scope(scope):
                with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                                    activation_fn=None):
                    # 先batch_norm,再ReLU,所以激活函数写在slim.batch_norm的参数空间中
                    with slim.arg_scope([slim.batch_norm],
                                        is_training=self._is_training,
                                        activation_fn=tf.nn.relu6,  # activation_fn是卷积中规定激活函数方式的参数名
                                        fused=True,decay=0.90):
                        # 一定要等到batch_norm的均值和方差稳定了,在测试时才能有高精度
                        # 为了使得均值方差快速稳定,可以另batch_norm的decay变小，0.95不够就0.90甚至更小
                        net = slim.convolution2d(inputs,num_outputs=32, kernel_size=[3,3], stride=2,
                                                 padding='SAME',scope='conv_1')
                        net = slim.batch_norm(net, scope="b1")

                        net = self.bootleneck(net, 1, 16, 1, "bottleneck1")
                        # net = self.bootleneck(net, 6, 24, 2, "bottleneck2_1")
                        # net = self.bootleneck(net, 6, 24, 1, "bottleneck2_2")
                        #
                        net = self.bootleneck(net, 6, 32, 2, "bottleneck3_1")
                        net = self.bootleneck(net, 6, 32, 1, "bottleneck3_2")
                        net = self.bootleneck(net, 6, 32, 1, "bottleneck3_3")

                        net = self.bootleneck(net, 6, 64, 2, "bottleneck4_1")
                        net = self.bootleneck(net, 6, 64, 1, "bottleneck4_2")
                        net = self.bootleneck(net, 6, 64, 1, "bottleneck4_3")
                        net = self.bootleneck(net, 6, 64, 1, "bottleneck4_4")

                        net = self.bootleneck(net, 6, 96, 1, "bottleneck5_1")
                        net = self.bootleneck(net, 6, 96, 1, "bottleneck5_2")
                        net = self.bootleneck(net, 6, 96, 1, "bottleneck5_3")

                        net = self.bootleneck(net, 6, 160, 2, "bottleneck6_1")
                        net = self.bootleneck(net, 6, 160, 1, "bottleneck6_2")
                        net = self.bootleneck(net, 6, 160, 1, "bottleneck6_3")

                        net= slim.convolution2d(net,num_outputs=320,kernel_size=[3,3],stride=1,scope='conv_2')
                        net=slim.batch_norm(net,scope="b2")
                        net=slim.avg_pool2d(net,kernel_size=[2,2],stride=1)
                        net = slim.convolution2d(net, num_outputs=self.num_classes, kernel_size=[3, 3],
                                                 stride=1, scope='conv_3')
                        print("net:",net.shape)

                        # 28x28x3的图片到这里得到的feature map为(1, 1, 36)
                        #一般情况下最好用tf.reshape
                        logits=tf.squeeze(net,axis=[1,2])

                        # logits = tf.reshape(net,shape=[-1,self.num_classes])
        return logits

    def postprocess(self, logits):
        softmax = tf.nn.softmax(logits)
        classes = tf.cast(tf.argmax(softmax, axis=1), tf.int32)
        # softmax:  N*num_classes,classes:N*1,其中N为样本数
        return softmax, classes

    def loss(self, logits, labels):
        softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits + 1e-8, labels=labels), name="softmax_loss")
        tf.add_to_collection("Loss", softmax_loss)
        loss_all = tf.add_n(tf.get_collection("Loss"), name="total_loss")
        return loss_all

    def mobilenet_arg_scope(self,weight_decay=0.0):
      with slim.arg_scope(
          [slim.convolution2d, slim.separable_convolution2d],
          weights_initializer=slim.initializers.xavier_initializer(),
          biases_initializer=slim.init_ops.zeros_initializer(),
          weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
        return sc