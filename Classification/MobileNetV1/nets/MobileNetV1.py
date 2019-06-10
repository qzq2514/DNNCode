#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np



class MobileNetV1(object):
    def __init__(self, is_training, num_classes):
        self.num_classes = num_classes
        self._is_training = is_training

    def preprocess(self, inputs):
        preprocessed_inputs = tf.to_float(inputs)
        preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return preprocessed_inputs

    def _depthwise_separable_conv(self,inputs,
                                  num_pwc_filters,
                                  width_multiplier,
                                  sc,
                                  downsample=False):
        num_pwc_filters = round(num_pwc_filters * width_multiplier)
        _stride = 2 if downsample else 1

        # skip pointwise by setting num_outputs=None
        # num_outputs:pointwise 卷积的卷积核个数，如果为空，将跳过pointwise卷积的步骤,后面我们通过一般的1x1卷积自己实现pointwise
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=_stride,
                                                      depth_multiplier=1,
                                                      kernel_size=[3, 3],
                                                      scope=sc + '/depthwise_conv')

        bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_batch_norm')
        pointwise_conv = slim.convolution2d(depthwise_conv,
                                            num_pwc_filters,  # 该层卷积核个数,也是输出的feature map的通道数
                                            kernel_size=[1, 1],
                                            scope=sc + '/pointwise_conv')
        bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_batch_norm')
        return bn

    # 宽度乘数在(0,1]之间,改变卷积核个数，进而改变输入和输出的通道数
    # MobileNetV1默认输入224*224
    def inference(self, inputs,width_multiplier=1,scope='MobileNet'):
        with slim.arg_scope(self.mobilenet_arg_scope()):
            with tf.variable_scope(scope) as sc:
                end_points_collection = sc.name + '_end_points'
                with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                                    activation_fn=None,
                                    # 把每一层的输出保存到名为end_points_collection的集合中
                                    outputs_collections=[end_points_collection]):
                    with slim.arg_scope([slim.batch_norm],
                                        is_training=self._is_training,
                                        activation_fn=tf.nn.relu,  # activation_fn是卷积中规定激活函数方式的参数名
                                        fused=True):
                        net = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=2, padding='SAME',scope='conv_1')
                        # net = slim.batch_norm(net, scope='conv_1/batch_norm')
                        net = self._depthwise_separable_conv(net, 64, width_multiplier,sc='conv_ds_2')  # 进行深度可卷积-depthwise和pointwise都执行
                        net = self._depthwise_separable_conv(net, 128, width_multiplier, downsample=True, sc='conv_ds_3')
                        net = self._depthwise_separable_conv(net, 128, width_multiplier, sc='conv_ds_4')
                        net = self._depthwise_separable_conv(net, 256, width_multiplier, downsample=True, sc='conv_ds_5')
                        net = self._depthwise_separable_conv(net, 256, width_multiplier, sc='conv_ds_6')
                        net = self._depthwise_separable_conv(net, 512, width_multiplier, downsample=True, sc='conv_ds_7')

                        net = self._depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_8')
                        net = self._depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_9')
                        net = self._depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_10')
                        net = self._depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_11')
                        net = self._depthwise_separable_conv(net, 512, width_multiplier, sc='conv_ds_12')

                        net = self._depthwise_separable_conv(net, 1024, width_multiplier, downsample=True, sc='conv_ds_13')
                        net = self._depthwise_separable_conv(net, 1024, width_multiplier, sc='conv_ds_14')
                        net = slim.avg_pool2d(net, [2, 2], scope='avg_pool_15')

                net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                logits = slim.fully_connected(net, self.num_classes, activation_fn=None, scope='fc_16')

                prediction_dict = {'logits': logits}

        return prediction_dict

    def postprocess(self, prediction_dict):
        logits = prediction_dict['logits']
        logits = tf.nn.softmax(logits)
        classes = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
        postprecessed_dict = {'classes': classes}
        postprecessed_dict['logits'] = logits  # tf.clip_by_value(logits,1e-8,np.inf)
        return postprecessed_dict

    def loss(self, prediction_dict, groundtruth_lists):
        # print("groundtruth_lists:", groundtruth_lists)

        logits = prediction_dict['logits']  # 'logits'
        logits = tf.nn.softmax(logits)

        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(  # sparse_
                logits=logits + 1e-8, labels=groundtruth_lists))

        loss_dict = {'loss': loss}
        return loss_dict

    def mobilenet_arg_scope(self,weight_decay=0.0):
      with slim.arg_scope(
          [slim.convolution2d, slim.separable_convolution2d],
          weights_initializer=slim.initializers.xavier_initializer(),
          biases_initializer=slim.init_ops.zeros_initializer(),
          weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
        return sc