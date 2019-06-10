import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class MobileNetForDigit(object):
    def __init__(self,is_training,num_classes):
        self.num_classes=num_classes
        self.is_training=is_training

    def preprocess(self,inputs):
        # MobileNetV1暂不需要对输入进行0均值化和归一化
        # preprocessed_inputs = tf.to_float(inputs)
        # preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        # preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return inputs

    def depthwise_separable_conv(self,preprocessed_inputs,num_pwc_filters,width_multiplier,
                                 name,downsampling=False):
        num_pwc_filters=round(num_pwc_filters*width_multiplier)     #使用宽度乘数进一步减小模型

        _stride=2 if downsampling else 1

        # skip pointwise by setting num_outputs=None
        # num_outputs:pointwise 卷积的卷积核个数，如果为空，将跳过pointwise卷积的步骤,后面我们通过一般的1x1卷积自己实现pointwise
        depthwise_conv=slim.separable_convolution2d(preprocessed_inputs,num_outputs=None,stride=_stride,
                                                    depth_multiplier=1,kernel_size=[3,3],scope=name+"/depthwise_conv")

        depthwise_bn=slim.batch_norm(depthwise_conv,scope=name+"/depthwise_batch_norm")


        #通过一般的1x1卷积实现pointwise卷积
        # num_pwc_filters:宽度乘数下减少后的pointwise卷积核个数,也就是输出的feature map的通道数
        pointwise_conv=slim.convolution2d(depthwise_bn,num_pwc_filters,kernel_size=[1,1],scope=name+"/pointwise_conv")

        pointwise_bn=slim.batch_norm(pointwise_conv,scope=name+"pointwise_batch_norm")

        return pointwise_bn

    def inference(self, preprocessed_inputs,width_multiplier=1,scope="MobileNetV1"):
        with slim.arg_scope(self.mobilenet_arg_scope()):
            with tf.variable_scope(scope) as sc:
                #在每一层卷积后不使用激活函数
                with slim.arg_scope([slim.convolution2d,slim.separable_convolution2d],activation_fn=None):
                    #仅仅在归一化层后使用激活函数-ReLU
                    with slim.arg_scope([slim.batch_norm],is_training=self.is_training,
                                        activation_fn=tf.nn.relu,fused=True,decay=0.95):  #fused:是否使用一种更快的融合方法
                        net=slim.convolution2d(preprocessed_inputs,round(32*width_multiplier),[3,3],stride=2,padding="SAME",scope="conv_1")
                        net = slim.batch_norm(net, scope='conv_1/batch_norm')

                        net = self.depthwise_separable_conv(net, 64, width_multiplier,name='conv_ds_2')  # 进行深度可卷积-depthwise和pointwise都执行
                        net = self.depthwise_separable_conv(net, 128, width_multiplier, downsampling=True, name='conv_3')
                        net = self.depthwise_separable_conv(net, 128, width_multiplier, name='conv_ds_4')
                        net = self.depthwise_separable_conv(net, 256, width_multiplier, downsampling=True, name='conv_5')
                        # net = self.depthwise_separable_conv(net, 256, width_multiplier, name='conv_ds_6')
                        # net = self.depthwise_separable_conv(net, 512, width_multiplier, downsampling=True, name='conv_7')

                        net = self.depthwise_separable_conv(net, 512, width_multiplier, name='conv_8')
                        net = self.depthwise_separable_conv(net, 512, width_multiplier, name='conv_9')
                        net = self.depthwise_separable_conv(net, 512, width_multiplier, name='conv_10')
                        net = self.depthwise_separable_conv(net, 512, width_multiplier, name='conv_11')
                        net = self.depthwise_separable_conv(net, 512, width_multiplier, name='conv_12')

                        # net = self.depthwise_separable_conv(net, 1024, width_multiplier, downsampling=True, name='conv_13')
                        # net = self.depthwise_separable_conv(net, 1024, width_multiplier, name='conv_ds_14')
                        net = slim.avg_pool2d(net, [2, 2], scope='avg_pool_15')

                shape=net.get_shape().as_list()
                flat_height,flat_width,flat_channals=shape[1:]
                flat_size=flat_height*flat_width*flat_channals
                net = tf.reshape(net, shape=[-1, flat_size])

                net=slim.fully_connected(net,self.num_classes,activation_fn=None, scope='fc_16')

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

    def mobilenet_arg_scope(self,weight_decay=0.0):
      with slim.arg_scope(
          [slim.convolution2d, slim.separable_convolution2d],
          weights_initializer=slim.initializers.xavier_initializer(),
          biases_initializer=slim.init_ops.zeros_initializer(),
          weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
        return sc
