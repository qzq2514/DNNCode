import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np

class ShuffleNetV2(object):
    def __init__(self,num_classes,group_num,model_size,is_training):
        self.num_classes=num_classes
        self.group_num=group_num
        self.is_training=is_training
        self.model_scales=self.select_model_scalse(model_size)

    def select_model_scalse(self,model_size):
        if model_size==0.5:
            return [(48, 4), (96, 8), (192, 4), (1024, 1)]
        elif model_size == 1.0:
            return [(116, 4), (232, 8), (464, 4), (1024, 1)]
        elif model_size == 1.5:
            return [(176, 4), (352, 8), (704, 4), (1024, 1)]
        elif model_size == 2.0:
            return [(244, 4), (488, 8), (976, 4), (2048, 1)]
        else:
            raise ValueError('Unsupported model size!Please set model size in [0.5,1.0,1.5,2.0]!')
    #grouped_conv的kernel_size=[1,1],stride=[1,1]
    def grouped_conv(self,inputs,group_num,output_channels,strides,padding,name):
        input_channels=inputs.get_shape().as_list()[-1]
        # print("input_channels:",input_channels)
        input_groups_channel = [input_channels//group_num]*group_num
        output_groups_channel = [output_channels//group_num]*group_num

        input_groups_channel[-1] = input_channels-input_groups_channel[0]*(group_num-1)
        output_groups_channel[-1] = output_channels - output_groups_channel[0] * (group_num - 1)

        group_conv_list=[]
        channels_start =0
        # print("input_groups_channel:",input_groups_channel)
        for gooup_id in range(group_num):
            channels_end=channels_start+input_groups_channel[gooup_id]
            cur_conv=self.conv2d(inputs[:,:,:,channels_start:channels_end],
                                     kernel_size=[1,1],filters_num=output_groups_channel[gooup_id],
                                     strides=strides,padding=padding,name=name+"_conv"+str(gooup_id))
            channels_start=channels_end
            group_conv_list.append(cur_conv)

        group_conv_result=tf.concat(group_conv_list,axis=-1)
        return group_conv_result

    def channel_shuffle(self,inputs,group_num,name):
        N,H,W,C=inputs.get_shape().as_list()
        inputs_reshaped=tf.reshape(inputs,[-1,H,W,group_num,C//group_num],name=name+"_reshape1")
        inputs_transposed=tf.transpose(inputs_reshaped,[0,1,2,4,3],name=name+"transpose")
        result=tf.reshape(inputs_transposed,[-1,H,W,C],name=name+"_reshape2")
        return result

    def conv_bn_relu(self,name,inputs,out_channels,kernel_size=1,stride=1):
        with tf.variable_scope(name):
            net=slim.convolution2d(inputs,out_channels,kernel_size=kernel_size,stride=stride)
            net=slim.batch_norm(net)
        return net
    def depthwise_conv_bn(self,name,inputs,stride=1):
        with tf.variable_scope(name):
            net=slim.separable_convolution2d(inputs,None,kernel_size=3,depth_multiplier=1,
                                             stride=stride)
            net=slim.batch_norm(net,activation_fn=None)
        return net
    def shuffleNetV2_unit(self,name,inputs,unit_output_channels,stride):
        with tf.variable_scope(name):
            if stride==1:   #对指定输出通道
                top,bottom=tf.split(inputs,num_or_size_splits=2,axis=3)

                half_channel=unit_output_channels//2

                top = self.conv_bn_relu("conv_bn_relu1",top,half_channel)
                top = self.depthwise_conv_bn("depthwise_conv_bn1", top)
                top = self.conv_bn_relu("conv_bn_relu2", top, half_channel)

                unit_out=tf.concat([top,bottom],axis=3)
                unit_out=self.channel_shuffle(unit_out,group_num=self.group_num,name="shuffle")

            else:
                half_channel = unit_output_channels // 2

                branch1=self.conv_bn_relu("conv_bn_relu_b11",inputs,half_channel)
                branch1=self.depthwise_conv_bn("depthwise_conv_b1",branch1,stride=2)
                branch1=self.conv_bn_relu("conv_bn_relu_b12",branch1,half_channel)

                branch2=self.depthwise_conv_bn("depthwise_conv_b2",inputs,stride=2)
                branch2 = self.conv_bn_relu("conv_bn_relu_b21", branch2, half_channel)

                unit_out = tf.concat([branch1, branch2], axis=3)
                unit_out = self.channel_shuffle(unit_out, group_num=self.group_num, name="shuffle")
        return unit_out

    def avg_pool2d(self,inputs):
        n, h, w, c = inputs.get_shape().as_list()
        return slim.avg_pool2d(inputs,kernel_size=[h,w],stride=1)

    def preprocess(self, inputs):
        MEAN = [103.94, 116.78, 123.68]
        NORMALIZER = 0.017

        processed_inputs = tf.to_float(inputs)

        red, green, blue = tf.split(processed_inputs, num_or_size_splits=3, axis=3)
        preprocessed_input = tf.concat([
            tf.subtract(blue, MEAN[0]) * NORMALIZER,
            tf.subtract(green, MEAN[1]) * NORMALIZER,
            tf.subtract(red, MEAN[2]) * NORMALIZER,
        ], 3)

        return preprocessed_input


    def inference(self,preprocessed_inputs):
        with slim.arg_scope([slim.convolution2d,slim.separable_convolution2d],
                            weights_initializer=slim.initializers.xavier_initializer(),
                            biases_initializer=slim.init_ops.zeros_initializer(),
                            activation_fn=None) as sc1:
            with slim.arg_scope([slim.batch_norm], is_training=self.is_training,
                                activation_fn=tf.nn.relu, fused=True, decay=0.95) as sc2:
                with tf.variable_scope("init_input"):
                    net = self.conv_bn_relu("conv_bn_relu",preprocessed_inputs,24,kernel_size=3,stride=2)
                    net = slim.max_pool2d(net,kernel_size=3,stride=2)
                for stage_ind,block_info in enumerate(self.model_scales[:-1]):
                    output_channels,repeat_time=block_info

                    #每个stage的第一个unit的stride=2用于下采样
                    net=self.shuffleNetV2_unit("stage{}_unit{}".format(stage_ind+2,1),
                                               net,output_channels,stride=2)

                    for unit_ind in range(repeat_time-1):
                        net=self.shuffleNetV2_unit("stage{}_unit{}".format(stage_ind+2,unit_ind+2),
                                                   net,output_channels,stride=1)


                net = self.conv_bn_relu("conv_bn_relu_end",net,self.model_scales[-1][0],)
                net = self.avg_pool2d(net)

            fc_conv = slim.convolution2d(net, self.num_classes, kernel_size=1, stride=1)
            logits = tf.squeeze(fc_conv, axis=[1, 2])

            # net = tf.squeeze(net,axis=[1,2])
            # logits = slim.fully_connected(net,self.num_classes,activation_fn=None,
            #                               weights_regularizer=slim.l2_regularizer(0.9))

        return logits


    def postprocess(self,logits):
        softmax = tf.nn.softmax(logits)
        classes = tf.cast(tf.argmax(softmax, axis=1), tf.int32)
        return softmax, classes

    def loss(self,logits,labels):
        softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=logits+1e-8,labels=labels),name="softmax_loss")
        # l2_loss=tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),name="regular_loss")
        return softmax_loss








