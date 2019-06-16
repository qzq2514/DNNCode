import tensorflow as tf
import numpy as np

class ShuffleNetV1ForDigit(object):
    def __init__(self,num_classes,group_num,is_training):
        self.num_classes=num_classes
        self.group_num=group_num
        self.is_training=is_training

    #shape=[3,3,input_channels,output_channels]
    def get_variable_with_l2_loss(self,shape,weight_decay,name,is_depthwise_conv=False):
        if is_depthwise_conv:
            bias_num=shape[2]
        else:
            bias_num=shape[-1]
        # print("shape:",shape)
        # print(name+"_weights")
        # print("------")
        weights=tf.get_variable(name+"_weights",shape=shape,
                                initializer=tf.contrib.layers.xavier_initializer())
        biases=tf.get_variable(name+"biases",shape=[bias_num],dtype=tf.float32)
        if weight_decay is not None:
            l2_loss=tf.multiply(tf.nn.l2_loss(weights),weight_decay,name=name+"_l2_loss")
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,l2_loss)
        return weights,biases

    # kernel_size=[3,3]
    # strides=[1,1]
    def conv2d(self,inputs,kernel_size,filters_num,strides,padding,name,weight_decay=None):
        strides=[1,strides[0],strides[1],1]
        input_channels=inputs.get_shape().as_list()[-1]
        weights_shape=[kernel_size[0],kernel_size[1],input_channels,filters_num]
        weights,biases=self.get_variable_with_l2_loss(shape=weights_shape,
                                                      weight_decay=weight_decay,name=name+"_conv")
        result=tf.nn.conv2d(inputs,weights,strides=strides,padding=padding)
        result=tf.nn.bias_add(result,biases)
        return result

    # kernel_size=[3,3]
    # strides=[1,1]
    def depthwise_conv(self,inputs,kernel_size,strides,padding,name,weight_decay=None):
        input_channels = inputs.get_shape().as_list()[-1]
        strides = [1, strides[0], strides[1], 1]
        #深度可卷积的卷积核参数是[filter_height, filter_width, in_channels, channel_multiplier],最后一个是宽度乘数
        weights_shape = [kernel_size[0], kernel_size[1], input_channels, 1]
        weights, biases = self.get_variable_with_l2_loss(shape=weights_shape,is_depthwise_conv=True,
                                                         weight_decay=weight_decay, name=name)
        result = tf.nn.depthwise_conv2d(inputs,filter=weights,strides=strides,padding=padding)
        result = tf.nn.bias_add(result, biases)
        return result

    #grouped_conv的kernel_size=[1,1],stride=[1,1]
    #grouped_conv不使用激活函数
    def grouped_conv(self,inputs,group_num,output_channels,strides,padding,is_batch_norm,is_activation,name):
        input_channels=inputs.get_shape().as_list()[-1]
        # print("input_channels:",input_channels)
        input_groups_channel = [input_channels//group_num]*group_num
        output_groups_channel = [output_channels//group_num]*group_num

        input_groups_channel[-1] = input_channels-input_groups_channel[0]*(group_num-1)
        output_groups_channel[-1] = output_channels - output_groups_channel[0] * (group_num - 1)

        group_conv_list=[]
        channels_start =0
        for gooup_id in range(group_num):
            channels_end=channels_start+input_groups_channel[gooup_id]
            cur_conv=self.conv2d(inputs[:,:,:,channels_start:channels_end],
                                     kernel_size=[1,1],filters_num=output_groups_channel[gooup_id],
                                     strides=strides,padding=padding,name=name+"_conv"+str(gooup_id),weight_decay=0.001)
            if is_batch_norm:
                cur_conv=tf.layers.batch_normalization(cur_conv,training=self.is_training,
                                                       epsilon=1e-5,name=name+"_bn"+str(gooup_id))
            if is_activation:
                cur_conv=tf.nn.relu(cur_conv)
            group_conv_list.append(cur_conv)
            channels_start = channels_end

        group_conv_result=tf.concat(group_conv_list,axis=-1)
        return group_conv_result

    #此channal_shuffle方式是将最后的通道维度reshape成一个矩阵,然后将该矩阵转置再拉回成向量
    #eg:某个feature map在(x,y)处的通道元素为[1,2,3,4,5,6],若组数为2，则变换过程如下:
    #[[1,2,3],[4,5,6]]-->[[1,4],[2,5],[3,6]]-->[1,4,2,5,3,6]
    #所以这里只要输入数据和组数是固定的，那么输出也是固定的,其实并未达到随机的效果
    def channel_shuffle(self,inputs,group_num,name):
        N,H,W,C=inputs.get_shape().as_list()
        inputs_reshaped=tf.reshape(inputs,[-1,H,W,group_num,C//group_num],name=name+"_reshape1")
        inputs_transposed=tf.transpose(inputs_reshaped,[0,1,2,4,3],name=name+"transpose")
        result=tf.reshape(inputs_transposed,[-1,H,W,C],name=name+"_reshape2")
        return result

    def shuffleNet_unit(self,name,inputs,unit_output_channels,strides,residual_type="concat",is_use_group_shuffle=True):
        residual=inputs
        input_channels=residual.get_shape().as_list()[-1]
        if residual_type=="concat":
            bottleneck_channels=(unit_output_channels-input_channels)//4
            bottleneck_output_channels=unit_output_channels-input_channels
        else:
            bottleneck_channels=unit_output_channels//4
            bottleneck_output_channels = unit_output_channels

        if not is_use_group_shuffle:
            #grouped_conv中的卷积都是1步长和1宽高的卷积核
            bottleneck=self.conv2d(inputs,kernel_size=[1,1],filters_num=bottleneck_channels,
                                   strides=[1,1],padding="VALID",name=name+"_conv")
            bottleneck = tf.nn.relu(bottleneck)
        else:
            bottleneck = self.grouped_conv(inputs,output_channels=bottleneck_channels,padding="VALID",
                                           strides=[1,1],group_num=self.group_num,is_batch_norm=True,
                                           is_activation=True,name=name+"_group1")

            bottleneck = tf.layers.batch_normalization(bottleneck, training=self.is_training, epsilon=1e-5,name=name+"_bn1")
            bottleneck = tf.nn.relu(bottleneck)
            # print("inputs5:", bottleneck)
            bottleneck=self.channel_shuffle(bottleneck,self.group_num,name=name+"_shuffle")

        #长宽维度上两边各填充一个常数
        bottleneck=tf.pad(bottleneck,[[0,0],[1,1],[1,1],[0,0]],"CONSTANT")

        bottleneck=self.depthwise_conv(bottleneck,kernel_size=[3,3],strides=strides,
                                       padding="VALID",name=name+"_depthwise",weight_decay=0.001)
        bottleneck=tf.layers.batch_normalization(bottleneck,training=self.is_training,
                                                 epsilon=1e-5,name=name+"_bn2")

        bottleneck_out=self.grouped_conv(bottleneck,self.group_num,bottleneck_output_channels,
                                     strides=[1,1],padding="VALID",is_activation=False,
                                     is_batch_norm=True,name=name+"_group2")
        # bottleneck_out = tf.layers.batch_normalization(bottleneck_out, training=self.is_training,
        #                                            epsilon=1e-5, name=name + "_bn3")

        if strides==[2,2]:
            residual_pooled=tf.nn.avg_pool(residual,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")
        else:
            residual_pooled=residual

        residual_out=residual_pooled

        if residual_type=="concat":
            unit_output = tf.concat([residual_out,bottleneck_out],axis=-1)
        else:
            residual_out = residual_pooled
            if residual_pooled.get_shape().as_list()[-1] != unit_output_channels:
                residual_out = self.conv2d(residual_out, kernel_size=[1, 1], filters_num=unit_output_channels,
                                           strides=[1, 1], padding="VALID", name="rematch_residual_channels")
            unit_output =residual_out+bottleneck_out

        unit_output=tf.nn.relu(unit_output)
        return unit_output

    def fully_connect(self,inputs,output_nodes_num,name):
        input_nodes_num=inputs.get_shape().as_list()[-1]
        weight_fc, biases_fc = self.get_variable_with_l2_loss(shape=[input_nodes_num, output_nodes_num],
                                                              weight_decay=0.001, name=name, )
        net = tf.matmul(inputs, weight_fc)
        net = tf.nn.relu(tf.nn.bias_add(net, biases_fc))
        return net

    #保证均值为0,范围在[-1,1]
    def preprocess(self,inputs):
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
        shape = preprocessed_inputs.get_shape().as_list()
        height, width, num_channels = shape[1:]

        net=self.conv2d(preprocessed_inputs,kernel_size=[3,3],filters_num=24,strides=[2,2],
                        padding="SAME",name="conv1")
        net=tf.nn.relu(net)
        net = tf.nn.max_pool(net,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="max_pool")

        #stage2
        net=self.shuffleNet_unit("stage2_out", net,unit_output_channels=240,
                                 strides=[2,2],is_use_group_shuffle=False)
        for i in range(3):
            net = self.shuffleNet_unit("stage2_in"+str(i), net, unit_output_channels=240,
                                       strides=[1, 1], is_use_group_shuffle=True,residual_type="add")

        # stage3
        net = self.shuffleNet_unit("stage3_out", net, unit_output_channels=480,
                                   strides=[2, 2], is_use_group_shuffle=True)
        for i in range(3):
            net = self.shuffleNet_unit("stage3_in" + str(i), net, unit_output_channels=480,
                                           strides=[1, 1], is_use_group_shuffle=True, residual_type="add")

        # stage4
        net = self.shuffleNet_unit("stage4_out", net, unit_output_channels=960,
                                   strides=[1, 1], is_use_group_shuffle=True)
        for i in range(3):
            net = self.shuffleNet_unit("stage4_in" + str(i), net, unit_output_channels=960,
                                           strides=[1, 1], is_use_group_shuffle=True, residual_type="add")

        net=tf.nn.avg_pool(net,ksize=[1,2,2,1],strides=[1,1,1,1],name="Global_pooling",padding="VALID")

        fc_conv=self.conv2d(net,kernel_size=[1,1],filters_num=self.num_classes,strides=[1,1],
                           padding="SAME",name="fc1")

        logits=tf.squeeze(fc_conv,axis=[1,2])
        # net=tf.squeeze(net,axis=[1,2])

        # net = self.fully_connect(net,256,"fc1")
        # logits = self.fully_connect(net,self.num_classes,"fc2")

        return logits

    def postprocess(self,logits):
        softmax = tf.nn.softmax(logits)
        classes = tf.cast(tf.argmax(softmax, axis=1), tf.int32)
        return softmax, classes

    def loss(self,logits,labels):

        softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=logits+1e-8,labels=labels),name="softmax_loss")
        regular_loss=tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),name="regular_loss")
        return softmax_loss








