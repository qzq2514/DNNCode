import tensorflow as tf
import numpy as np

class ShuffleNetV1(object):
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

        # print("result:",result)
        # print("biases:",biases)
        result = tf.nn.bias_add(result, biases)
        return result

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
        else:
            bottleneck = self.grouped_conv(inputs,output_channels=bottleneck_channels,padding="VALID",strides=[1,1],
                                         group_num=self.group_num,name=name+"_group1")

            bottleneck = tf.layers.batch_normalization(bottleneck, training=self.is_training, epsilon=1e-5,name=name+"batchnorm1")
            bottleneck = tf.nn.relu(bottleneck)

            # print("inputs5:", bottleneck)
            bottleneck=self.channel_shuffle(bottleneck,self.group_num,name=name+"_shuffle")
        # print("inputs4:", bottleneck)
        #长宽维度上两边各填充一个常数
        bottleneck=tf.pad(bottleneck,[[0,0],[1,1],[1,1],[0,0]],"CONSTANT")
        # print("inputs3:", bottleneck)
        bottleneck=self.depthwise_conv(bottleneck,kernel_size=[3,3],strides=strides,
                                       padding="VALID",name=name+"_depthwise")
        # print("inputs2:", bottleneck)
        bottleneck=tf.layers.batch_normalization(bottleneck,training=self.is_training,
                                                 epsilon=1e-5,name=name+"_batchnorm2")
        # print("inputs1:", bottleneck)
        bottleneck=self.grouped_conv(bottleneck,self.group_num,bottleneck_output_channels,
                                     strides=[1,1],padding="VALID",name=name+"_group2")
        bottleneck_out = tf.layers.batch_normalization(bottleneck, training=self.is_training,
                                                   epsilon=1e-5, name=name + "batchnorm3")

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
                                                              weight_decay=0.01, name=name, )
        net = tf.matmul(inputs, weight_fc)
        net = tf.nn.relu(tf.nn.bias_add(net, biases_fc))
        return net

    #保证均值为0,范围在[-1,1]
    def preprocess(self,inputs):
        # processed_inputs = tf.to_float(inputs)
        # processed_inputs = tf.subtract(processed_inputs,128.0)
        # processed_inputs = tf.div(processed_inputs, 128)
        return inputs

    def inference(self,preprocessed_inputs):
        shape = preprocessed_inputs.get_shape().as_list()
        height, width, num_channels = shape[1:]

        net=self.conv2d(preprocessed_inputs,kernel_size=[3,3],filters_num=24,strides=[2,2],
                        padding="SAME",name="conv1")
        net = tf.nn.max_pool(net,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME",name="max_pool")

        #stage2
        net=self.shuffleNet_unit("stage2_out", net,unit_output_channels=200,
                                 strides=[2,2],is_use_group_shuffle=False)
        for i in range(3):
            net = self.shuffleNet_unit("stage2_in"+str(i), net, unit_output_channels=200,
                                       strides=[1, 1], is_use_group_shuffle=True,residual_type="add")

        # stage3
        net = self.shuffleNet_unit("stage3_out", net, unit_output_channels=400,
                                   strides=[2, 2], is_use_group_shuffle=True)
        for i in range(3):
            net = self.shuffleNet_unit("stage3_in" + str(i), net, unit_output_channels=400,
                                           strides=[1, 1], is_use_group_shuffle=True, residual_type="add")

        # stage4
        net = self.shuffleNet_unit("stage4_out", net, unit_output_channels=800,
                                   strides=[2, 2], is_use_group_shuffle=True)
        for i in range(3):
            net = self.shuffleNet_unit("stage4_in" + str(i), net, unit_output_channels=800,
                                       strides=[1, 1], is_use_group_shuffle=True, residual_type="add")

        net=tf.nn.avg_pool(net,ksize=[1,7,7,1],strides=[1,1,1,1],name="Global_pooling",padding="VALID")

        net=tf.squeeze(net,axis=[1,2])

        net = self.fully_connect(net,256,"fc1")
        logits = self.fully_connect(net,self.num_classes,"fc2")

        return logits

    def postprocess(self,logits):
       pass

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








