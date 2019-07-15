import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim

class ResNeXt_cifar(object):
    def __init__(self, is_training, num_classes,cardinality,bottleneck_type):
        self.num_classes = num_classes
        self._is_training = is_training
        self.cardinality=cardinality
        self.bottleneck_type=bottleneck_type


    def preprocess(self, inputs):
        # ResNet暂不需要做输入预处理
        # preprocessed_inputs = tf.to_float(inputs)
        # preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        # preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return inputs


    def bottleneck_MultiBranch(self,inputs,bottleneck_width,stride):
        branchs=[]
        for branch_id in range(self.cardinality):
            with tf.variable_scope("branch_{}".format(branch_id)):
                branch = slim.convolution2d(inputs,num_outputs=bottleneck_width,
                                   kernel_size=1,stride=1,scope="1x1conv")
                branch = slim.batch_norm(branch,scope="bn1")

                branch = slim.convolution2d(branch,num_outputs=bottleneck_width,
                                          kernel_size=3,stride=stride,scope="3x3conv")
                branch = slim.batch_norm(branch, scope="bn2")
            branchs.append(branch)
        concat_bottleneck = tf.concat(branchs,axis=3,name="concat")

        return concat_bottleneck

    def bottleneck_GroupConv(self,inputs,bottleneck_output_channel,stride):

        group_num=self.cardinality
        #先进行一次1x1卷积
        group_conv=slim.convolution2d(inputs,num_outputs=bottleneck_output_channel,kernel_size=1,
                                      stride=1,padding="SAME")
        group_conv = slim.batch_norm(group_conv, scope="bn1")
        input_channels = group_conv.get_shape().as_list()[-1]

        #对每组分配输入通道和输出通道数
        input_groups_channel = [input_channels // group_num] * group_num
        output_groups_channel = [bottleneck_output_channel // group_num] * group_num

        #防止组数溢出
        input_groups_channel[-1] = input_channels - input_groups_channel[0] * (group_num - 1)
        output_groups_channel[-1] = bottleneck_output_channel - output_groups_channel[0] * (group_num - 1)

        groups = []
        channels_start = 0

        for gooup_id in range(self.cardinality):
            with tf.variable_scope("branch_{}".format(gooup_id)):
                channels_end = channels_start + input_groups_channel[gooup_id]

                group = slim.convolution2d(group_conv[:,:,:,channels_start:channels_end],
                                           num_outputs=output_groups_channel[gooup_id],
                                   kernel_size=3,stride=stride,scope="3x3conv")
                group = slim.batch_norm(group,scope="bn1")

                groups.append(group)
                channels_start=channels_end

        concat_bottleneck = tf.concat(groups,axis=-1,name="concat")
        return concat_bottleneck

    def resNeXt_unit(self,inputs,unit_output_channel,scope_name):
        with tf.variable_scope(scope_name):
            input_channel=inputs.get_shape().as_list()[-1]
            bottleneck_width = unit_output_channel // 2 // self.cardinality
            if input_channel*2==unit_output_channel:
                stride=2    #该resNeXt单元进行下采样
            # 除了在下采样的resNeXt单元单元,其余resNeXt单元的输入和输出通道数都是相同的
            elif input_channel==unit_output_channel:
                stride=1
            else:
                raise ValueError("input_channel does not match to output_channel in resNeXt_unit!!")

            if self.bottleneck_type=="MultiBranch":
                concat_bottleneck=self.bottleneck_MultiBranch(inputs,bottleneck_width,stride)
            elif self.bottleneck_type=="GroupConv":
                concat_bottleneck = self.bottleneck_GroupConv(inputs, unit_output_channel // 2, stride)
            else:
                raise ValueError(
                    " Wrong bottleneck type-{},type must be one of ['MultiBranch','GroupConv']".format(self.bottleneck_type))

            #对连接后的
            bottleneck_1x1=slim.convolution2d(concat_bottleneck,num_outputs=unit_output_channel,
                                              kernel_size=1,stride=1)
            _input=inputs
            if stride==2:   #下采样时bottleneck输出和idendity mapping分支的特征图大小不一样
                            #并且会导致两者的通道数不一致
                downsample_input=slim.avg_pool2d(_input,kernel_size=2,stride=2,padding="VALID")
                #下采样时,idendity mapping分支的通道数input_channel和bottleneck分支的
                #通道数output_channel满足:output_channel=input_channel*2
                #所以为了保证后面能够对两个分支进行逐元素的相加,这里要对idendity mapping分支
                #在通道维度两侧各填充input_channel//2
                _input=tf.pad(downsample_input,[[0,0],[0,0],[0,0],[input_channel//2,input_channel//2]])

            unit_output=tf.nn.relu(bottleneck_1x1+_input)

        return unit_output

    # inputs-Cifar:[batch_size,32,32,3]
    def inference(self, inputs):
        print("Using {},cardinality={}".format(self.bottleneck_type,self.cardinality))

        with slim.arg_scope(self.ResNeXt_arg_scope(is_training=self._is_training)):
            net = slim.convolution2d(inputs,num_outputs=64,kernel_size=3,stride=1,padding="SAME",scope="conv1")

            for block_id,block_output_channel in enumerate([64,128,256]):
                for unit_id in range(3):
                    net=self.resNeXt_unit(net,block_output_channel,"unit{}_{}".format(block_id,unit_id))

            net_global_pool = tf.reduce_mean(net, [1, 2], name="global_pool", keep_dims=True)

            net = slim.convolution2d(net_global_pool, num_outputs=self.num_classes,
                                     kernel_size=1, activation_fn=None, normalizer_fn=None, scope="full_conv")

            logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            print("logits:", logits)

        return logits

    def postprocess(self,logits):
        softmax=tf.nn.softmax(logits)
        classes=tf.cast(tf.argmax(softmax,axis=1),tf.int32)
        return softmax,classes

    def loss(self,logits,labels):
        softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=logits+1e-8,labels=labels),name="softmax_loss")
        tf.add_to_collection("Loss",softmax_loss)
        loss_all=tf.add_n(tf.get_collection("Loss"),name="total_loss")
        return loss_all

    def ResNeXt_arg_scope(self,is_training,weight_decay=0.0001,batch_norm_decay=0.997,
                         batch_norm_epsilon=1e-5,batch_norm_scale=True):
        batch_norm_params={
            'is_training':is_training,
            'decay':batch_norm_decay,
            'epsilon':batch_norm_epsilon,
            'scale':batch_norm_scale,
            # 'updates_collections:':tf.GraphKeys.UPDATE_OPS
        }

        #resNetV2都是前置激活,不在卷积后进行bn和relu
        with slim.arg_scope(
            [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):

            with slim.arg_scope([slim.batch_norm],**batch_norm_params) :
                with slim.arg_scope([slim.max_pool2d],padding="SAME")  as arg_sc:
                    return arg_sc