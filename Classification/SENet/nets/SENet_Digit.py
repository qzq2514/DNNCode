import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim

class SENet_Digit(object):
    def __init__(self, is_training, num_classes,cardinality,reduction_ratio):
        self.num_classes = num_classes
        self._is_training = is_training
        self.cardinality=cardinality
        self.reduction_ratio=reduction_ratio


    def preprocess(self, inputs):
        # ResNet暂不需要做输入预处理
        # preprocessed_inputs = tf.to_float(inputs)
        # preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        # preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return inputs

    def SE_Moudle(self,inputs):
        num_channels=inputs.get_shape().as_list()[-1]

        with tf.variable_scope("SE_Moudle"):
            #Squeeze,通过全局池化得到全局信息
            moudle = tf.reduce_mean(inputs,axis=[1,2],keepdims=True,name="global_avg_pooling")

            #Excitation,这里使用全卷积充当全连接的作用,减少参数量,不改变计算量
            #通过全连接得到feature map 每个通道的权重
            #这也是SENet的核心
            moudle = slim.convolution2d(moudle,num_outputs=num_channels/self.reduction_ratio,
                                        kernel_size=1,stride=1,normalizer_fn=None,scope="dim_decrease")

            channel_weights = slim.convolution2d(moudle,num_outputs=num_channels,kernel_size=1,
                                        stride=1,activation_fn=tf.nn.sigmoid,normalizer_fn=None,
                                        scope="dim_increase")

            scale = inputs * channel_weights

            return scale

    def ResNeXt_bottleneck_MultiBranch(self,inputs,bottleneck_depth,stride):
        branchs=[]
        for branch_id in range(self.cardinality):
            with tf.variable_scope("branch_{}".format(branch_id)):
                branch = slim.convolution2d(inputs,num_outputs=bottleneck_depth,
                                   kernel_size=1,stride=1,scope="1x1conv")
                branch = slim.batch_norm(branch,scope="bn1")

                branch = slim.convolution2d(branch,num_outputs=bottleneck_depth,
                                          kernel_size=3,stride=stride)
                branch = slim.batch_norm(branch, scope="bn2")
            branchs.append(branch)

        concat_bottleneck = tf.concat(branchs,axis=3,name="concat")
        return concat_bottleneck

    def SE_ResNeXt_unit(self,inputs,unit_output_channel,downsample,scope_name):
        with tf.variable_scope(scope_name):
            input_channel=inputs.get_shape().as_list()[-1]
            stride=2 if downsample else 1   #该resNeXt单元进行下采时stride=2

            #分支数cardinality是固定的,但是bottleneck的通道数是不断改变的,两者和整个单元输出通道数有以下关系：
            bottleneck_depth=unit_output_channel//2//self.cardinality
            concat_bottleneck=self.ResNeXt_bottleneck_MultiBranch(inputs,bottleneck_depth,stride)

            #对连接后的的特征图再进行一次整体转换
            bottleneck_1x1=slim.convolution2d(concat_bottleneck,num_outputs=unit_output_channel,
                                              kernel_size=1,stride=1)

            #以ResNeXt作为baseline,同时使用SE_Moudle
            SE_moudle = self.SE_Moudle(bottleneck_1x1)

            _input=inputs
            if downsample:   #下采样时bottleneck输出和idendity mapping分支的特征图大小不一样
                _input=slim.avg_pool2d(_input,kernel_size=2,stride=2)

            if input_channel != unit_output_channel:
                _input=slim.convolution2d(_input,num_outputs=unit_output_channel,kernel_size=1,stride=1,
                                          scope="matchDim")

            unit_output=tf.nn.relu(SE_moudle+_input,name="Addition_relu")
        return unit_output

    #inputs:[batch_size,224,224,3]
    def inference(self, inputs):
        with slim.arg_scope(self.SE_ResNeXt_arg_scope(is_training=self._is_training)):
            net = slim.convolution2d(inputs,num_outputs=64,kernel_size=3,stride=2)
            # net = slim.max_pool2d(net,kernel_size=3,stride=2,padding="SAME")

            scope_name_format = "unit{}_{}"

            for unit_id in range(3):
                net = self.SE_ResNeXt_unit(net, 64, False,
                                        scope_name_format.format(1,unit_id))

            for unit_id in range(4):
                net = self.SE_ResNeXt_unit(net, 128, unit_id == 0,
                                        scope_name_format.format(2,unit_id))

            for unit_id in range(6):
                net = self.SE_ResNeXt_unit(net, 256, unit_id == 0,
                                        scope_name_format.format(3,unit_id))

            print("net:",net)

            net_global_pool = tf.reduce_mean(net, [1, 2],name="global_pool",keep_dims=True)
            net = slim.convolution2d(net_global_pool, num_outputs=self.num_classes,
                                     kernel_size=1, activation_fn=None, normalizer_fn=None, scope="full_conv")

            logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
            
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

    def SE_ResNeXt_arg_scope(self,is_training,weight_decay=0.0001,batch_norm_decay=0.95,
                         batch_norm_epsilon=1e-5,batch_norm_scale=True):
        batch_norm_params={
            'is_training':is_training,
            'decay':batch_norm_decay,
            'epsilon':batch_norm_epsilon,
            'scale':batch_norm_scale,
            # 'updates_collections:':tf.GraphKeys.UPDATE_OPS
        }

        with slim.arg_scope(
            [slim.convolution2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):

            with slim.arg_scope([slim.batch_norm],**batch_norm_params) :
                with slim.arg_scope([slim.max_pool2d,slim.avg_pool2d],padding="SAME")  as arg_sc:
                    return arg_sc