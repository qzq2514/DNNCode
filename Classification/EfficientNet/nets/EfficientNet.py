import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim

class MBConvBlock(collections.namedtuple("MBConv",["scope","output_channels","ratio","kernel_size",
                                                   "downsanple_blocks","repeate_times"])):
    """

    """

#EfficientNet-B0
class EfficientNet(object):
    def __init__(self, is_training, num_classes):
        self.num_classes = num_classes
        self._is_training = is_training


    def EfficientNetB0_config(self):
        config=[
            MBConvBlock("block1", 16,  1, 3, False, 1),
            MBConvBlock("block2", 24,  6, 3, True, 2),
            MBConvBlock("block3", 40,  6, 5, True, 2),
            MBConvBlock("block4", 80,  6, 3, False, 3),
            MBConvBlock("block5", 112, 6, 5, True, 3),
            MBConvBlock("block6", 192, 6, 5, True, 4),
            MBConvBlock("block7", 320, 6, 3, False, 1),
        ]
        return config

    def preprocess(self, inputs):
        # ResNet暂不需要做输入预处理
        # preprocessed_inputs = tf.to_float(inputs)
        # preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        # preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return inputs

    def MBConv(self,inputs,output_channels,depth_ratio,kernel_size,is_downsample,name):
        stride=2 if is_downsample else 1
        with tf.variable_scope(name):
            inputs_channels=inputs.get_shape().as_list()[-1]
            bottlenect_channels=int(depth_ratio*inputs_channels)

            MBConv_net=slim.convolution2d(inputs,num_outputs=bottlenect_channels,
                                          kernel_size=1,stride=1,scope="1x1_conv1")
            MBConv_net=slim.separable_convolution2d(MBConv_net,num_outputs=None,kernel_size=kernel_size,
                                                    stride=stride,scope="DWConv")
            MBConv_net = slim.convolution2d(MBConv_net, num_outputs=output_channels,
                                            kernel_size=1, stride=1, scope="1x1_conv2")

            if not is_downsample and inputs_channels==output_channels:
                MBConv_net = MBConv_net+inputs
                MBConv_net = tf.identity(MBConv_net,name="out")
            return MBConv_net

    #inputs:[batch_size,224,224,3]
    def inference(self, inputs):
        with slim.arg_scope(self.EfficientNet_arg_scope(is_training=self._is_training)):
            net = slim.convolution2d(inputs,num_outputs=32,kernel_size=3,
                                     stride=2,scope="conv1")

            for block_id , blocks in enumerate(self.EfficientNetB0_config()):
                scope=blocks.scope
                output_channels=blocks.output_channels
                ratio=blocks.ratio
                kernel_size=blocks.kernel_size
                downsanple_blocks=blocks.downsanple_blocks
                repeate_times=blocks.repeate_times
                for unit_id in range(repeate_times):
                    net = self.MBConv(net,output_channels,ratio,kernel_size,
                                      downsanple_blocks and unit_id==0,"{}_{}".format(scope,unit_id+1))

            net = slim.convolution2d(net, num_outputs=1280,kernel_size=1,scope="conv2")

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

    def EfficientNet_arg_scope(self,is_training,weight_decay=0.0001,batch_norm_decay=0.95,
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
            normalizer_params=batch_norm_params,
                padding="SAME"):

            with slim.arg_scope([slim.batch_norm],**batch_norm_params) :
                with slim.arg_scope([slim.max_pool2d],padding="SAME")  as arg_sc:
                    return arg_sc