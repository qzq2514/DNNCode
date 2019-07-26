import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim


class MobileNetV3_block(collections.namedtuple("MobileNetV3_block",
                                   ["scope","kernel_size","bottleneck_channels",
                                    "block_output_channels","stride","se_moudle","h_wish"])):
    '''

    '''

class MobileNetV3(object):
    def __init__(self, is_training, num_classes):
        self.num_classes = num_classes
        self._is_training = is_training


    def preprocess(self, inputs):
        # ResNet暂不需要做输入预处理
        # preprocessed_inputs = tf.to_float(inputs)
        # preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        # preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return inputs

    def MobileNetV3_small_config(self):
        config=[
            MobileNetV3_block("block1", 3,  16,   16, 2,True,False),
            MobileNetV3_block("block2", 3,  72,   24, 2, False, False),
            MobileNetV3_block("block3", 3,  88,   24, 1, False, False),
            MobileNetV3_block("block4", 5,  96,   40, 1, True, True),
            MobileNetV3_block("block5", 5,  240,  40, 1, True, True),
            MobileNetV3_block("block6", 5,  240,  40, 1, True, True),
            MobileNetV3_block("block7", 5,  120,  48, 1, True, True),
            MobileNetV3_block("block8", 5,  144,  48, 1, True, True),
            MobileNetV3_block("block9", 5,  288,  96, 2, True, True),
            MobileNetV3_block("block10", 5,  576,  96, 1, True, True),
            MobileNetV3_block("block11", 5,  576,  96, 1, True, True),
        ]
        return config

    def hard_sigmod(self,inputs):
        with tf.variable_scope("hard_sigmod"):
            #还有中说法是:如果 -2.5 <= inputs <= 2.5，返回 0.2 * inputs + 0.5
            h_sigmod=tf.nn.relu6(inputs+3)/6
        return h_sigmod

    def hard_swish(self,inputs):
        with tf.variable_scope("hard_swish"):
            h_swish=inputs*tf.nn.relu6(inputs+3)/6
        return h_swish

    def SE_Moudle(self,inputs,ratio):
        num_channels=inputs.get_shape().as_list()[-1]

        with tf.variable_scope("SE_Moudle"):

            moudle = tf.reduce_mean(inputs,axis=[1,2],keepdims=True,name="global_avg_pooling")

            moudle = slim.convolution2d(moudle,num_outputs=int(num_channels/ratio),
                                        kernel_size=1,stride=1,normalizer_fn=None,scope="dim_decrease")
            moudle = tf.nn.relu6(moudle)

            channel_weights = slim.convolution2d(moudle,num_outputs=num_channels,kernel_size=1,
                                        stride=1,normalizer_fn=None,scope="dim_increase")

            channel_weights=self.hard_sigmod(channel_weights)

            scale = inputs * channel_weights
            return scale

    def MobileNetV3_bolck(self,inputs,kernel_size,bottleneck_channels,block_output_channels,stride,h_wish,se_moudle,name,SE_ratio=16):
        with tf.variable_scope(name):
            block_net = slim.convolution2d(inputs,num_outputs=bottleneck_channels,
                                           kernel_size=1,stride=1)
            if h_wish:
                block_net=self.hard_swish(block_net)
            else:
                block_net=tf.nn.relu6(block_net)

            block_net=slim.separable_convolution2d(block_net, num_outputs=None,
                                                   kernel_size=kernel_size,stride=stride)

            if h_wish:
                block_net=self.hard_swish(block_net)
            else:
                block_net=tf.nn.relu6(block_net)

            #SE_moudle:squeeze and excitation
            if se_moudle:
                block_net=self.SE_Moudle(block_net,ratio=SE_ratio)


            #point wise
            block_net=slim.convolution2d(block_net,num_outputs=block_output_channels,
                                         kernel_size=1,stride=1)

            #element wise add,onle for stride=1
            input_channels=inputs.get_shape().as_list()[-1]
            if stride==1 and input_channels==block_output_channels:
                block_net=block_net+inputs
                block_net=tf.identity(block_net,name="output")

        return block_net

    #inputs:[batch_size,224,224,3]
    def inference(self, inputs):
        with slim.arg_scope(self.MobileNetV3_arg_scope(is_training=self._is_training)):
            net = slim.convolution2d(inputs,num_outputs=16,kernel_size=3,stride=2,
                                     padding="SAME")
            net=self.hard_swish(net)

            for block in self.MobileNetV3_digit_config():
                name=block.scope
                kernel_size=block.kernel_size
                bottleneck_channels=block.bottleneck_channels
                block_output_channels=block.block_output_channels
                stride=block.stride
                se_moudle=block.se_moudle
                h_wish=block.h_wish
                net=self.MobileNetV3_bolck(net,kernel_size,bottleneck_channels,
                                           block_output_channels,stride,h_wish,se_moudle,name)

            print("net:",net)
            net = slim.convolution2d(net, num_outputs=576,
                                     kernel_size=1, scope="full_conv")
            net=self.hard_swish(net)

            net_global_pool = tf.reduce_mean(net, [1, 2],name="global_pool",keep_dims=True)

            net_global_pool = self.hard_swish(net_global_pool)

            net = slim.convolution2d(net_global_pool, num_outputs=1280,
                                     kernel_size=1, )
            net = self.hard_swish(net)

            net = slim.convolution2d(net, num_outputs=self.num_classes,
                                     kernel_size=1, normalizer_fn=None)

            logits = tf.squeeze(net, [1, 2], name='SpatialSqueeze')

            print("logits:",logits)

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

    def MobileNetV3_arg_scope(self,is_training,weight_decay=0.0001,batch_norm_decay=0.997,
                         batch_norm_epsilon=1e-5,batch_norm_scale=True):
        batch_norm_params={
            'is_training':is_training,
            'decay':batch_norm_decay,
            'epsilon':batch_norm_epsilon,
            'scale':batch_norm_scale,
            # 'updates_collections:':tf.GraphKeys.UPDATE_OPS
        }

        with slim.arg_scope(
            [slim.convolution2d,slim.separable_convolution2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=None,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):

            with slim.arg_scope([slim.batch_norm],**batch_norm_params) :
                with slim.arg_scope([slim.max_pool2d],padding="SAME")  as arg_sc:
                    return arg_sc