import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim

class SqueezeNet(object):
    def __init__(self, is_training, num_classes):
        self.num_classes = num_classes
        self._is_training = is_training



    def preprocess(self, inputs):
        # ResNet暂不需要做输入预处理
        # preprocessed_inputs = tf.to_float(inputs)
        # preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        # preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return inputs

    def squeeze(self,inputs,output_channels):
        return slim.convolution2d(inputs,num_outputs=output_channels,kernel_size=1,
                                  stride=1,scope="squeeze")

    def expand(self,inputs,output_channels):
        with tf.variable_scope("expand"):
            expand_x1=slim.convolution2d(inputs,num_outputs=output_channels,
                                         kernel_size=1,stride=1,scope="1x1")
            expand_x3 = slim.convolution2d(inputs, num_outputs=output_channels,
                                           kernel_size=3,stride=1, scope="3x3")
        return tf.concat([expand_x1,expand_x3],axis=-1)

    def fire_moudle(self,inputs,squeeze_channels,expand_channels,name):
        with tf.variable_scope(name):
            with slim.arg_scope(self.SqueezeNet_arg_scope(is_training=self._is_training)):
                fire_net=self.squeeze(inputs,squeeze_channels)
                fire_net=self.expand(fire_net,expand_channels)
            return fire_net

    #inputs:[batch_size,224,224,3]
    def inference(self, inputs):

        # 以下为不包含bypass的squeeze网络
        net = slim.convolution2d(inputs,96,[7,7],stride=2,scope="conv1")

        net = slim.max_pool2d(net,[3,3],2,scope="max_pool1")

        net = self.fire_moudle(net, 16, 64, name="fire2")
        net = self.fire_moudle(net, 16, 64, name="fire3")
        net = self.fire_moudle(net, 32, 128, name="fire4")

        net = slim.max_pool2d(net, [3, 3], 2, scope="max_pool4")

        net = self.fire_moudle(net, 32, 128, name="fire5")
        net = self.fire_moudle(net, 48, 192, name="fire6")
        net = self.fire_moudle(net, 48, 192, name="fire7")
        net = self.fire_moudle(net, 64, 256, name="fire8")

        net = slim.max_pool2d(net, [2, 2], 2, scope="max_pool8")

        #原文中在fire9之后使用保留率为0.5的dropout,这里省略
        net = self.fire_moudle(net, 64, 256, name="fire9")

        net = slim.convolution2d(net, self.num_classes, [1, 1], activation_fn=None,
                                 normalizer_fn=None, scope="conv10")

        global_avg_pool = tf.reduce_mean(net,axis=[1,2],keepdims=True,name="global_pool")

        logits=tf.squeeze(global_avg_pool,axis=[1,2],name="squeeze")

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

    def SqueezeNet_arg_scope(self,is_training,weight_decay=0.0001,batch_norm_decay=0.90,
                         batch_norm_epsilon=1e-5,batch_norm_scale=True):

        batch_norm_params={
            'is_training':is_training,
            'decay':batch_norm_decay,
            'epsilon':batch_norm_epsilon,
            'scale':batch_norm_scale
            # 'updates_collections:':tf.GraphKeys.UPDATE_OPS
        }

        #DenseNet借鉴resNetV2,采用前置激活,不在卷积后进行bn和relu
        with slim.arg_scope(
            [slim.convolution2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params) as arg_sc:

            return arg_sc

            #最好使用这种方式,因为openvino转换是不支持dropout层,
            #这种参数空间的方式能保证is_training参数对slim.dropout也有效
            #这种方式生成的模型能直接转openvino,不用在调用freezing_graph.py
            # with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            #     with slim.arg_scope([slim.dropout],is_training=is_training):
            #         with slim.arg_scope([slim.avg_pool2d], padding="SAME")  as arg_sc:
            #             return arg_sc