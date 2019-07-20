import collections
import tensorflow as tf
import tensorflow.contrib.slim as slim

class WideResNet(object):
    def __init__(self, is_training, num_classes,keep_prob,unit_num_in_block,width_k):
        self.num_classes = num_classes
        self._is_training = is_training
        self.keep_prob=keep_prob
        self.unit_num_in_block=unit_num_in_block
        self.width_k=width_k



    def preprocess(self, inputs):
        # ResNet暂不需要做输入预处理
        # preprocessed_inputs = tf.to_float(inputs)
        # preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        # preprocessed_inputs = tf.div(preprocessed_inputs, 128.0)
        return inputs

    def wide_residual_unit(self,inputs,unit_output_channels,is_downsamping,name):
        input_channel=inputs.get_shape().as_list()[-1]
        unit_stride=2 if is_downsamping else 1

        with tf.variable_scope(name+"/residual_branch"):
            residual_branch = slim.batch_norm(inputs,scope="bn1")
            residual_branch = slim.convolution2d(residual_branch,num_outputs=unit_output_channels,
                                               kernel_size=3,stride=unit_stride,scope="conv1")

            residual_branch = slim.dropout(residual_branch,keep_prob=self.keep_prob)

            residual_branch = slim.batch_norm(residual_branch, scope="bn2")
            residual_branch = slim.convolution2d(residual_branch, num_outputs=unit_output_channels,
                                                 kernel_size=3, stride=1, scope="conv2")
        identity_branch=inputs
        with tf.variable_scope(name+"/identity_branch"):
            if unit_stride!=1 or input_channel!=unit_output_channels:   #下采样
                identity_branch=slim.batch_norm(identity_branch,scope="bn")
                identity_branch=slim.convolution2d(identity_branch,num_outputs=unit_output_channels,
                                                   kernel_size=1,stride=unit_stride)

        return residual_branch+identity_branch


    #inputs:[batch_size,32,32,3]
    def inference(self, inputs):
        with slim.arg_scope(self.WideResNet_arg_scope(is_training=self._is_training)):
            net=slim.convolution2d(inputs,num_outputs=16,kernel_size=3,stride=1,scope="conv1")
            print("net:", net)

            with tf.variable_scope("block1"):
                for unit_id in range(self.unit_num_in_block):
                    net=self.wide_residual_unit(net,unit_output_channels=16*self.width_k,
                                                is_downsamping=False,name="unit{}".format(unit_id+1))
                    print("block1_net:",net)

            with tf.variable_scope("block2"):
                for unit_id in range(self.unit_num_in_block):
                    net=self.wide_residual_unit(net,unit_output_channels=32*self.width_k,
                                                is_downsamping = unit_id == 0,
                                                name="unit{}".format(unit_id+1))
                    print("block2_net:", net)

            with tf.variable_scope("block3"):
                for unit_id in range(self.unit_num_in_block):
                    net = self.wide_residual_unit(net, unit_output_channels=64 * self.width_k,
                                                  is_downsamping = unit_id == 0,
                                                  name="unit{}".format(unit_id + 1))
                    print("block2_net:", net)

            net=slim.batch_norm(net,scope="global_avg_pool/bn")

            global_pool=tf.reduce_mean(net,axis=[1,2],keepdims=True,name="global_avg_pool")

            logits=slim.convolution2d(global_pool,num_outputs=self.num_classes,kernel_size=1,
                                      stride=1,scope="fc")
            print("logits:",logits)

            logits=tf.squeeze(logits,axis=[1,2],name="squeeze")

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

    def WideResNet_arg_scope(self,is_training,weight_decay=0.0001,batch_norm_decay=0.90,
                         batch_norm_epsilon=1e-5,batch_norm_scale=True):

        batch_norm_params={
            'is_training':is_training,
            'decay':batch_norm_decay,
            'epsilon':batch_norm_epsilon,
            'scale':batch_norm_scale,
            "activation_fn":tf.nn.relu
            # 'updates_collections:':tf.GraphKeys.UPDATE_OPS
        }

        #DenseNet借鉴resNetV2,采用前置激活,不在卷积后进行bn和relu
        with slim.arg_scope(
            [slim.convolution2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=None):

            with slim.arg_scope([slim.batch_norm],**batch_norm_params) :
                with slim.arg_scope([slim.avg_pool2d],padding="SAME")  as arg_sc:
                    return arg_sc

            #最好使用这种方式,因为openvino转换是不支持dropout层,
            #这种参数空间的方式能保证is_training参数对slim.dropout也有效
            #这种方式生成的模型能直接转openvino,不用在调用freezing_graph.py
            # with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            #     with slim.arg_scope([slim.dropout],is_training=is_training):
            #         with slim.arg_scope([slim.avg_pool2d], padding="SAME")  as arg_sc:
            #             return arg_sc