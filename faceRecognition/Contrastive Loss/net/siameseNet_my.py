import tensorflow.contrib.slim as slim
import tensorflow as tf

class siameseNet():
    def __init__(self):
        pass

    def pre_process(self,inputs):
        preprocessed_inputs = tf.to_float(inputs)
        preprocessed_inputs = tf.subtract(preprocessed_inputs, 128.0)
        preprocessed_inputs = tf.div(preprocessed_inputs, 255.0)
        return preprocessed_inputs

    def inference(self,input):
        preprocessed_inputs=self.pre_process(input)

        # with slim.arg_scope(self.siamese_arg_scope(is_training=True)):
        net = preprocessed_inputs
        net = slim.convolution2d(net,32,3,1)
        net = slim.convolution2d(net, 32,3,1)
        net = slim.max_pool2d(net, 3, 2, padding="SAME")
        net = slim.convolution2d(net, 64,3,1)
        net = slim.convolution2d(net, 64,3,1)
        net = slim.max_pool2d(net, 3, 2, padding="SAME")
        net = slim.convolution2d(net, 128,3,1)
        net = slim.convolution2d(net, 128,3,1)

        # 展开成向量形式以供全连接
        flat_shape = net.get_shape().as_list()
        flat_height, flat_width, channals = flat_shape[1:]
        flat_size = flat_height * flat_width * channals
        net = tf.reshape(net, shape=[-1, flat_size])

        net=slim.fully_connected(net,512)
        net = slim.fully_connected(net, 512)
        net = slim.fully_connected(net, 10)

        return net

    def siamese_arg_scope(self,is_training,weight_decay=0.0001,batch_norm_decay=0.90,
                         batch_norm_epsilon=1e-5,batch_norm_scale=True):

        batch_norm_params={
            'is_training':is_training,
            'decay':batch_norm_decay,
            'epsilon':batch_norm_epsilon,
            'scale':batch_norm_scale,
            "activation_fn":tf.nn.relu
            # 'updates_collections:':tf.GraphKeys.UPDATE_OPS
        }

        with slim.arg_scope(
            [slim.convolution2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu):

            with slim.arg_scope([slim.batch_norm],**batch_norm_params) :
                with slim.arg_scope([slim.avg_pool2d,slim.max_pool2d],padding="SAME")  as arg_sc:
                    return arg_sc

    def loss(self,left_faltten,right_flatten,pair_label,margin):
        dist=tf.sqrt(tf.reduce_mean(tf.pow(left_faltten-right_flatten,2),1,keepdims=True))
        similarity=pair_label*tf.square(dist)
        disimilarity=(1-pair_label)*tf.square(tf.maximum(margin-dist,0))
        return tf.reduce_mean(similarity+disimilarity)/2